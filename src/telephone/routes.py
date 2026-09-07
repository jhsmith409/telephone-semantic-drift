# Copyright (c) 2026 James H. Smith. MIT License.
"""Flask route handlers."""

from __future__ import annotations

import html
import json
import logging
import os
import re
from functools import wraps

from flask import Blueprint, Response, jsonify, render_template, request

from telephone.discovery import (
    DiscoveredService,
    discover_services,
    load_hosts,
)
from telephone.embeddings import compute_similarity
from telephone.game_logic import (
    PROMPT_STYLES,
    AgentConfig,
    RelayStep,
    compute_diff,
    export_chain,
    run_chain,
    run_judge,
)

bp = Blueprint("telephone", __name__)
logger = logging.getLogger(__name__)


@bp.before_request
def check_content_type():
    if request.method == "POST" and request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 415

# ─── Server-side Service Cache ────────────────────────────────

_cached_services: list[DiscoveredService] = []

# ─── Authentication ───────────────────────────────────────────


def require_auth(f):
    """Require Bearer token if APP_API_KEY is configured in .env."""

    @wraps(f)
    def decorated(*args, **kwargs):
        api_key = os.environ.get("APP_API_KEY", "").strip()
        if not api_key:
            return f(*args, **kwargs)
        auth = request.headers.get("Authorization", "")
        if auth == f"Bearer {api_key}":
            return f(*args, **kwargs)
        return jsonify({"error": "Unauthorized"}), 401

    return decorated


# ─── Input Sanitization ──────────────────────────────────────

# Matches HTML tags, script injections, event handlers, etc.
_DANGEROUS_PATTERN = re.compile(
    r"<[^>]*>|javascript:|vbscript:|data:\w+/\w+|expression\s*\(|on\w+\s*=", re.IGNORECASE
)


def sanitize_text(value: str, max_length: int = 5000) -> str:
    """Sanitize a user-provided text input.

    - Strips leading/trailing whitespace
    - Truncates to max_length
    - Escapes HTML entities
    """
    value = value.strip()[:max_length]
    value = html.escape(value, quote=True)
    return value


def validate_text(value: str) -> str | None:
    """Return an error message if the raw input contains dangerous patterns,
    or None if it's safe."""
    if _DANGEROUS_PATTERN.search(value):
        return "Input contains disallowed HTML or script content"
    return None


def sanitize_and_validate(value: str, field_name: str, max_length: int = 5000) -> tuple[str, str | None]:
    """Sanitize and validate a field. Returns (clean_value, error_or_None)."""
    error = validate_text(value)
    if error:
        return "", f"{field_name}: {error}"
    return sanitize_text(value, max_length), None


# ─── Service Resolution ──────────────────────────────────────


def resolve_service(service_index: int) -> tuple[DiscoveredService | None, str | None]:
    """Look up a cached service by index. Returns (service, error_or_None)."""
    if not isinstance(service_index, int) or service_index < 0:
        return None, "invalid service_index"
    if service_index >= len(_cached_services):
        return None, f"service_index {service_index} out of range"
    return _cached_services[service_index], None


def build_agent_from_request(
    data: dict,
    label: str,
    prompt_style: str = "",
    custom_prompt: str = "",
    temperature: float | None = None,
) -> tuple[AgentConfig | None, str | None]:
    """Build an AgentConfig from a client request dict using server-side service lookup.

    The client sends only {service_index, model} — all connection details
    (host, port, api_key) are resolved server-side from the cached services.
    """
    si = data.get("service_index")
    if si is None:
        return None, f"{label}: service_index is required"

    svc, err = resolve_service(si)
    if err:
        return None, f"{label}: {err}"

    model = data.get("model", "")
    if not model:
        return None, f"{label}: model is required"
    if model not in svc.models:
        return None, f"{label}: model not available on this service"

    return AgentConfig(
        service_type=svc.service_type,
        host=svc.host,
        port=svc.port,
        api_key="",
        model=model,
        prompt_style=prompt_style,
        custom_prompt=custom_prompt,
        temperature=temperature,
    ), None


# ─── Routes ──────────────────────────────────────────────────

@bp.route("/")
def index():
    """Serve the single-page app."""
    return render_template("index.html")


@bp.route("/api/services")
@require_auth
def api_services():
    """Load hosts from .env, probe for AI services, return discovered services + models.

    Caches discovered services server-side. The client receives service indices
    to reference services without needing host/port/api_key details.
    """
    global _cached_services
    try:
        hosts = load_hosts()
        if not hosts:
            return jsonify({
                "services": [],
                "warning": "No AI_HOSTS configured in .env file",
            })

        services = discover_services(hosts)
        _cached_services = services
        return jsonify({
            "services": [
                {
                    "index": i,
                    "host": s.host,
                    "port": s.port,
                    "service_type": s.service_type,
                    "models": s.models,
                }
                for i, s in enumerate(services)
            ],
        })
    except Exception as exc:
        logger.exception("Service discovery failed")
        return jsonify({"error": "Service discovery failed"}), 500


@bp.route("/api/start", methods=["POST"])
@require_auth
def api_start():
    """Start a game. Returns SSE stream of relay steps + final diff + optional judge.

    Agents are specified by {service_index, model} — connection details are
    resolved server-side from the cached service list.
    """
    data = request.get_json(silent=True) or {}
    raw_message = data.get("message", "")
    agents_data = data.get("agents", [])
    judge_data = data.get("judge")  # optional
    prompt_style = data.get("prompt_style", "paraphrase")
    custom_prompt = data.get("custom_prompt", "")

    # Validate prompt_style
    if prompt_style not in PROMPT_STYLES:
        return jsonify({"error": f"invalid prompt_style"}), 400

    # Parse optional temperature
    raw_temp = data.get("temperature")
    temperature: float | None = None
    if raw_temp is not None and raw_temp != "":
        try:
            temperature = float(raw_temp)
            if not (0.0 <= temperature <= 1.0):
                return jsonify({"error": "temperature must be between 0 and 1"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "temperature must be a number"}), 400

    # Validate initial message
    message, err = sanitize_and_validate(raw_message, "message", max_length=10000)
    if err:
        return jsonify({"error": err}), 400
    if not message:
        return jsonify({"error": "message is required"}), 400

    # Validate custom prompt if present
    if custom_prompt:
        custom_prompt, err = sanitize_and_validate(custom_prompt, "custom_prompt", max_length=1000)
        if err:
            return jsonify({"error": err}), 400

    if not agents_data:
        return jsonify({"error": "at least one agent is required"}), 400
    if len(agents_data) > 30:
        return jsonify({"error": "maximum 30 agents allowed"}), 400

    if not _cached_services:
        return jsonify({"error": "no services loaded — refresh services first"}), 400

    agents = []
    for i, a in enumerate(agents_data):
        agent, err = build_agent_from_request(
            a, f"agent {i + 1}",
            prompt_style=prompt_style,
            custom_prompt=custom_prompt,
            temperature=temperature,
        )
        if err:
            return jsonify({"error": err}), 400
        agents.append(agent)

    # Parse judge config if provided
    judge_config = None
    if judge_data:
        judge_config, err = build_agent_from_request(
            judge_data, "judge",
            prompt_style="judge",
            temperature=temperature,
        )
        if err:
            return jsonify({"error": err}), 400

    def generate():
        steps: list[RelayStep] = []
        try:
            for step in run_chain(message, agents):
                steps.append(step)
                sim = compute_similarity(message, step.output_message)
                step_data = step.to_dict()
                step_data["cosine_similarity"] = sim
                yield f"data: {json.dumps({'type': 'step', 'step': step_data})}\n\n"
        except Exception as exc:
            logger.exception("Chain execution error")
            yield f"data: {json.dumps({'type': 'error', 'error': 'An error occurred during chain execution'})}\n\n"
            return

        # Send diff
        if steps:
            diff = compute_diff(message, steps[-1].output_message)
            yield f"data: {json.dumps({'type': 'diff', 'diff': diff})}\n\n"

        # Run judge if configured
        if judge_config and steps:
            yield f"data: {json.dumps({'type': 'judge_start'})}\n\n"
            try:
                analysis = run_judge(message, steps, judge_config)
                yield f"data: {json.dumps({'type': 'judge', 'analysis': analysis})}\n\n"
            except Exception as exc:
                logger.exception("Judge execution error")
                yield f"data: {json.dumps({'type': 'judge_error', 'error': 'Judge analysis failed'})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return Response(generate(), mimetype="text/event-stream")


@bp.route("/api/export", methods=["POST"])
@require_auth
def api_export():
    """Export chain as text file download."""
    data = request.get_json(silent=True) or {}
    initial = data.get("message", "")
    steps_data = data.get("steps", [])

    steps = []
    for i, s in enumerate(steps_data):
        try:
            steps.append(RelayStep(
                agent_index=s["agent_index"],
                model=s["model"],
                prompt_style=s["prompt_style"],
                input_message=s["input_message"],
                output_message=s["output_message"],
            ))
        except (KeyError, TypeError):
            return jsonify({"error": f"malformed step at index {i}"}), 400

    judge_analysis = data.get("judge_analysis", "")
    text = export_chain(initial, steps, judge_analysis)
    return Response(
        text,
        mimetype="text/plain",
        headers={"Content-Disposition": "attachment; filename=telephone_chain.txt"},
    )
