# Copyright (c) 2026 James H. Smith. MIT License.
"""Batch testing API server — runs on port 5001, separate from the main GUI."""

from __future__ import annotations

import logging
import os
import time
from functools import wraps

from dotenv import load_dotenv
from flask import Blueprint, Flask, jsonify, request

from telephone.discovery import discover_services, load_hosts
from telephone.embeddings import compute_similarity
from telephone.game_logic import (
    PROMPT_STYLES,
    AgentConfig,
    compute_diff,
    run_chain,
)

logger = logging.getLogger(__name__)

VALID_SERVICE_TYPES = {"ollama", "openai"}

bp = Blueprint("batch", __name__)


@bp.before_request
def check_content_type():
    if request.method == "POST" and request.content_type != "application/json":
        return jsonify({"error": "Content-Type must be application/json"}), 415


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


@bp.route("/api/services")
@require_auth
def api_services():
    """Discover and return AI services with connection details."""
    try:
        hosts = load_hosts()
        if not hosts:
            return jsonify({
                "services": [],
                "warning": "No AI_HOSTS configured in .env file",
            })

        services = discover_services(hosts)
        return jsonify({
            "services": [s.to_dict() for s in services],
        })
    except Exception:
        logger.exception("Service discovery failed")
        return jsonify({"error": "Service discovery failed"}), 500


@bp.route("/api/run", methods=["POST"])
@require_auth
def api_run():
    """Run a single-agent test synchronously and return the result as JSON."""
    data = request.get_json(silent=True) or {}

    # Validate required fields
    required = ["message", "model", "service_type", "host", "port", "prompt_style"]
    missing = [f for f in required if f not in data or data[f] == ""]
    if missing:
        return jsonify({"error": f"missing required fields: {', '.join(missing)}"}), 400

    message = data["message"]
    model = data["model"]
    service_type = data["service_type"]
    host = data["host"]
    port = data["port"]
    prompt_style = data["prompt_style"]
    temperature = data.get("temperature")
    system_prompt = data.get("system_prompt", "")
    seed = data.get("seed")

    # Validate service_type
    if service_type not in VALID_SERVICE_TYPES:
        return jsonify({"error": f"invalid service_type: {service_type}"}), 400

    # Validate prompt_style
    if prompt_style not in PROMPT_STYLES:
        return jsonify({"error": f"invalid prompt_style: {prompt_style}"}), 400

    # Validate port is an integer
    try:
        port = int(port)
    except (ValueError, TypeError):
        return jsonify({"error": "port must be an integer"}), 400

    # Parse temperature if provided
    if temperature is not None and temperature != "":
        try:
            temperature = float(temperature)
        except (ValueError, TypeError):
            return jsonify({"error": "temperature must be a number"}), 400

    # Parse seed if provided
    if seed is not None and seed != "":
        try:
            seed = int(seed)
        except (ValueError, TypeError):
            return jsonify({"error": "seed must be an integer"}), 400

    agent = AgentConfig(
        service_type=service_type,
        host=host,
        port=port,
        api_key="",
        model=model,
        prompt_style=prompt_style,
        temperature=temperature,
        system_prompt=system_prompt,
        seed=seed,
    )

    start = time.time()
    try:
        steps = list(run_chain(message, [agent]))
        output_message = steps[-1].output_message
        diff = compute_diff(message, output_message)
        elapsed = round(time.time() - start, 2)

        # Compute semantic similarity (None if unavailable)
        similarity = compute_similarity(message, output_message)
        embedding_model = os.environ.get("EMBEDDING_MODEL", "").strip() or None

        return jsonify({
            "status": "success",
            "input_message": message,
            "output_message": output_message,
            "model": model,
            "prompt_style": prompt_style,
            "temperature": temperature,
            "service_type": service_type,
            "host": host,
            "port": port,
            "diff": diff,
            "cosine_similarity": similarity,
            "embedding_model": embedding_model if similarity is not None else None,
            "elapsed_seconds": elapsed,
        })
    except Exception as exc:
        elapsed = round(time.time() - start, 2)
        logger.exception("Batch run failed")
        return jsonify({
            "status": "error",
            "error": str(exc),
            "input_message": message,
            "model": model,
            "prompt_style": prompt_style,
            "temperature": temperature,
            "service_type": service_type,
            "host": host,
            "port": port,
            "elapsed_seconds": elapsed,
        }), 500


def create_batch_app() -> Flask:
    """Create the batch API Flask application."""
    load_dotenv()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024  # 1 MB

    app.register_blueprint(bp)
    return app


def main() -> None:
    app = create_batch_app()
    app.run(host="0.0.0.0", port=5001)


if __name__ == "__main__":
    main()
