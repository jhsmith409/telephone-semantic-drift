"""Tests for input validation, sanitization, auth, and service resolution in routes."""

from __future__ import annotations

import json
import os
from unittest.mock import patch

import pytest

from telephone.discovery import DiscoveredService
from telephone.main import create_app
from telephone.routes import (
    _cached_services,
    build_agent_from_request,
    resolve_service,
    sanitize_and_validate,
    sanitize_text,
    validate_text,
)


# ─── Sanitization Tests ──────────────────────────────────────


def test_validate_text_clean():
    assert validate_text("Hello world") is None
    assert validate_text("The cat sat on the mat.") is None
    assert validate_text("Use 2 < 3 comparison") is None  # no closing >


def test_validate_text_html_tags():
    assert validate_text("<script>alert('xss')</script>") is not None
    assert validate_text("<img src=x>") is not None
    assert validate_text("Hello <b>bold</b>") is not None


def test_validate_text_javascript_protocol():
    assert validate_text("javascript:alert(1)") is not None


def test_validate_text_event_handlers():
    assert validate_text('onerror=alert(1)') is not None
    assert validate_text('onload = doStuff()') is not None


def test_sanitize_text_escapes_html():
    result = sanitize_text('<script>alert("xss")</script>')
    assert "<script>" not in result
    assert "&lt;" in result


def test_sanitize_text_truncates():
    long = "a" * 10000
    result = sanitize_text(long, max_length=100)
    assert len(result) == 100


def test_sanitize_text_strips():
    assert sanitize_text("  hello  ") == "hello"


def test_sanitize_and_validate_clean():
    value, err = sanitize_and_validate("Hello world", "test")
    assert err is None
    assert value == "Hello world"


def test_sanitize_and_validate_dangerous():
    value, err = sanitize_and_validate("<script>xss</script>", "test")
    assert err is not None
    assert "test" in err


# ─── Service Resolution Tests ────────────────────────────────


@pytest.fixture()
def fake_services():
    """Populate the cached services list for testing."""
    import telephone.routes as routes_mod

    original = routes_mod._cached_services[:]
    routes_mod._cached_services.clear()
    routes_mod._cached_services.extend([
        DiscoveredService(
            host="<LAN-HOST-1>", port=11434,
            service_type="ollama", models=["llama3:latest", "mistral:latest"],
        ),
        DiscoveredService(
            host="<LAN-HOST-2>", port=8000,
            service_type="openai", models=["gpt-4"],
        ),
    ])
    yield routes_mod._cached_services
    routes_mod._cached_services.clear()
    routes_mod._cached_services.extend(original)


def test_resolve_service_valid(fake_services):
    svc, err = resolve_service(0)
    assert err is None
    assert svc.host == "<LAN-HOST-1>"

    svc, err = resolve_service(1)
    assert err is None
    assert svc.host == "<LAN-HOST-2>"


def test_resolve_service_out_of_range(fake_services):
    svc, err = resolve_service(99)
    assert svc is None
    assert "out of range" in err


def test_resolve_service_negative(fake_services):
    svc, err = resolve_service(-1)
    assert svc is None
    assert "invalid" in err


def test_resolve_service_not_int(fake_services):
    svc, err = resolve_service("abc")
    assert svc is None
    assert "invalid" in err


def test_build_agent_valid(fake_services):
    agent, err = build_agent_from_request(
        {"service_index": 0, "model": "llama3:latest"},
        "test agent",
        prompt_style="paraphrase",
    )
    assert err is None
    assert agent.host == "<LAN-HOST-1>"
    assert agent.port == 11434
    assert agent.model == "llama3:latest"
    assert agent.service_type == "ollama"


def test_build_agent_invalid_model(fake_services):
    agent, err = build_agent_from_request(
        {"service_index": 0, "model": "nonexistent"},
        "test agent",
    )
    assert agent is None
    assert "not available" in err


def test_build_agent_missing_service_index(fake_services):
    agent, err = build_agent_from_request(
        {"model": "llama3:latest"},
        "test agent",
    )
    assert agent is None
    assert "service_index is required" in err


def test_build_agent_missing_model(fake_services):
    agent, err = build_agent_from_request(
        {"service_index": 0},
        "test agent",
    )
    assert agent is None
    assert "model is required" in err


# ─── Auth Tests ──────────────────────────────────────────────


@pytest.fixture()
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


def test_no_auth_required_when_no_key(client, fake_services):
    """When APP_API_KEY is not set, endpoints are accessible without auth."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.get("/api/services")
        assert resp.status_code == 200


def test_auth_required_when_key_set(client, fake_services):
    """When APP_API_KEY is set, requests without valid Bearer token get 401."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False):
        resp = client.get("/api/services")
        assert resp.status_code == 401
        data = resp.get_json()
        assert data["error"] == "Unauthorized"


def test_auth_succeeds_with_correct_key(client, fake_services):
    """When APP_API_KEY is set, valid Bearer token grants access."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False):
        resp = client.get(
            "/api/services",
            headers={"Authorization": "Bearer secret123"},
        )
        assert resp.status_code == 200


def test_auth_rejects_wrong_key(client, fake_services):
    """Wrong Bearer token returns 401."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False):
        resp = client.get(
            "/api/services",
            headers={"Authorization": "Bearer wrongkey"},
        )
        assert resp.status_code == 401


def test_index_page_no_auth(client):
    """The index page does not require auth (it's static HTML)."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False):
        resp = client.get("/")
        assert resp.status_code == 200


# ─── API Start SSRF Prevention Tests ─────────────────────────


def test_start_rejects_raw_host_port(client, fake_services):
    """Agents must use service_index, not raw host/port."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/start", json={
            "message": "hello",
            "agents": [{
                "host": "169.254.169.254",
                "port": 80,
                "service_type": "openai",
                "model": "anything",
            }],
            "prompt_style": "paraphrase",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "service_index" in data["error"]


def test_start_validates_service_index(client, fake_services):
    """Out-of-range service_index is rejected."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/start", json={
            "message": "hello",
            "agents": [{"service_index": 999, "model": "llama3:latest"}],
            "prompt_style": "paraphrase",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "out of range" in data["error"]


def test_start_validates_model_on_service(client, fake_services):
    """Model must exist on the referenced service."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/start", json={
            "message": "hello",
            "agents": [{"service_index": 0, "model": "nonexistent-model"}],
            "prompt_style": "paraphrase",
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "not available" in data["error"]


# ─── CSRF Protection Tests ───────────────────────────────────


def test_post_without_json_content_type_returns_415(client):
    """POST requests without application/json Content-Type are rejected."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post(
            "/api/start",
            data="message=hello",
            content_type="application/x-www-form-urlencoded",
        )
        assert resp.status_code == 415
        data = resp.get_json()
        assert "Content-Type" in data["error"]


def test_post_with_json_content_type_passes_csrf_check(client, fake_services):
    """POST requests with application/json Content-Type pass the CSRF check."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.routes.run_chain", return_value=iter([])):
        resp = client.post("/api/start", json={
            "message": "hello",
            "agents": [{"service_index": 0, "model": "llama3:latest"}],
            "prompt_style": "paraphrase",
        })
        # Should pass CSRF check (may fail on other validation, but not 415)
        assert resp.status_code != 415


# ─── Expanded Regex Tests ────────────────────────────────────


def test_validate_text_vbscript():
    assert validate_text("vbscript:msgbox") is not None


def test_validate_text_data_uri():
    assert validate_text("data:text/html,<script>alert(1)</script>") is not None


def test_validate_text_css_expression():
    assert validate_text("expression(alert(1))") is not None
    assert validate_text("expression (alert(1))") is not None


def test_validate_text_clean_data_word():
    """The word 'data' in normal text should not trigger the pattern."""
    assert validate_text("send data to the server") is None


# ─── Export Malformed Input Tests ────────────────────────────


def test_export_malformed_step_returns_400(client):
    """Malformed step data in export returns 400, not 500."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/export", json={
            "message": "hello",
            "steps": [{"bad_key": "value"}],
        })
        assert resp.status_code == 400
        data = resp.get_json()
        assert "malformed step" in data["error"]


def test_export_empty_step_returns_400(client):
    """A null step in the list returns 400."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/export", json={
            "message": "hello",
            "steps": [None],
        })
        assert resp.status_code == 400


def test_export_valid_step_succeeds(client):
    """A well-formed export request succeeds."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/export", json={
            "message": "hello",
            "steps": [{
                "agent_index": 0,
                "model": "llama3",
                "prompt_style": "paraphrase",
                "input_message": "hello",
                "output_message": "hi there",
            }],
        })
        assert resp.status_code == 200


# ─── Security Headers Tests ─────────────────────────────────


def test_response_has_security_headers(client):
    """Every response should include CSP, X-Content-Type-Options, X-Frame-Options."""
    resp = client.get("/")
    assert resp.headers.get("Content-Security-Policy") is not None
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"


# ─── Similarity in SSE Events Tests ─────────────────────────


def test_step_event_includes_cosine_similarity(client, fake_services):
    """Step SSE events include cosine_similarity from the embedding service."""
    from telephone.game_logic import RelayStep

    fake_step = RelayStep(
        agent_index=0,
        model="llama3:latest",
        prompt_style="paraphrase",
        input_message="hello",
        output_message="hi there",
    )

    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.routes.run_chain", return_value=iter([fake_step])), \
         patch("telephone.routes.compute_similarity", return_value=0.85):
        resp = client.post("/api/start", json={
            "message": "hello",
            "agents": [{"service_index": 0, "model": "llama3:latest"}],
            "prompt_style": "paraphrase",
        })
        assert resp.status_code == 200

        # Parse SSE events from the response
        data = resp.get_data(as_text=True)
        step_event = None
        for line in data.splitlines():
            if line.startswith("data: "):
                payload = json.loads(line[6:])
                if payload.get("type") == "step":
                    step_event = payload
                    break

        assert step_event is not None
        assert step_event["step"]["cosine_similarity"] == 0.85
