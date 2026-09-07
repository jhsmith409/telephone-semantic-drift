"""Tests for the batch API server."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from telephone.batch_api import create_batch_app
from telephone.discovery import DiscoveredService
from telephone.game_logic import RelayStep


@pytest.fixture()
def app():
    app = create_batch_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture()
def client(app):
    return app.test_client()


FAKE_SERVICES = [
    DiscoveredService(
        host="<GPU-HOST-A>", port=11434,
        service_type="ollama", models=["llama3:latest"],
    ),
    DiscoveredService(
        host="<GPU-HOST-A>", port=8000,
        service_type="openai", models=["gpt-4"],
    ),
]


# ─── GET /api/services Tests ────────────────────────────────


def test_services_returns_discovered(client):
    """GET /api/services returns discovered services."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.discover_services", return_value=FAKE_SERVICES), \
         patch("telephone.batch_api.load_hosts", return_value=["<GPU-HOST-A>"]):
        resp = client.get("/api/services")
    assert resp.status_code == 200
    data = resp.get_json()
    assert len(data["services"]) == 2
    assert data["services"][0]["host"] == "<GPU-HOST-A>"
    assert data["services"][0]["service_type"] == "ollama"
    assert "llama3:latest" in data["services"][0]["models"]


def test_services_requires_auth_when_key_set(client):
    """GET /api/services requires auth when APP_API_KEY is set."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False):
        resp = client.get("/api/services")
    assert resp.status_code == 401


def test_services_auth_succeeds(client):
    """GET /api/services succeeds with correct Bearer token."""
    with patch.dict(os.environ, {"APP_API_KEY": "secret123"}, clear=False), \
         patch("telephone.batch_api.discover_services", return_value=FAKE_SERVICES), \
         patch("telephone.batch_api.load_hosts", return_value=["<GPU-HOST-A>"]):
        resp = client.get(
            "/api/services",
            headers={"Authorization": "Bearer secret123"},
        )
    assert resp.status_code == 200


# ─── POST /api/run Tests ────────────────────────────────────


VALID_RUN_REQUEST = {
    "message": "The quick brown fox jumps over the lazy dog",
    "service_type": "ollama",
    "host": "<GPU-HOST-A>",
    "port": 11434,
    "model": "llama3:latest",
    "prompt_style": "paraphrase",
    "temperature": 0.7,
}


def _mock_run_chain(initial_message, agents):
    """Fake run_chain that yields a single step."""
    yield RelayStep(
        agent_index=0,
        model=agents[0].model,
        prompt_style=agents[0].prompt_style,
        input_message=initial_message,
        output_message="A swift auburn fox leaps over the idle hound",
    )


def test_run_success(client):
    """POST /api/run returns success with expected fields."""
    with patch.dict(os.environ, {"APP_API_KEY": "", "EMBEDDING_MODEL": "test-embed"}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_mock_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.872):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "success"
    assert data["input_message"] == VALID_RUN_REQUEST["message"]
    assert data["output_message"] == "A swift auburn fox leaps over the idle hound"
    assert data["model"] == "llama3:latest"
    assert data["prompt_style"] == "paraphrase"
    assert data["temperature"] == 0.7
    assert data["service_type"] == "ollama"
    assert isinstance(data["diff"], list)
    assert isinstance(data["elapsed_seconds"], float)


def test_run_includes_cosine_similarity(client):
    """POST /api/run response includes cosine_similarity field."""
    with patch.dict(os.environ, {"APP_API_KEY": "", "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B"}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_mock_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.872):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    data = resp.get_json()
    assert data["cosine_similarity"] == 0.872
    assert data["embedding_model"] == "Qwen/Qwen3-Embedding-0.6B"


def test_run_cosine_similarity_null_when_unavailable(client):
    """cosine_similarity is null when embedding service is unavailable."""
    with patch.dict(os.environ, {"APP_API_KEY": "", "EMBEDDING_MODEL": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_mock_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=None):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    data = resp.get_json()
    assert data["cosine_similarity"] is None
    assert data["embedding_model"] is None


def test_run_missing_message(client):
    """POST /api/run with missing message returns 400."""
    req = {**VALID_RUN_REQUEST}
    del req["message"]
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/run", json=req)
    assert resp.status_code == 400
    assert "message" in resp.get_json()["error"]


def test_run_invalid_prompt_style(client):
    """POST /api/run with invalid prompt_style returns 400."""
    req = {**VALID_RUN_REQUEST, "prompt_style": "nonexistent"}
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/run", json=req)
    assert resp.status_code == 400
    assert "prompt_style" in resp.get_json()["error"]


def test_run_invalid_service_type(client):
    """POST /api/run with invalid service_type returns 400."""
    req = {**VALID_RUN_REQUEST, "service_type": "azure"}
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/run", json=req)
    assert resp.status_code == 400
    assert "service_type" in resp.get_json()["error"]


def test_run_wrong_content_type(client):
    """POST /api/run with wrong Content-Type returns 415."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post(
            "/api/run",
            data="message=hello",
            content_type="application/x-www-form-urlencoded",
        )
    assert resp.status_code == 415
    assert "Content-Type" in resp.get_json()["error"]


def test_run_ai_error_returns_500(client):
    """POST /api/run returns 500 with error details when AI fails."""
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=Exception("Model unavailable")):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    assert resp.status_code == 500
    data = resp.get_json()
    assert data["status"] == "error"
    assert "Model unavailable" in data["error"]
    assert data["model"] == "llama3:latest"


# ─── system_prompt and seed Tests ──────────────────────────


def test_run_passes_system_prompt(client):
    """POST /api/run passes system_prompt to AgentConfig."""
    req = {**VALID_RUN_REQUEST, "system_prompt": "You are a helpful assistant."}
    captured = {}

    def _capture_run_chain(initial_message, agents):
        captured["system_prompt"] = agents[0].system_prompt
        yield from _mock_run_chain(initial_message, agents)

    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_capture_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.9):
        resp = client.post("/api/run", json=req)

    assert resp.status_code == 200
    assert captured["system_prompt"] == "You are a helpful assistant."


def test_run_system_prompt_defaults_empty(client):
    """POST /api/run defaults system_prompt to empty string when not provided."""
    captured = {}

    def _capture_run_chain(initial_message, agents):
        captured["system_prompt"] = agents[0].system_prompt
        yield from _mock_run_chain(initial_message, agents)

    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_capture_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.9):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    assert resp.status_code == 200
    assert captured["system_prompt"] == ""


def test_run_passes_seed(client):
    """POST /api/run passes seed to AgentConfig."""
    req = {**VALID_RUN_REQUEST, "seed": 42}
    captured = {}

    def _capture_run_chain(initial_message, agents):
        captured["seed"] = agents[0].seed
        yield from _mock_run_chain(initial_message, agents)

    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_capture_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.9):
        resp = client.post("/api/run", json=req)

    assert resp.status_code == 200
    assert captured["seed"] == 42


def test_run_seed_defaults_none(client):
    """POST /api/run defaults seed to None when not provided."""
    captured = {}

    def _capture_run_chain(initial_message, agents):
        captured["seed"] = agents[0].seed
        yield from _mock_run_chain(initial_message, agents)

    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False), \
         patch("telephone.batch_api.run_chain", side_effect=_capture_run_chain), \
         patch("telephone.batch_api.compute_similarity", return_value=0.9):
        resp = client.post("/api/run", json=VALID_RUN_REQUEST)

    assert resp.status_code == 200
    assert captured["seed"] is None


def test_run_invalid_seed(client):
    """POST /api/run with non-integer seed returns 400."""
    req = {**VALID_RUN_REQUEST, "seed": "not_a_number"}
    with patch.dict(os.environ, {"APP_API_KEY": ""}, clear=False):
        resp = client.post("/api/run", json=req)
    assert resp.status_code == 400
    assert "seed" in resp.get_json()["error"]
