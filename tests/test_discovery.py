"""Tests for the discovery module."""

from __future__ import annotations

import socket
from unittest.mock import MagicMock, patch

from telephone.discovery import (
    DiscoveredService,
    discover_services,
    is_inference_model,
    load_hosts,
    probe_ollama,
    probe_openai_compatible,
    scan_hosts,
    scan_port,
)


def test_load_hosts_from_env():
    with patch.dict("os.environ", {"AI_HOSTS": "<LAN-HOST-1>, <LAN-HOST-2>, myhost"}):
        hosts = load_hosts()
    assert hosts == ["<LAN-HOST-1>", "<LAN-HOST-2>", "myhost"]


def test_load_hosts_empty():
    with patch.dict("os.environ", {"AI_HOSTS": ""}):
        hosts = load_hosts()
    assert hosts == []


def test_load_hosts_missing():
    with patch.dict("os.environ", {}, clear=True):
        hosts = load_hosts()
    assert hosts == []


def test_scan_port_open():
    """Start a temporary server and verify scan_port detects it."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    try:
        assert scan_port("127.0.0.1", port, timeout=1.0) is True
    finally:
        server.close()


def test_scan_port_closed():
    assert scan_port("127.0.0.1", 1, timeout=0.1) is False


def test_scan_hosts():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", 0))
    port = server.getsockname()[1]
    server.listen(1)
    try:
        results = scan_hosts(["127.0.0.1"], [port], timeout=1.0)
        assert ("127.0.0.1", port) in results
    finally:
        server.close()


@patch("telephone.discovery.ollama.Client")
def test_probe_ollama_success(mock_client_cls):
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.model = "llama3:latest"
    mock_response = MagicMock()
    mock_response.models = [mock_model]
    mock_client.list.return_value = mock_response
    mock_client_cls.return_value = mock_client

    result = probe_ollama("<LAN-HOST-1>", 11434)
    assert len(result) == 1
    assert result[0]["name"] == "llama3:latest"


@patch("telephone.discovery.ollama.Client")
def test_probe_ollama_failure(mock_client_cls):
    mock_client_cls.return_value.list.side_effect = Exception("connection refused")
    result = probe_ollama("<LAN-HOST-1>", 11434)
    assert result == []


@patch("telephone.discovery.openai.OpenAI")
def test_probe_openai_compatible_success(mock_openai_cls):
    mock_client = MagicMock()
    mock_model = MagicMock()
    mock_model.id = "gpt-4"
    mock_client.models.list.return_value = MagicMock(data=[mock_model])
    mock_openai_cls.return_value = mock_client

    result = probe_openai_compatible("<LAN-HOST-2>", 8000)
    assert len(result) == 1
    assert result[0]["id"] == "gpt-4"


@patch("telephone.discovery.openai.OpenAI")
def test_probe_openai_compatible_failure(mock_openai_cls):
    mock_openai_cls.return_value.models.list.side_effect = Exception("timeout")
    result = probe_openai_compatible("<LAN-HOST-2>", 8000)
    assert result == []


def test_is_inference_model_true():
    assert is_inference_model("llama3:latest") is True
    assert is_inference_model("mistral-7b") is True
    assert is_inference_model("phi3:mini") is True


def test_is_inference_model_false():
    assert is_inference_model("nomic-embed-text") is False
    assert is_inference_model("bge-large-en") is False
    assert is_inference_model("text-embedding-ada-002") is False
    assert is_inference_model("mxbai-embed-large") is False
    assert is_inference_model("rerank-english-v3.0") is False


@patch("telephone.discovery.probe_openai_compatible")
@patch("telephone.discovery.probe_ollama")
@patch("telephone.discovery.scan_hosts")
def test_discover_services(mock_scan, mock_probe_ollama, mock_probe_openai):
    mock_scan.side_effect = [
        [("<LAN-HOST-1>", 11434)],  # Ollama scan
        [("<LAN-HOST-2>", 8000)],   # OpenAI scan
    ]
    mock_probe_ollama.return_value = [
        {"name": "llama3:latest"},
        {"name": "nomic-embed-text"},  # should be filtered
    ]
    mock_probe_openai.return_value = [{"id": "mistral-7b"}]

    services = discover_services(["<LAN-HOST-1>", "<LAN-HOST-2>"])
    assert len(services) == 2
    assert services[0].service_type == "ollama"
    assert services[0].models == ["llama3:latest"]  # embed filtered out
    assert services[1].service_type == "openai"
    assert services[1].models == ["mistral-7b"]


def test_discover_services_no_hosts():
    services = discover_services([])
    assert services == []


def test_discovered_service_to_dict():
    svc = DiscoveredService(
        host="<LAN-GATEWAY>",
        port=11434,
        service_type="ollama",
        models=["phi3", "llama3"],
    )
    d = svc.to_dict()
    assert d["host"] == "<LAN-GATEWAY>"
    assert d["port"] == 11434
    assert d["service_type"] == "ollama"
    assert len(d["models"]) == 2
