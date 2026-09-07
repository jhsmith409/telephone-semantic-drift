# Copyright (c) 2026 James H. Smith. MIT License.
"""AI service discovery via .env-configured host list."""

from __future__ import annotations

import os
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import ollama
import openai


def load_hosts() -> list[str]:
    """Load AI server IPs/hostnames from the AI_HOSTS env var.

    Expected format: comma-separated list, e.g. "<LAN-HOST-1>,<LAN-HOST-2>,myhost"
    Returns empty list if not set.
    """
    raw = os.environ.get("AI_HOSTS", "").strip()
    if not raw:
        return []
    return [h.strip() for h in raw.split(",") if h.strip()]


def scan_port(ip: str, port: int, timeout: float = 0.5) -> bool:
    """TCP connect check — returns True if port is open."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return True
    except (OSError, TimeoutError):
        return False


def scan_hosts(
    hosts: list[str], ports: list[int], timeout: float = 0.5
) -> list[tuple[str, int]]:
    """Scan specific hosts for given ports concurrently.

    Returns list of (host, port) pairs that are open.
    """
    results: list[tuple[str, int]] = []

    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = {}
        for host in hosts:
            for port in ports:
                fut = executor.submit(scan_port, host, port, timeout)
                futures[fut] = (host, port)

        for future in as_completed(futures):
            if future.result():
                results.append(futures[future])

    return results


def probe_ollama(host: str, port: int = 11434) -> list[dict]:
    """Query an Ollama instance for available models.

    Returns list of model dicts with 'name' key, or [] on failure.
    """
    try:
        client = ollama.Client(host=f"http://{host}:{port}")
        response = client.list()
        return [{"name": m.model} for m in response.models]
    except Exception:
        return []


def probe_openai_compatible(
    host: str, port: int, api_key: str = ""
) -> list[dict]:
    """Query an OpenAI-compatible endpoint for available models.

    Returns list of model dicts with 'id' key, or [] on failure.
    """
    try:
        client = openai.OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key or "none",
        )
        response = client.models.list()
        return [{"id": m.id} for m in response.data]
    except Exception:
        return []


# Model name patterns that indicate non-inference models (embedding, reranking, etc.)
_NON_INFERENCE_PATTERNS = (
    "embed", "embedding", "bge", "e5-", "rerank", "reranking",
    "text-embedding", "mxbai-embed",
)


def is_inference_model(name: str) -> bool:
    """Return True if the model name looks like a full inference model."""
    lower = name.lower()
    return not any(pat in lower for pat in _NON_INFERENCE_PATTERNS)


@dataclass
class DiscoveredService:
    host: str
    port: int
    service_type: str  # "ollama" or "openai"
    models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "service_type": self.service_type,
            "models": self.models,
        }


def discover_services(hosts: list[str] | None = None) -> list[DiscoveredService]:
    """Discover AI services on the given hosts.

    1. Scan port 11434 on each host -> probe as Ollama
    2. Scan ports 8000-8010 on each host -> probe as OpenAI-compatible
    3. Filter out non-inference models (embedding, reranking)
    4. Return consolidated list of discovered services with their models.

    If hosts is None, loads from AI_HOSTS env var.
    """
    if hosts is None:
        hosts = load_hosts()
    if not hosts:
        return []

    services: list[DiscoveredService] = []

    # Scan for Ollama instances
    ollama_hits = scan_hosts(hosts, [11434])
    for ip, port in ollama_hits:
        models = probe_ollama(ip, port)
        if models:
            inference_models = [m["name"] for m in models if is_inference_model(m["name"])]
            if inference_models:
                services.append(
                    DiscoveredService(
                        host=ip, port=port, service_type="ollama",
                        models=inference_models,
                    )
                )

    # Scan for OpenAI-compatible instances (ports 8000-8010)
    openai_ports = list(range(8000, 8011))
    openai_hits = scan_hosts(hosts, openai_ports)
    for ip, port in openai_hits:
        models = probe_openai_compatible(ip, port)
        if models:
            inference_models = [m["id"] for m in models if is_inference_model(m["id"])]
            if inference_models:
                services.append(
                    DiscoveredService(
                        host=ip, port=port, service_type="openai",
                        models=inference_models,
                    )
                )

    return services
