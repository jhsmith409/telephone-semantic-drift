# Copyright (c) 2026 James H. Smith. MIT License.
"""Semantic similarity via OpenAI-compatible embedding endpoint."""

from __future__ import annotations

import logging
import math
import os

import openai

logger = logging.getLogger(__name__)


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Call an OpenAI-compatible /v1/embeddings endpoint.

    Reads EMBEDDING_HOST, EMBEDDING_PORT, and EMBEDDING_MODEL from env vars.
    Returns a list of embedding vectors (one per input text).
    """
    host = os.environ.get("EMBEDDING_HOST", "").strip()
    port = os.environ.get("EMBEDDING_PORT", "").strip()
    model = os.environ.get("EMBEDDING_MODEL", "").strip()

    if not host or not port or not model:
        raise ValueError("EMBEDDING_HOST, EMBEDDING_PORT, and EMBEDDING_MODEL must be set")

    client = openai.OpenAI(
        base_url=f"http://{host}:{port}/v1",
        api_key="none",
    )
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Returns 0.0 if either vector has zero magnitude.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def compute_similarity(initial: str, output: str) -> float | None:
    """Compute cosine similarity between two texts via embeddings.

    Returns None if the embedding service is unavailable or not configured,
    so batch tests still work without an embedding model.
    """
    try:
        vectors = get_embeddings([initial, output])
        return cosine_similarity(vectors[0], vectors[1])
    except Exception:
        logger.debug("Embedding service unavailable, skipping similarity", exc_info=True)
        return None
