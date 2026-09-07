"""Tests for the embeddings module."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from telephone.embeddings import compute_similarity, cosine_similarity, get_embeddings


# ─── cosine_similarity Tests ────────────────────────────────


def test_cosine_similarity_identical_vectors():
    """Identical vectors have cosine similarity of 1.0."""
    a = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    """Orthogonal vectors have cosine similarity of 0.0."""
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors():
    """Opposite vectors have cosine similarity of -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert cosine_similarity(a, b) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector():
    """A zero vector returns 0.0 (no division error)."""
    a = [0.0, 0.0, 0.0]
    b = [1.0, 2.0, 3.0]
    assert cosine_similarity(a, b) == 0.0
    assert cosine_similarity(b, a) == 0.0


def test_cosine_similarity_both_zero():
    """Two zero vectors return 0.0."""
    a = [0.0, 0.0]
    b = [0.0, 0.0]
    assert cosine_similarity(a, b) == 0.0


# ─── get_embeddings Tests ───────────────────────────────────


def test_get_embeddings_calls_openai_client():
    """get_embeddings uses the OpenAI client with correct base_url and model."""
    mock_embedding_1 = MagicMock()
    mock_embedding_1.embedding = [0.1, 0.2, 0.3]
    mock_embedding_2 = MagicMock()
    mock_embedding_2.embedding = [0.4, 0.5, 0.6]

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_1, mock_embedding_2]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    env = {
        "EMBEDDING_HOST": "<LAN-HOST-3>",
        "EMBEDDING_PORT": "8002",
        "EMBEDDING_MODEL": "test-model",
    }

    with patch.dict(os.environ, env, clear=False), \
         patch("telephone.embeddings.openai.OpenAI", return_value=mock_client) as mock_openai:
        result = get_embeddings(["hello", "world"])

    mock_openai.assert_called_once_with(
        base_url="http://<LAN-HOST-3>:8002/v1",
        api_key="none",
    )
    mock_client.embeddings.create.assert_called_once_with(
        input=["hello", "world"],
        model="test-model",
    )
    assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]


def test_get_embeddings_raises_when_env_not_set():
    """get_embeddings raises ValueError when env vars are missing."""
    env = {"EMBEDDING_HOST": "", "EMBEDDING_PORT": "", "EMBEDDING_MODEL": ""}
    with patch.dict(os.environ, env, clear=False):
        with pytest.raises(ValueError, match="EMBEDDING_HOST"):
            get_embeddings(["hello"])


# ─── compute_similarity Tests ───────────────────────────────


def test_compute_similarity_returns_none_when_env_not_set():
    """compute_similarity returns None when embedding env vars are not set."""
    env = {"EMBEDDING_HOST": "", "EMBEDDING_PORT": "", "EMBEDDING_MODEL": ""}
    with patch.dict(os.environ, env, clear=False):
        result = compute_similarity("hello", "world")
    assert result is None


def test_compute_similarity_returns_none_when_service_unreachable():
    """compute_similarity returns None when the embedding service is down."""
    env = {
        "EMBEDDING_HOST": "<LAN-HOST-3>",
        "EMBEDDING_PORT": "8002",
        "EMBEDDING_MODEL": "test-model",
    }
    with patch.dict(os.environ, env, clear=False), \
         patch("telephone.embeddings.openai.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.side_effect = Exception("Connection refused")
        result = compute_similarity("hello", "world")
    assert result is None


def test_compute_similarity_returns_score():
    """compute_similarity returns a float score when embeddings succeed."""
    mock_embedding_1 = MagicMock()
    mock_embedding_1.embedding = [1.0, 0.0]
    mock_embedding_2 = MagicMock()
    mock_embedding_2.embedding = [1.0, 0.0]

    mock_response = MagicMock()
    mock_response.data = [mock_embedding_1, mock_embedding_2]

    env = {
        "EMBEDDING_HOST": "<LAN-HOST-3>",
        "EMBEDDING_PORT": "8002",
        "EMBEDDING_MODEL": "test-model",
    }
    with patch.dict(os.environ, env, clear=False), \
         patch("telephone.embeddings.openai.OpenAI") as mock_openai:
        mock_openai.return_value.embeddings.create.return_value = mock_response
        result = compute_similarity("hello", "hello")
    assert result == pytest.approx(1.0)
