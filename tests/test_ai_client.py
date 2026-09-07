"""Tests for the AI client module."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from telephone.ai_client import AIClient


@patch("telephone.ai_client.ollama.Client")
def test_ollama_generate_response(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Hello back!"
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    client = AIClient("ollama", "localhost", 11434)
    result = client.generate_response("llama3", "Be helpful", "Hello")

    assert result == "Hello back!"
    mock_client.chat.assert_called_once()
    call_kwargs = mock_client.chat.call_args
    assert call_kwargs.kwargs["model"] == "llama3"


@patch("telephone.ai_client.openai.OpenAI")
def test_openai_generate_response(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Greetings!"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = AIClient("openai", "localhost", 8000, api_key="test-key")
    result = client.generate_response("gpt-4", "", "Hi")

    assert result == "Greetings!"


@patch("telephone.ai_client.ollama.Client")
def test_retry_on_failure(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "Success on retry"
    mock_client.chat.side_effect = [Exception("first fail"), mock_response]
    mock_client_cls.return_value = mock_client

    client = AIClient("ollama", "localhost", 11434)
    result = client.generate_response("llama3", "", "test")
    assert result == "Success on retry"
    assert mock_client.chat.call_count == 2


@patch("telephone.ai_client.ollama.Client")
def test_raises_after_retry_exhausted(mock_client_cls):
    mock_client = MagicMock()
    mock_client.chat.side_effect = Exception("persistent failure")
    mock_client_cls.return_value = mock_client

    client = AIClient("ollama", "localhost", 11434)
    with pytest.raises(RuntimeError, match="Failed after retry"):
        client.generate_response("llama3", "", "test")
    assert mock_client.chat.call_count == 2


@patch("telephone.ai_client.ollama.Client")
def test_ollama_with_temperature(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "warm response"
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    client = AIClient("ollama", "localhost", 11434)
    result = client.generate_response("llama3", "", "test", temperature=0.5)
    assert result == "warm response"
    call_kwargs = mock_client.chat.call_args
    assert call_kwargs.kwargs["options"]["temperature"] == 0.5


@patch("telephone.ai_client.ollama.Client")
def test_ollama_without_temperature(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "default response"
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    client = AIClient("ollama", "localhost", 11434)
    result = client.generate_response("llama3", "", "test", temperature=None)
    assert result == "default response"
    call_kwargs = mock_client.chat.call_args
    assert "temperature" not in call_kwargs.kwargs["options"]


@patch("telephone.ai_client.openai.OpenAI")
def test_openai_with_temperature(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "warm response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    client = AIClient("openai", "localhost", 8000)
    result = client.generate_response("model", "", "test", temperature=0.3)
    assert result == "warm response"
    call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs.kwargs["temperature"] == 0.3


# ─── Service Type Validation Tests ───────────────────────────


def test_invalid_service_type_raises():
    with pytest.raises(ValueError, match="service_type must be one of"):
        AIClient("invalid", "localhost", 8000)


def test_valid_service_types_accepted():
    """Both 'ollama' and 'openai' are accepted without error."""
    # These should not raise
    AIClient("ollama", "localhost", 11434)
    AIClient("openai", "localhost", 8000)


# ─── Timeout Tests ───────────────────────────────────────────


@patch("telephone.ai_client.ollama.Client")
def test_ollama_client_uses_timeout(mock_client_cls):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.message.content = "ok"
    mock_client.chat.return_value = mock_response
    mock_client_cls.return_value = mock_client

    from telephone.ai_client import _REQUEST_TIMEOUT

    client = AIClient("ollama", "localhost", 11434)
    client.generate_response("llama3", "", "test")

    mock_client_cls.assert_called_with(
        host="http://localhost:11434", timeout=_REQUEST_TIMEOUT
    )


@patch("telephone.ai_client.openai.OpenAI")
def test_openai_client_uses_timeout(mock_openai_cls):
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "ok"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai_cls.return_value = mock_client

    from telephone.ai_client import _REQUEST_TIMEOUT

    client = AIClient("openai", "localhost", 8000)
    client.generate_response("model", "", "test")

    mock_openai_cls.assert_called_with(
        base_url="http://localhost:8000/v1",
        api_key="none",
        timeout=_REQUEST_TIMEOUT,
    )
