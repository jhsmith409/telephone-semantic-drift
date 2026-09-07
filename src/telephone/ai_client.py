# Copyright (c) 2026 James H. Smith. MIT License.
"""Unified AI client for Ollama and OpenAI-compatible services."""

from __future__ import annotations

import ollama
import openai


_VALID_SERVICE_TYPES = ("ollama", "openai")

_REQUEST_TIMEOUT = 600  # seconds per call


class AIClient:
    """Sends chat completion requests to Ollama or OpenAI-compatible endpoints."""

    def __init__(
        self,
        service_type: str,
        host: str,
        port: int,
        api_key: str = "",
    ) -> None:
        if service_type not in _VALID_SERVICE_TYPES:
            raise ValueError(
                f"service_type must be one of {_VALID_SERVICE_TYPES}, got '{service_type}'"
            )
        self.service_type = service_type
        self.host = host
        self.port = port
        self.api_key = api_key

    def generate_response(
        self,
        model: str,
        system_prompt: str,
        user_message: str,
        temperature: float | None = None,
        max_tokens: int = 16384,
        seed: int | None = None,
    ) -> str:
        """Send a chat request and return the assistant's text response.

        If temperature is None, the model's default is used.
        max_tokens caps the response length.
        Retries once on failure before raising.
        """
        last_error: Exception | None = None
        for _ in range(2):
            try:
                if self.service_type == "ollama":
                    return self._call_ollama(model, system_prompt, user_message, temperature, max_tokens, seed)
                else:
                    return self._call_openai(model, system_prompt, user_message, temperature, max_tokens, seed)
            except Exception as exc:
                last_error = exc
        raise RuntimeError(
            f"Failed after retry ({self.service_type} @ {self.host}:{self.port}): "
            f"{last_error}"
        )

    def _call_ollama(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float | None, max_tokens: int, seed: int | None = None,
    ) -> str:
        client = ollama.Client(host=f"http://{self.host}:{self.port}", timeout=_REQUEST_TIMEOUT)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        options: dict = {"num_predict": max_tokens}
        if temperature is not None:
            options["temperature"] = temperature
        if seed is not None:
            options["seed"] = seed
        response = client.chat(
            model=model,
            messages=messages,
            options=options,
        )
        return response.message.content or ""

    def _call_openai(
        self, model: str, system_prompt: str, user_message: str,
        temperature: float | None, max_tokens: int, seed: int | None = None,
    ) -> str:
        client = openai.OpenAI(
            base_url=f"http://{self.host}:{self.port}/v1",
            api_key=self.api_key or "none",
            timeout=_REQUEST_TIMEOUT,
        )
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_message})
        kwargs: dict = {"model": model, "messages": messages, "max_tokens": max_tokens}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if seed is not None:
            kwargs["seed"] = seed
        response = client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""
