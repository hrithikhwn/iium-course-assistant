"""Swappable LLM client.

Talks to any OpenAI-compatible /chat/completions endpoint: Ollama, LM Studio,
llama.cpp server, vLLM, or a hosted API. Change LLM_BASE_URL / LLM_MODEL in
.env to switch models — nothing else in the code needs to change.
"""
from __future__ import annotations

from typing import Protocol

import httpx

from .config import settings


class LLM(Protocol):
    """Anything with this method can drive the agent (handy for tests)."""

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str: ...


class OpenAICompatibleLLM:
    def __init__(
        self,
        base_url: str = settings.llm_base_url,
        model: str = settings.llm_model,
        api_key: str = settings.llm_api_key,
        timeout: int = settings.llm_timeout,
    ):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._client.post(
            "/chat/completions",
            json={
                "model": self.model,
                "temperature": temperature,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
