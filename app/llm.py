"""Swappable LLM client.

Talks to any OpenAI-compatible /chat/completions endpoint: Ollama, LM Studio,
llama.cpp server, vLLM, or a hosted API. Change LLM_BASE_URL / LLM_MODEL in
.env to switch models (and LLM_API_KEY, plus LLM_EXTRA_BODY if the provider
needs options, for a hosted API) — nothing else in the code needs to change.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Protocol

import httpx

from .config import settings

log = logging.getLogger(__name__)

# Worth asking again: a request timeout (408), too many requests (429), or a
# failed or overloaded server (5xx). Any other error, like a wrong key, an
# empty balance or an invalid request, would only fail the same way again.
RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
# 6 tries: waits of 2, 4, 8, 16 and 32 s, over a minute in all, outlast a
# per-minute rate limit
MAX_ATTEMPTS = 6
MAX_WAIT = 60.0   # seconds between two tries


def backoff(attempt: int) -> float:
    """Seconds to wait after failed try n: 2, 4, 8, ..., at most MAX_WAIT."""
    return min(2.0 ** attempt, MAX_WAIT)


def retry_after(resp: httpx.Response) -> float | None:
    """The wait the server asked for in a Retry-After header, at most MAX_WAIT;
    None without one, or when it gives a date rather than seconds."""
    try:
        return min(max(float(resp.headers["retry-after"]), 0.0), MAX_WAIT)
    except (KeyError, ValueError):
        return None


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
        extra_body: dict | None = None,
        max_attempts: int = MAX_ATTEMPTS,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.model = model
        # provider options sent with every request (LLM_EXTRA_BODY), e.g.
        # {"thinking": {"type": "disabled"}} for DeepSeek
        self.extra_body = settings.llm_extra_body if extra_body is None else extra_body
        self.max_attempts = max_attempts
        self._sleep = sleep            # the tests pass a fake, so they don't wait
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
            transport=transport,       # the tests pass a fake server
        )

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        resp = self._post({
            "model": self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            **self.extra_body,   # added last, so it can also override the fields above
        })
        return resp.json()["choices"][0]["message"]["content"].strip()

    def _post(self, body: dict) -> httpx.Response:
        """POST the request, and ask again after a rate limit, a server error or
        a dropped connection: up to max_attempts tries, waiting 2 s and then
        twice as long each time, or as long as the server's Retry-After says."""
        for attempt in range(1, self.max_attempts + 1):
            last = attempt == self.max_attempts
            try:
                resp = self._client.post("/chat/completions", json=body)
            except httpx.TransportError as e:   # timed out, refused or dropped connection
                if last:
                    raise
                wait, why = backoff(attempt), type(e).__name__
            else:
                if resp.status_code not in RETRY_STATUSES or last:
                    resp.raise_for_status()
                    return resp
                wait, why = retry_after(resp) or backoff(attempt), f"HTTP {resp.status_code}"
            log.warning("LLM request failed (%s); trying again in %g s (try %d of %d)",
                        why, wait, attempt + 1, self.max_attempts)
            self._sleep(wait)
        raise ValueError(f"max_attempts must be at least 1, not {self.max_attempts}")
