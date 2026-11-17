"""Embeddings via Ollama's native /api/embed endpoint."""
from __future__ import annotations

import httpx

from .config import settings


class OllamaEmbedder:
    def __init__(self, url: str = settings.ollama_url, model: str = settings.embed_model):
        self.model = model
        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=120)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return vectors
