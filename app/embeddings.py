"""Embeddings via Ollama's native /api/embed endpoint."""
from __future__ import annotations

import httpx

from .config import settings


class OllamaEmbedder:
    def __init__(self, url: str = settings.ollama_url, model: str = settings.embed_model,
                 doc_prefix: str = settings.embed_doc_prefix, query_prefix: str = settings.embed_query_prefix):
        self.model = model
        # nomic-embed-text is trained with task prefixes and retrieves much
        # worse without them; set both to "" in .env for models that don't use them
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self._client = httpx.Client(base_url=url.rstrip("/"), timeout=120)

    def embed(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        vectors: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self._client.post("/api/embed", json={"model": self.model, "input": batch})
            resp.raise_for_status()
            vectors.extend(resp.json()["embeddings"])
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed([self.doc_prefix + t for t in texts])

    def embed_query(self, text: str) -> list[float]:
        return self.embed([self.query_prefix + text])[0]
