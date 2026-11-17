"""Vector search over the Chroma collection built by ingest.py."""
from __future__ import annotations

from typing import Protocol

from .config import settings
from .text import Chunk, plan_filter


class Retriever(Protocol):
    def search(self, query: str, k: int = settings.top_k,
               plan: str | None = None, programme: str | None = None) -> list[Chunk]: ...


class ChromaRetriever:
    def __init__(self, collection=None, embedder=None):
        if collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            collection = client.get_or_create_collection(settings.collection)
        if embedder is None:
            from .embeddings import OllamaEmbedder

            embedder = OllamaEmbedder()
        self.collection = collection
        self.embedder = embedder

    def search(self, query: str, k: int = settings.top_k,
               plan: str | None = None, programme: str | None = None) -> list[Chunk]:
        """Top-k chunks for `query`. With a plan and programme (see
        text.plan_for), only that plan's documents and the 'all' documents
        are searched; without them, everything is."""
        if self.collection.count() == 0:
            raise RuntimeError("Vector store is empty — run `python -m app.ingest` first.")
        if (plan is None) != (programme is None):
            raise ValueError("pass both plan and programme, or neither")
        vec = self.embedder.embed_query(query)
        where = plan_filter(plan, programme.upper()) if plan else None
        res = self.collection.query(query_embeddings=[vec], n_results=k, where=where)
        chunks = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            chunks.append(Chunk(
                text=text,
                source=meta.get("source", "?"),
                page=meta.get("page") or None,
                plan=meta.get("plan", "all"),
                programme=meta.get("programme", "all"),
            ))
        return chunks
