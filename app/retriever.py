"""Vector search over the Chroma collection built by ingest.py."""
from __future__ import annotations

from typing import Protocol

from .config import settings
from .text import Chunk


class Retriever(Protocol):
    def search(self, query: str, k: int = settings.top_k) -> list[Chunk]: ...


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

    def search(self, query: str, k: int = settings.top_k) -> list[Chunk]:
        if self.collection.count() == 0:
            raise RuntimeError("Vector store is empty — run `python -m app.ingest` first.")
        vec = self.embedder.embed([query])[0]
        res = self.collection.query(query_embeddings=[vec], n_results=k)
        chunks = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            page = meta.get("page") or None
            chunks.append(Chunk(text=text, source=meta.get("source", "?"), page=page))
        return chunks
