"""Search over the Chroma collection built by ingest.py.

`search` is hybrid: embeddings (meaning) and BM25 keyword search (exact
names and course codes) are run separately and merged with reciprocal rank
fusion. Course-table chunks are lists of many similar courses, so on
embeddings alone the one you asked about gets outranked by general text
about the same topic.
"""
from __future__ import annotations

from typing import Protocol

from .config import settings
from .text import BM25, Chunk, matches_filter, plan_filter, rrf


class Retriever(Protocol):
    def search(self, query: str, k: int = settings.top_k,
               plan: str | None = None, programme: str | None = None) -> list[Chunk]: ...


def _chunk(text: str, meta: dict) -> Chunk:
    return Chunk(
        text=text,
        source=meta.get("source", "?"),
        page=meta.get("page") or None,
        plan=meta.get("plan", "all"),
        programme=meta.get("programme", "all"),
    )


class ChromaRetriever:
    def __init__(self, collection=None, embedder=None, candidates: int = 20):
        if collection is None:
            import chromadb

            client = chromadb.PersistentClient(path=str(settings.chroma_dir))
            collection = client.get_or_create_collection(settings.collection)
        if embedder is None:
            from .embeddings import OllamaEmbedder

            embedder = OllamaEmbedder()
        self.collection = collection
        self.embedder = embedder
        self.candidates = candidates      # how deep each method looks before merging
        self._bm25 = None                 # built on first keyword search

    def _check(self, plan, programme) -> str | None:
        if self.collection.count() == 0:
            raise RuntimeError("Vector store is empty — run `python -m app.ingest` first.")
        if (plan is None) != (programme is None):
            raise ValueError("pass both plan and programme, or neither")
        return programme.upper() if programme else None

    def _vector(self, query: str, n: int, plan, programme) -> list[tuple[str, str, dict]]:
        vec = self.embedder.embed_query(query)
        where = plan_filter(plan, programme) if plan else None
        res = self.collection.query(query_embeddings=[vec], n_results=n, where=where)
        return list(zip(res["ids"][0], res["documents"][0], res["metadatas"][0]))

    def _keyword(self, query: str, n: int, plan, programme) -> list[tuple[str, str, dict]]:
        if self._bm25 is None:
            data = self.collection.get(include=["documents", "metadatas"])
            self._docs = list(zip(data["ids"], data["documents"], data["metadatas"]))
            self._bm25 = BM25([d for _, d, _ in self._docs])
        hits = [self._docs[i] for i in self._bm25.top(query, len(self._docs))]
        return [h for h in hits if matches_filter(h[2], plan, programme)][:n]

    def vector_search(self, query: str, k: int = settings.top_k,
                      plan: str | None = None, programme: str | None = None) -> list[Chunk]:
        """Embeddings only (the old behaviour), kept for comparison."""
        programme = self._check(plan, programme)
        return [_chunk(d, m) for _, d, m in self._vector(query, k, plan, programme)]

    def search(self, query: str, k: int = settings.top_k,
               plan: str | None = None, programme: str | None = None) -> list[Chunk]:
        """Top-k chunks for `query`. With a plan and programme (see
        text.plan_for), only that plan's documents and the 'all' documents
        are searched; without them, everything is."""
        programme = self._check(plan, programme)
        vector = self._vector(query, self.candidates, plan, programme)
        keyword = self._keyword(query, self.candidates, plan, programme)
        by_id = {i: (d, m) for i, d, m in vector + keyword}
        merged = rrf([[i for i, _, _ in vector], [i for i, _, _ in keyword]])
        return [_chunk(*by_id[i]) for i in merged[:k]]
