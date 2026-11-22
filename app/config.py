"""Settings loaded from environment variables / .env."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # dotenv is optional (e.g. in CI unit tests)
    pass


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name, "").strip().lower()
    return raw in {"1", "true", "yes", "on"} if raw else default


def _json_object(name: str) -> dict:
    """A JSON object from the environment; {} when unset or empty."""
    raw = os.getenv(name, "").strip()
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"{name} is not valid JSON ({e}): {raw}") from None
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object, like {{\"key\": \"value\"}}, not: {raw}")
    return value


@dataclass(frozen=True)
class Settings:
    # LLM — any OpenAI-compatible endpoint (Ollama exposes one at /v1)
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 120))
    # Provider options added to every LLM request, as a JSON object. DeepSeek,
    # for one, thinks before it answers unless sent {"thinking": {"type": "disabled"}}
    llm_extra_body: dict = field(default_factory=lambda: _json_object("LLM_EXTRA_BODY"))

    # Embeddings — Ollama native API
    ollama_url: str = field(default_factory=lambda: os.getenv("OLLAMA_URL", "http://localhost:11434"))
    embed_model: str = field(default_factory=lambda: os.getenv("EMBED_MODEL", "nomic-embed-text"))
    # task prefixes nomic-embed-text expects; "" for embedding models without them
    embed_doc_prefix: str = field(default_factory=lambda: os.getenv("EMBED_DOC_PREFIX", "search_document: "))
    embed_query_prefix: str = field(default_factory=lambda: os.getenv("EMBED_QUERY_PREFIX", "search_query: "))

    # Storage
    raw_dir: Path = field(default_factory=lambda: Path(os.getenv("RAW_DIR", "data/raw")))
    chroma_dir: Path = field(default_factory=lambda: Path(os.getenv("CHROMA_DIR", "data/chroma")))
    courses_dir: Path = field(default_factory=lambda: Path(os.getenv("COURSES_DIR", "data/courses")))
    collection: str = field(default_factory=lambda: os.getenv("COLLECTION", "iium_docs"))

    # Retrieval / agent
    chunk_size: int = field(default_factory=lambda: _int("CHUNK_SIZE", 900))
    chunk_overlap: int = field(default_factory=lambda: _int("CHUNK_OVERLAP", 150))
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    # hybrid search: this many of the best embedding hits and the best keyword
    # hits always make the top k (0 = plain rank fusion)
    search_guaranteed: int = field(default_factory=lambda: _int("SEARCH_GUARANTEED", 2))
    # The next three switches are what the Phase 4 baseline eval chose
    # (eval/results/summary.csv: 70 answers to the answerable questions per run).
    #
    # plan step: the LLM rewrites the question into a search query, searched along
    # with the question. Off by default: with qwen2.5:7b it got no more answers
    # right (51 of 70 either way) and cost an LLM call (3.9 per question, not 3.0).
    query_rewrite: bool = field(default_factory=lambda: _bool("QUERY_REWRITE", False))
    # grade step: the LLM checks whether the excerpts answer the question, and the
    # agent searches again if not. Off = plain RAG: one search, then the answer,
    # which got fewer right with every model (qwen2.5:7b 50 vs 51, llama3.1:8b 47
    # vs 55, qwen2.5:3b 44 vs 51).
    grading: bool = field(default_factory=lambda: _bool("GRADING", True))
    # strict: "not found" when the last grade still fails. Off: the grader turns
    # down excerpts that do answer the question, and strict qwen2.5:7b said "not
    # found" in 28 of its 70 answers and got 34 right, not 51.
    refuse_on_failed_grade: bool = field(default_factory=lambda: _bool("REFUSE_ON_FAILED_GRADE", False))
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 2))
    # Most excerpt text sent to the LLM in one prompt. Ollama's default window is
    # 4096 tokens (~3.5 characters each) and longer prompts are cut silently,
    # losing the instructions. Raise it only together with OLLAMA_CONTEXT_LENGTH.
    max_context_chars: int = field(default_factory=lambda: _int("MAX_CONTEXT_CHARS", 11000))


settings = Settings()
