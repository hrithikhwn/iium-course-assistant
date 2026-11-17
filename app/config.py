"""Settings loaded from environment variables / .env."""
from __future__ import annotations

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


@dataclass(frozen=True)
class Settings:
    # LLM — any OpenAI-compatible endpoint (Ollama exposes one at /v1)
    llm_base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "qwen2.5:7b"))
    llm_api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY", "ollama"))
    llm_timeout: int = field(default_factory=lambda: _int("LLM_TIMEOUT", 120))

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
    max_retries: int = field(default_factory=lambda: _int("MAX_RETRIES", 2))


settings = Settings()
