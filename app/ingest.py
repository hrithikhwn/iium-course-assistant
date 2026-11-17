"""Ingest PDFs / text files from data/raw into a Chroma vector store.

Usage:
    python -m app.ingest            # add/refresh everything in data/raw
    python -m app.ingest --reset    # wipe the collection first
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from .config import settings
from .text import Chunk, chunk_text

SUPPORTED = {".pdf", ".txt", ".md"}


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return (text, page_number) pairs. Text files have no page numbers."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i) for i, page in enumerate(reader.pages, 1)]
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def build_chunks(raw_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        for text, page in load_file(path):
            for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(text=piece, source=path.name, page=page))
        print(f"  read {path.name}")
    return chunks


def chunk_id(c: Chunk) -> str:
    return hashlib.sha1(f"{c.source}|{c.page}|{c.text}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the collection before ingesting")
    args = parser.parse_args()

    import chromadb

    from .embeddings import OllamaEmbedder

    raw_dir = settings.raw_dir
    if not raw_dir.exists():
        raise SystemExit(f"{raw_dir} does not exist — create it and add your documents.")

    print(f"Reading documents from {raw_dir}/ ...")
    chunks = build_chunks(raw_dir)
    if not chunks:
        raise SystemExit(f"No .pdf/.txt/.md files with text found in {raw_dir}/")

    client = chromadb.PersistentClient(path=str(settings.chroma_dir))
    if args.reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(settings.collection)

    print(f"Embedding {len(chunks)} chunks with {settings.embed_model} ...")
    embedder = OllamaEmbedder()
    vectors = embedder.embed([c.text for c in chunks])

    collection.upsert(
        ids=[chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c.source, "page": c.page or 0} for c in chunks],
    )
    print(f"Done. Collection '{settings.collection}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
