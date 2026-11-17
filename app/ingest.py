"""Ingest PDFs / text files from data/raw into a Chroma vector store.

Usage:
    python -m app.ingest            # add/refresh everything in data/raw
    python -m app.ingest --reset    # wipe the collection first
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .config import settings
from .text import Chunk, chunk_text, course_chunks, folder_tags

SUPPORTED = {".pdf", ".txt", ".md"}

# Study plans whose course tables were extracted to data/courses/<folder>.csv
# (scripts/extract_tables.py). pypdf scrambles those tables, so their pages
# are indexed from the hand-checked CSV; the plan's other pages (the
# semester-by-semester plan) are still indexed from the PDF text.
STUDY_PLANS = {
    "bcs-201": "STUDY PLAN FOR BCS BATCH 201.pdf",
    "bit-201": "STUDY PLAN FOR BIT BATCH 201.pdf",
    "bcs-241": "BCS Study Plan (Batch 241) Rev_3.pdf",
    "bit-241": "BIT-Study-Plan-Batch-241-Rev-3_2.pdf",
    "bit-261": "BIT-PROGRAMME-STRUCTURE-AND-STUDY-PLAN-261.pdf",
}


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return (text, page_number) pairs. Text files have no page numbers."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i) for i, page in enumerate(reader.pages, 1)]
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def load_course_csv(folder: str, courses_dir: Path) -> list[dict] | None:
    path = courses_dir / f"{folder}.csv"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_chunks(raw_dir: Path, courses_dir: Path | None = None) -> list[Chunk]:
    courses_dir = courses_dir or settings.courses_dir
    chunks: list[Chunk] = []
    for path in sorted(raw_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED or not path.is_file():
            continue
        rel = path.relative_to(raw_dir)
        if len(rel.parts) < 2:
            raise SystemExit(f"{rel} is not in a folder: put it in data/raw/all/ or data/raw/<programme>-<plan>/")
        folder = rel.parts[0]
        plan, programme = folder_tags(folder)

        covered: set[int] = set()
        note = ""
        if STUDY_PLANS.get(folder) == path.name:
            rows = load_course_csv(folder, courses_dir)
            if rows is None:
                note = f" (no {folder}.csv in {courses_dir}, using PDF text for course tables)"
            else:
                table_chunks = course_chunks(rows, path.name, settings.chunk_size)
                chunks.extend(table_chunks)
                covered = {int(r["page"]) for r in rows}
                note = f" ({len(rows)} courses from {folder}.csv, pages {sorted(covered)})"

        for text, page in load_file(path):
            if page in covered:
                continue
            for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(piece, path.name, page, plan, programme))
        print(f"  read {folder}/{path.name} [{plan}/{programme}]{note}")
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
    vectors = embedder.embed_documents([c.text for c in chunks])

    collection.upsert(
        ids=[chunk_id(c) for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=vectors,
        metadatas=[{"source": c.source, "page": c.page or 0, "plan": c.plan, "programme": c.programme}
                   for c in chunks],
    )
    print(f"Done. Collection '{settings.collection}' now has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
