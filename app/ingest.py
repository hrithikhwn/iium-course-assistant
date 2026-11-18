"""Ingest PDFs / text files from data/raw into a Chroma vector store.

Usage:
    python -m app.ingest            # sync the store with data/raw and data/courses
    python -m app.ingest --reset    # wipe the collection first (needed after changing
                                    # EMBED_MODEL or the EMBED_*_PREFIX settings)

The store is synced, not just added to: chunks whose text changed (an edited
CSV row, a replaced PDF) get new ids, and the old versions are deleted so
they can't be retrieved any more.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

from .config import settings
from .text import Chunk, chunk_text, course_chunks, folder_tags, semester_chunks

SUPPORTED = {".pdf", ".txt", ".md"}

# Study plans whose tables were extracted to hand-checked CSVs: the course
# tables to data/courses/<folder>.csv (scripts/extract_tables.py), the
# semester-by-semester plan to data/courses/<folder>-semesters.csv
# (scripts/extract_semesters.py). pypdf scrambles those tables, so their
# pages are indexed from the CSVs; any other page from the PDF text.
STUDY_PLANS = {
    "bcs-201": "STUDY PLAN FOR BCS BATCH 201.pdf",
    "bit-201": "STUDY PLAN FOR BIT BATCH 201.pdf",
    "bcs-241": "BCS Study Plan (Batch 241) Rev_3.pdf",
    "bit-241": "BIT-Study-Plan-Batch-241-Rev-3_2.pdf",
    "bit-261": "BIT-PROGRAMME-STRUCTURE-AND-STUDY-PLAN-261.pdf",
}

CSV_COLUMNS = {"plan", "programme", "category", "course_code", "course_name",
               "credit_hours", "prerequisite", "offering_dept", "page"}
SEMESTER_COLUMNS = {"plan", "programme", "year", "semester", "course_code", "course_name",
                    "credit_hours", "prerequisite", "page"}


def load_file(path: Path) -> list[tuple[str, int | None]]:
    """Return (text, page_number) pairs. Text files have no page numbers."""
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(page.extract_text() or "", i) for i, page in enumerate(reader.pages, 1)]
    return [(path.read_text(encoding="utf-8", errors="ignore"), None)]


def read_plan_csv(path: Path, columns: set[str], folder: str, script: str) -> list[dict] | None:
    """Rows of one of a study plan's CSVs, checked, or None if there is no CSV.

    utf-8-sig: Excel adds a byte-order mark when it saves a CSV as UTF-8,
    which would otherwise turn the first column name into '\\ufeffplan'."""
    if not path.exists():
        return None
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = columns - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"{path}: missing columns {sorted(missing)} (re-run scripts/{script})")
        rows = list(reader)

    plan, programme = folder_tags(folder)
    for line, r in enumerate(rows, 2):            # line 1 is the header
        if (r["plan"], r["programme"]) != (plan, programme):
            raise SystemExit(f"{path} line {line}: plan/programme {r['plan']}/{r['programme']} "
                             f"doesn't match its folder {folder} ({plan}/{programme})")
        if not r["page"].strip().isdigit():
            raise SystemExit(f"{path} line {line}: page must be a page number, not {r['page']!r}")
    return rows


def load_course_csv(folder: str, courses_dir: Path) -> list[dict] | None:
    """Rows of data/courses/<folder>.csv, the plan's course tables."""
    return read_plan_csv(courses_dir / f"{folder}.csv", CSV_COLUMNS, folder, "extract_tables.py")


def load_semester_csv(folder: str, courses_dir: Path) -> list[dict] | None:
    """Rows of data/courses/<folder>-semesters.csv, the plan's semester-by-semester
    page. Year and semester are both numbers, or both empty on the rows for the
    total graduating credits."""
    path = courses_dir / f"{folder}-semesters.csv"
    rows = read_plan_csv(path, SEMESTER_COLUMNS, folder, "extract_semesters.py")
    for line, r in enumerate(rows or [], 2):
        year, semester = r["year"].strip(), r["semester"].strip()
        if not (year.isdigit() and semester.isdigit()) and (year or semester):
            raise SystemExit(f"{path} line {line}: year and semester must both be numbers or both be "
                             f"empty, not {year!r}/{semester!r}")
    return rows


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
            found = []
            rows = load_course_csv(folder, courses_dir)
            if rows is None:
                found.append(f"no {folder}.csv in {courses_dir}, using PDF text for course tables")
            else:
                chunks.extend(course_chunks(rows, path.name, settings.chunk_size))
                covered |= {int(r["page"]) for r in rows}
                found.append(f"{len(rows)} courses from {folder}.csv")
            semesters = load_semester_csv(folder, courses_dir)
            if semesters is not None:
                chunks.extend(semester_chunks(semesters, path.name, settings.chunk_size))
                covered |= {int(r["page"]) for r in semesters}
                found.append(f"{len(semesters)} semester plan rows from {folder}-semesters.csv")
            note = f" ({', '.join(found)}" + (f", pages {sorted(covered)})" if covered else ")")

        for text, page in load_file(path):
            if page in covered:
                continue
            for piece in chunk_text(text, settings.chunk_size, settings.chunk_overlap):
                chunks.append(Chunk(piece, path.name, page, plan, programme))
        print(f"  read {folder}/{path.name} [{plan}/{programme}]{note}")

    # a renamed study plan PDF would silently lose its CSV: say so
    for folder, name in STUDY_PLANS.items():
        if (raw_dir / folder).is_dir() and not (raw_dir / folder / name).is_file():
            print(f"  warning: {folder}/{name} not found, so its CSVs in data/courses are not used "
                  f"(update STUDY_PLANS in app/ingest.py if the file was renamed)")
    return chunks


def chunk_id(c: Chunk) -> str:
    # plan and programme are part of the id: the same file can sit in two folders
    return hashlib.sha1(f"{c.plan}|{c.programme}|{c.source}|{c.page}|{c.text}".encode()).hexdigest()


def unique_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """Drop exact duplicates (same id), which Chroma would reject in one upsert."""
    seen: set[str] = set()
    out = []
    for c in chunks:
        cid = chunk_id(c)
        if cid not in seen:
            seen.add(cid)
            out.append(c)
    return out


def sync_collection(collection, chunks: list[Chunk], vectors: list[list[float]], batch: int = 500) -> int:
    """Make the collection hold exactly `chunks`: upsert them and delete every
    other id (old versions of edited chunks). Returns how many were deleted."""
    ids = [chunk_id(c) for c in chunks]
    stale = sorted(set(collection.get(include=[])["ids"]) - set(ids))
    for i in range(0, len(stale), batch):
        collection.delete(ids=stale[i : i + batch])
    for i in range(0, len(chunks), batch):
        part = chunks[i : i + batch]
        collection.upsert(
            ids=ids[i : i + batch],
            documents=[c.text for c in part],
            embeddings=vectors[i : i + batch],
            metadatas=[{"source": c.source, "page": c.page or 0, "plan": c.plan, "programme": c.programme}
                       for c in part],
        )
    return len(stale)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reset", action="store_true", help="delete the collection before ingesting")
    args = parser.parse_args()

    from .embeddings import OllamaEmbedder
    from .retriever import chroma_client

    raw_dir = settings.raw_dir
    if not raw_dir.exists():
        raise SystemExit(f"{raw_dir} does not exist — create it and add your documents.")

    print(f"Reading documents from {raw_dir}/ ...")
    chunks = unique_chunks(build_chunks(raw_dir))
    if not chunks:
        raise SystemExit(f"No .pdf/.txt/.md files with text found in {raw_dir}/")

    client = chroma_client()
    if args.reset:
        try:
            client.delete_collection(settings.collection)
        except Exception:
            pass
    collection = client.get_or_create_collection(settings.collection)

    print(f"Embedding {len(chunks)} chunks with {settings.embed_model} ...")
    vectors = OllamaEmbedder().embed_documents([c.text for c in chunks])
    removed = sync_collection(collection, chunks, vectors)
    print(f"Done. Collection '{settings.collection}' now has {collection.count()} chunks"
          f"{f' ({removed} outdated chunks removed)' if removed else ''}.")


if __name__ == "__main__":
    main()
