"""How often does search find the right page? A baseline before the agent loop.

    python scripts/check_retrieval.py              # top-k from .env, filtered by plan
    python scripts/check_retrieval.py --k 10
    python scripts/check_retrieval.py --no-filter  # search every plan, to compare

For each question in eval/questions.csv that has a source_file, the question
is searched as-is (no query rewriting) and counts as a hit when one of the
top-k chunks comes from the expected file and page. Questions tagged with a
plan and programme are searched with that filter; 'all' questions search
everything. Needs Ollama running and `python -m app.ingest` done first.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.retriever import ChromaRetriever  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--no-filter", action="store_true", help="ignore the question's plan and programme")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.csv")
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8-sig", newline="") as f:
        questions = [q for q in csv.DictReader(f) if q["source_file"].strip()]

    retriever = ChromaRetriever()
    page_hits = file_hits = 0
    print(f"{'id':<5}{'hit':<6}{'expected':<45}top result")
    for q in questions:
        filtered = not args.no_filter and q["plan"] != "all"
        kwargs = {"plan": q["plan"], "programme": q["programme"]} if filtered else {}
        results = retriever.search(q["question"], k=args.k, **kwargs)

        want_file, want_page = q["source_file"], int(q["page"]) if q["page"] else None
        on_file = [c for c in results if c.source == want_file]
        on_page = [c for c in on_file if want_page is None or c.page == want_page]
        page_hits += bool(on_page)
        file_hits += bool(on_file)

        mark = "page" if on_page else ("file" if on_file else "MISS")
        top = results[0].citation() if results else "-"
        print(f"{q['id']:<5}{mark:<6}{(want_file[:34] + f', p.{want_page}'):<45}{top[:60]}")

    n = len(questions)
    print(f"\nright page in top {args.k}: {page_hits}/{n} ({page_hits / n:.0%})"
          f"   right file: {file_hits}/{n} ({file_hits / n:.0%})"
          f"   {'no filter' if args.no_filter else 'filtered by plan'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
