"""How often does search find the answer? A baseline before the agent loop.

    python scripts/check_retrieval.py              # top-k from .env, filtered by plan
    python scripts/check_retrieval.py --k 10
    python scripts/check_retrieval.py --no-filter  # search every plan, to compare
    python scripts/check_retrieval.py --vector-only

For each question in eval/questions.csv that has a source_file, the question
is searched as-is (no query rewriting) and graded:

  fact  one of the top-k chunks contains every key fact of the answer
  page  a chunk from the expected file and page is in the top k
  file  a chunk from the expected file is in the top k
  MISS  none of these

Key facts come from an optional `key_facts` column (separated by ";"), or else
from the course codes in the first sentence of `expected_answer`. Questions with neither are graded
on page only, which is fine for long PDF text but too lenient for course
tables, where one page holds many chunks.

Questions tagged with a plan and programme are searched with that filter;
'all' questions search everything. Needs Ollama running and
`python -m app.ingest` done first.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import settings  # noqa: E402
from app.retriever import ChromaRetriever  # noqa: E402
from app.text import codes_in  # noqa: E402


def key_facts(q: dict) -> list[str]:
    if q.get("key_facts", "").strip():
        return [f.strip() for f in q["key_facts"].split(";") if f.strip()]
    # codes in the first sentence only: later sentences are notes, e.g. q12's
    # "the plan's page 2 lists the code as BICS 4902"
    first = re.split(r"(?<=[.!?])\s", q["expected_answer"].strip(), maxsplit=1)[0]
    return codes_in(first)


def has_facts(text: str, facts: list[str]) -> bool:
    flat = " ".join(text.split()).lower()
    return all(f.lower() in flat for f in facts)


def grade(q: dict, results) -> tuple[str, int | None]:
    """Best level reached, and the rank (1-based) where it was reached."""
    want_file, want_page = q["source_file"], int(q["page"]) if q["page"] else None
    facts = key_facts(q)
    levels = []
    if facts:
        levels.append(("fact", lambda c: c.source == want_file and has_facts(c.text, facts)))
    else:
        levels.append(("page", lambda c: c.source == want_file and (want_page is None or c.page == want_page)))
    levels.append(("file", lambda c: c.source == want_file))
    for name, ok in levels:
        rank = next((i for i, c in enumerate(results, 1) if ok(c)), None)
        if rank:
            return name, rank
    return "MISS", None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--k", type=int, default=settings.top_k)
    parser.add_argument("--no-filter", action="store_true", help="ignore the question's plan and programme")
    parser.add_argument("--vector-only", action="store_true", help="embeddings only, no keyword search")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.csv")
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8-sig", newline="") as f:
        questions = [q for q in csv.DictReader(f) if q["source_file"].strip()]

    retriever = ChromaRetriever()
    search = retriever.vector_search if args.vector_only else retriever.search
    tally = {"fact": 0, "page": 0, "file": 0, "MISS": 0}
    graded_on_facts = 0
    print(f"{'id':<5}{'hit':<6}{'rank':<6}{'needs':<38}top result")
    for q in questions:
        filtered = not args.no_filter and q["plan"] != "all"
        kwargs = {"plan": q["plan"], "programme": q["programme"]} if filtered else {}
        results = search(q["question"], k=args.k, **kwargs)
        level, rank = grade(q, results)
        tally[level] += 1
        facts = key_facts(q)
        graded_on_facts += bool(facts)
        needs = ", ".join(facts) if facts else f"p.{q['page']} of {q['source_file'][:22]}"
        top = results[0].citation() if results else "-"
        print(f"{q['id']:<5}{level:<6}{rank or '-':<6}{needs[:36]:<38}{top[:55]}")

    n = len(questions)
    good = tally["fact"] + tally["page"]
    mode = "vector only" if args.vector_only else "hybrid"
    print(f"\nanswer found in top {args.k}: {good}/{n} ({good / n:.0%})"
          f"   [{graded_on_facts} graded on key facts, {n - graded_on_facts} on page]"
          f"   wrong chunk, right file: {tally['file']}   miss: {tally['MISS']}"
          f"   ({mode}, {'no filter' if args.no_filter else 'filtered by plan'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
