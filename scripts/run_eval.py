"""Run the eval questions through the agent and score the answers.

    python scripts/run_eval.py                        # every question, model from .env
    python scripts/run_eval.py --model qwen2.5:3b     # another model at LLM_BASE_URL
    python scripts/run_eval.py --variant plain        # or no-rewrite, strict (see VARIANTS)
    python scripts/run_eval.py --ids q02,q11 --repeat 3
    python scripts/run_eval.py --rescore eval/results/<run>.csv   # after editing key facts

Each run writes eval/results/<date>_<time>_<model>_<variant>.csv, one row per
answer with its scores, cited sources and trace, and a line in
eval/results/summary.csv. Next to the CSV, <run>.jsonl keeps the excerpts
each answer saw (document text, so git-ignored); --rescore uses it to score
the run again with the current key facts, without asking the model again.
Needs the LLM endpoint from .env (Ollama by default) and `python -m app.ingest`
done first.

Scores (see app/scoring.py):
  correct    answerable questions: it answered, and the answer has every key
             fact; 'refuse' questions: it said it couldn't find it
  supported  the cited excerpts together contain every key fact
  retrieved  the excerpts the model was shown contain every key fact
Each question is asked as the student its `batch` column names (else the
first batch of its plan), and questions about everyone as BCS 241.
"""
from __future__ import annotations

import argparse
import csv
import inspect
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.agent import CourseAgent  # noqa: E402
from app.config import settings  # noqa: E402
from app.llm import OpenAICompatibleLLM  # noqa: E402
from app.retriever import ChromaRetriever  # noqa: E402
from app.scoring import score_answer, summarise  # noqa: E402
from app.text import Chunk  # noqa: E402

# The agent's switches for each variant, all set here so .env can't change what
# a variant means (other settings, like TOP_K, still come from .env):
#   agentic     plan (rewrite), retrieve, grade, retry, answer
#   no-rewrite  the same, searching with the question only
#   plain       plain RAG: one search with the question, then the answer
#   strict      agentic, but "not found" when the last grade still fails
VARIANTS = {
    "agentic": {"rewrite": True, "grading": True, "refuse_on_failed_grade": False},
    "no-rewrite": {"rewrite": False, "grading": True, "refuse_on_failed_grade": False},
    "plain": {"rewrite": False, "grading": False, "refuse_on_failed_grade": False},
    "strict": {"rewrite": True, "grading": True, "refuse_on_failed_grade": True},
}
FIELDS = ["id", "type", "repeat", "correct", "found", "missing", "supported", "retrieved", "cites_source",
          "seconds", "llm_calls", "context", "cited", "answer", "question", "trace", "error"]
SUMMARY_FIELDS = ["run", "commit", "model", "variant", "repeat", "answers", "errors", "answerable", "correct",
                  "false_refusals", "supported", "retrieved", "should_refuse", "refused",
                  "median_seconds", "mean_llm_calls"]


class CountingLLM:
    """Passes chat calls through and counts them: what the agentic steps cost."""

    def __init__(self, llm):
        self.llm = llm
        self.calls = 0

    def chat(self, system: str, user: str, temperature: float = 0.0) -> str:
        self.calls += 1
        return self.llm.chat(system, user, temperature)


def student(q: dict) -> tuple[int, str]:
    """Who asks: the batch in the question's batch column (else the first batch
    of its plan) and its programme; BCS 241 for questions about everyone."""
    batch = (q.get("batch") or "").strip() or ("241" if q["plan"] == "all" else q["plan"])
    return int(batch), "BCS" if q["programme"] == "all" else q["programme"]


def code_version() -> str:
    """Commit the run used, with '+' if the code or questions had uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "app", "scripts", "eval/questions.csv"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return head + ("+" if dirty else "")
    except OSError:
        return ""


def verdict(row: dict) -> str:
    if row["correct"] == "":
        return "unscored"
    if row["correct"]:
        return "ok"
    if row["type"] == "refuse":
        return "WRONG (answered)"
    return "WRONG (not found)" if not row["found"] else f"WRONG (missing {row['missing']})"


# ----------------------------------------------------------- result files --

def excerpts(chunks: list[Chunk]) -> list[dict]:
    return [{"source": c.source, "page": c.page, "text": c.text} for c in chunks]


def chunks_from(excerpt_dicts: list[dict]) -> list[Chunk]:
    return [Chunk(d["text"], d["source"], d["page"]) for d in excerpt_dicts]


def typed(row: dict) -> dict:
    """A row read back from a results CSV, with its booleans and numbers restored."""
    out = dict(row)
    for key in ("correct", "found", "supported", "retrieved", "cites_source"):
        value = row.get(key, "")
        out[key] = value if isinstance(value, bool) else {"True": True, "False": False}.get(value, "")
    if not row.get("error"):
        out["seconds"], out["llm_calls"] = float(row["seconds"]), int(row["llm_calls"])
    return out


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_summary(out: Path) -> list[dict]:
    path = out / "summary.csv"
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_summary(out: Path, line: dict) -> None:
    """Add the run's line to summary.csv, or replace it (after a rescore)."""
    lines = read_summary(out)
    i = next((i for i, old in enumerate(lines) if old["run"] == line["run"]), None)
    if i is None:
        lines.append(line)
    else:
        lines[i] = line
    with (out / "summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(lines)


def print_summary(s: dict, path: Path) -> None:
    print(f"\nanswered correctly      {s['correct']}/{s['answerable']}   (said not found: {s['false_refusals']})")
    print(f"supported by citations  {s['supported']}/{s['answerable']}")
    print(f"facts were retrieved    {s['retrieved']}/{s['answerable']}")
    print(f"refused correctly       {s['refused']}/{s['should_refuse']}")
    print(f"median {s['median_seconds']}s per question, {s['mean_llm_calls']} LLM calls on average"
          + (f", {s['errors']} errors" if s["errors"] else ""))
    print(f"\n{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")


# -------------------------------------------------------------------- runs --

def rescore(path: Path, questions: list[dict]) -> int:
    """Score a finished run again with the current key facts."""
    sidecar = path.with_suffix(".jsonl")
    if not sidecar.exists():
        raise SystemExit(f"{sidecar} not found: it is written next to each run's CSV")
    with sidecar.open(encoding="utf-8") as f:
        seen = {(d["id"], d["repeat"]): d for d in map(json.loads, f)}
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = [typed(r) for r in csv.DictReader(f)]
    by_id = {q["id"]: q for q in questions}
    for row in rows:
        d = seen.get((row["id"], int(row["repeat"])))
        if d and not row["error"] and row["id"] in by_id:
            row.update(score_answer(by_id[row["id"]], d["answer"], d["found"],
                                    chunks_from(d["sources"]), chunks_from(d["context"])))
            print(f"{row['id']:<5} {verdict(row)}")
    write_rows(path, rows)
    s = summarise(rows)
    old = next((line for line in read_summary(path.parent) if line["run"] == path.stem), {"run": path.stem})
    write_summary(path.parent, {**old, **s})
    print_summary(s, path)
    return 0


def run(args, questions: list[dict]) -> int:
    options = VARIANTS[args.variant]
    missing = sorted(set(options) - set(inspect.signature(CourseAgent).parameters))
    if missing:
        raise SystemExit(f"variant '{args.variant}' needs CourseAgent({', '.join(missing)}=...), "
                         "which app/agent.py doesn't have yet")

    llm = CountingLLM(OpenAICompatibleLLM(model=args.model))
    retriever = ChromaRetriever()
    agent = CourseAgent(llm, retriever, **options)
    try:   # load both models first, so the first question's time doesn't include it
        llm.chat("Reply with OK.", "OK")
        retriever.search("warm up", k=1, plan="241", programme="BCS")
    except Exception as e:
        raise SystemExit(f"can't reach the model '{args.model}' or the vector store: {e}")

    name = f"{datetime.now():%Y-%m-%d_%H%M}_{re.sub(r'[^A-Za-z0-9.]+', '-', args.model)}_{args.variant}"
    print(f"{name}: {len(questions)} questions x {args.repeat}\n")
    rows, seen = [], []
    for rep in range(1, args.repeat + 1):
        for q in questions:
            batch, programme = student(q)
            llm.calls = 0
            row = {"id": q["id"], "type": q["type"], "repeat": rep, "question": q["question"]}
            start = time.perf_counter()
            try:
                result = agent.run(q["question"], batch=batch, programme=programme)
            except Exception as e:     # a timeout or a dropped connection: note it and go on
                row.update(error=f"{type(e).__name__}: {e}", seconds=round(time.perf_counter() - start, 1),
                           llm_calls=llm.calls)
                rows.append(row)
                print(f"{q['id']:<5} ERROR {row['error'][:100]}")
                continue
            row.update(score_answer(q, result.answer, result.found, result.sources, result.context))
            row.update(found=result.found, seconds=round(result.seconds, 1), llm_calls=llm.calls,
                       context=len(result.context), cited="; ".join(c.citation() for c in result.sources),
                       answer=result.answer, trace=" | ".join(f"{s.name}: {s.detail}" for s in result.steps),
                       error="")
            rows.append(row)
            seen.append({"id": q["id"], "repeat": rep, "found": result.found, "answer": result.answer,
                         "sources": excerpts(result.sources), "context": excerpts(result.context)})
            print(f"{q['id']:<5} {row['seconds']:>5.1f}s {row['llm_calls']} calls  {verdict(row)}")

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{name}.csv"
    write_rows(path, rows)
    with path.with_suffix(".jsonl").open("w", encoding="utf-8") as f:
        f.writelines(json.dumps(d, ensure_ascii=False) + "\n" for d in seen)
    s = summarise(rows)
    write_summary(args.out, {"run": name, "commit": code_version(), "model": args.model,
                             "variant": args.variant, "repeat": args.repeat, **s})
    print_summary(s, path)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=settings.llm_model, help="model name at LLM_BASE_URL (default: .env)")
    parser.add_argument("--variant", choices=VARIANTS, default="agentic")
    parser.add_argument("--ids", help="only these questions, e.g. q02,q11")
    parser.add_argument("--repeat", type=int, default=1, help="ask each question this many times")
    parser.add_argument("--questions", type=Path, default=ROOT / "eval" / "questions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "eval" / "results")
    parser.add_argument("--rescore", type=Path, metavar="RUN_CSV", help="score a finished run again, no LLM")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    with args.questions.open(encoding="utf-8-sig", newline="") as f:
        questions = list(csv.DictReader(f))
    if args.rescore:
        return rescore(args.rescore, questions)
    if args.ids:
        wanted = {i.strip() for i in args.ids.split(",")}
        if unknown := wanted - {q["id"] for q in questions}:
            raise SystemExit(f"no such question: {', '.join(sorted(unknown))}")
        questions = [q for q in questions if q["id"] in wanted]
    return run(args, questions)


if __name__ == "__main__":
    sys.exit(main())
