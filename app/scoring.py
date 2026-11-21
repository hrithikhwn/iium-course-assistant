"""Scoring for the eval: do an answer and its sources contain the key facts?

Pure functions (no LLM, no database), used by scripts/run_eval.py and
scripts/check_retrieval.py.

Key facts come from the `key_facts` column of eval/questions.csv. Facts are
separated by ";", a fact can have alternatives separated by "|", and "&"
joins parts that must all appear:

    1.67; 2.00; 18 credit|eighteen credit
    7 credit|3 credit & 4 credit; 90        (the total, or FYP I + FYP II)

Matching ignores case, punctuation and spacing ("eighteen (18) credits"
contains "18 credit"), and "BIIT4321" matches "BIIT 4321". A fact must start
at a word boundary, and one that ends in a digit must end at one too, so "18"
doesn't match "180" while "18 credit" matches "18 credits".
"""
from __future__ import annotations

import re
import statistics
from collections.abc import Iterable

from .text import CITATION_RE, Chunk, codes_in

Facts = list[list[str]]     # one list of alternatives per fact


def parse_facts(cell: str) -> Facts:
    """'1.67; 18 credit|eighteen credit' -> [['1.67'], ['18 credit', 'eighteen credit']]"""
    facts = []
    for fact in cell.split(";"):
        alternatives = [a.strip() for a in fact.split("|") if a.strip()]
        if alternatives:
            facts.append(alternatives)
    return facts


def question_facts(q: dict) -> Facts:
    """An eval question's key facts: its key_facts column, or else the course
    codes in the first sentence of expected_answer (later sentences are notes)."""
    if (q.get("key_facts") or "").strip():
        return parse_facts(q["key_facts"])
    first = re.split(r"(?<=[.!?])\s", (q.get("expected_answer") or "").strip(), maxsplit=1)[0]
    return [[code] for code in codes_in(first)]


def normalise(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w./%]+", " ", text)                     # punctuation -> space; keep . / %
    text = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", text)            # dots only inside numbers: 1.67
    text = re.sub(r"\b([a-z]{2,4}) ?(\d{4})\b", r"\1 \2", text)  # biit4321 -> biit 4321
    return " ".join(text.split())


def _found(norm_text: str, alternative: str) -> bool:
    parts = [normalise(p) for p in alternative.split("&")]
    return all(p and re.search(rf"(?<![\w.]){re.escape(p)}" + (r"(?!\d)" if p[-1].isdigit() else ""),
                               norm_text) for p in parts)


def missing_facts(text: str, facts: Facts) -> list[str]:
    """The facts the text doesn't contain (the first alternative of each)."""
    norm = normalise(text)
    return [alts[0] for alts in facts if not any(_found(norm, a) for a in alts)]


def missing_from(chunks: Iterable[Chunk], facts: Facts) -> list[str]:
    """The facts none of the chunks contain. Each fact may come from a
    different chunk: an answer can combine excerpts."""
    texts = [normalise(c.text) for c in chunks]
    return [alts[0] for alts in facts if not any(_found(t, a) for t in texts for a in alts)]


def rank_covering(chunks: list[Chunk], facts: Facts, source: str | None = None) -> int | None:
    """1-based position in `chunks` by which the chunks so far (only those from
    `source`, if given) together contain every fact; None if they never do."""
    remaining = list(facts)
    for i, c in enumerate(chunks, 1):
        if source is not None and c.source != source:
            continue
        norm = normalise(c.text)
        remaining = [alts for alts in remaining if not any(_found(norm, a) for a in alts)]
        if not remaining:
            return i
    return None


# ------------------------------------------------------------ one answer --

def score_answer(q: dict, answer: str, found: bool, sources: list[Chunk], context: list[Chunk]) -> dict:
    """Score one agent result against an eval question.

    correct    'refuse' questions: the agent said not found. Others: it answered
               and the answer contains every key fact ("" if the question has none)
    supported  the cited excerpts together contain every key fact
    retrieved  the excerpts the model was shown contain every key fact
    cites_source  a cited excerpt is from the expected file (and page, if given)
    """
    if q.get("type") == "refuse":
        return {"correct": not found, "missing": "", "supported": "", "retrieved": "", "cites_source": ""}
    facts = question_facts(q)
    want_file, want_page = q.get("source_file") or "", q.get("page") or ""
    cites = "" if not want_file else any(
        c.source == want_file and (not want_page or c.page == int(want_page)) for c in sources)
    if not facts:
        return {"correct": "", "missing": "", "supported": "", "retrieved": "", "cites_source": cites}
    # citation numbers aren't facts: "[4]" mustn't count as the fact "4"
    text = CITATION_RE.sub(" ", answer)
    missing = missing_facts(text, facts) if found else [alts[0] for alts in facts]
    return {
        "correct": found and not missing,
        "missing": "; ".join(missing),
        "supported": found and not missing_from(sources, facts),
        "retrieved": not missing_from(context, facts),
        "cites_source": cites,
    }


# --------------------------------------------------------------- one run --

def summarise(rows: list[dict]) -> dict:
    """Totals for one run. Rows are score_answer() results plus 'type',
    'found', 'seconds', 'llm_calls' and 'error'."""
    ok = [r for r in rows if not r.get("error")]
    refuse = [r for r in ok if r.get("type") == "refuse"]
    scored = [r for r in ok if r.get("type") != "refuse" and r.get("correct") != ""]
    seconds = [float(r["seconds"]) for r in ok]
    return {
        "answers": len(rows),
        "errors": len(rows) - len(ok),
        "answerable": len(scored),
        "correct": sum(r["correct"] is True for r in scored),
        "false_refusals": sum(not r["found"] for r in scored),
        "supported": sum(r["supported"] is True for r in scored),
        "retrieved": sum(r["retrieved"] is True for r in scored),
        "should_refuse": len(refuse),
        "refused": sum(r["correct"] is True for r in refuse),
        "median_seconds": round(statistics.median(seconds), 1) if seconds else "",
        "mean_llm_calls": round(statistics.mean(r["llm_calls"] for r in ok), 1) if ok else "",
    }
