"""Pure helpers: chunking, batch tags, course tables, JSON extraction,
context formatting.

No network or database access here, so everything is easy to unit-test.
"""
from __future__ import annotations

import json
import re
from collections.abc import Hashable
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    source: str
    page: int | None = None
    plan: str = "all"          # "201", "241", "261" or "all"
    programme: str = "all"     # "BCS", "BIT" or "all"

    def citation(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


# ------------------------------------------------------------ batch tags --

PROGRAMMES = {"BCS", "BIT"}


def folder_tags(folder: str) -> tuple[str, str]:
    """Folder under data/raw -> (plan, programme).

    'all' -> ('all', 'all'); 'bcs-241' -> ('241', 'BCS')."""
    if folder == "all":
        return "all", "all"
    m = re.fullmatch(r"(bcs|bit)-(\d{3})", folder)
    if not m:
        raise ValueError(f"unknown document folder '{folder}': expected 'all' or e.g. 'bcs-241'")
    return m.group(2), m.group(1).upper()


def plan_for(batch: int | str, programme: str) -> str:
    """Which study plan a student follows, from their batch (e.g. 251, or a
    matric number starting with it) and programme.

    201-231 -> 201; 241-251 -> 241; 261+ -> 261 for BIT. BCS 261+ has no plan
    of its own yet and is assumed to follow 241 (see sources.md)."""
    programme = programme.strip().upper()
    if programme not in PROGRAMMES:
        raise ValueError(f"programme must be BCS or BIT, not '{programme}'")
    digits = re.sub(r"\D", "", str(batch))
    if len(digits) < 3:
        raise ValueError(f"batch should look like 241, not '{batch}'")
    b = int(digits[:3])
    if b < 201:
        raise ValueError(f"batch {b} is older than the 201 plan; no study plan for it")
    if b < 241:
        return "201"
    if b < 261 or programme == "BCS":
        return "241"
    return "261"


def plan_filter(plan: str, programme: str) -> dict:
    """Chroma `where` filter: the student's own plan and programme, plus
    documents that apply to everyone."""
    return {"$or": [
        {"$and": [{"plan": plan}, {"programme": programme}]},
        {"plan": "all"},
    ]}


# ---------------------------------------------------------- keyword search --

# Short words that sit before a 4-digit number without being a course prefix:
# "Semester II 2026/2027", "Dec 2025", "Year 2027".
NOT_PREFIXES = ("I|II|III|IV|V|VI|VII|VIII|IX|X|JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|SEPT|OCT|NOV|DEC"
                "|YEAR|SEM|WEEK|DAY|PAGE|NO|AND|THE|FOR|OF|IN|ON|AT|BY|TO|IS|OR")
CODE_RE = re.compile(rf"\b(?!(?:{NOT_PREFIXES})\b)([A-Z]{{2,4}})\s?(\d{{4}})\b")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "before", "can", "do", "does", "for", "from",
    "how", "i", "i'm", "im", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "what", "when", "where", "which", "who", "will", "with", "you",
    "m", "s", "t", "d", "ll", "re", "ve",     # left over from splitting "I'm", "student's", "don't"
}


def codes_in(text: str) -> list[str]:
    """Course codes in a string, normalised: 'BIIT4321' -> 'BIIT 4321'."""
    return [f"{a} {b}" for a, b in CODE_RE.findall(text.upper())]


def drop_invented_codes(query: str, question: str) -> str:
    """Remove course codes from an LLM-written query that the student's
    question doesn't contain. 'BICS 3062 prerequisite' for a question about
    Data Structures -> 'prerequisite'. Codes the student wrote are kept.

    Only upper-case codes count here (that is how the model writes them):
    matching any case would also catch words like 'exam 2026'."""
    allowed = set(codes_in(question))

    def keep(m: re.Match) -> str:
        return m.group(0) if f"{m.group(1)} {m.group(2)}" in allowed else ""

    return re.sub(r"\s+", " ", CODE_RE.sub(keep, query)).strip()


def fit_context(chunks: list[Chunk], max_chars: int) -> list[Chunk]:
    """Keep chunks, in order, while their total size fits in max_chars
    (always at least one). Ollama gives models a 4096-token window by default
    and silently cuts longer prompts from the front, which removes the system
    prompt, so the agent must never send more than fits."""
    out: list[Chunk] = []
    used = 0
    for c in chunks:
        size = len(c.text) + len(c.citation()) + 10     # format_context adds "[n] (citation)\n"
        if out and used + size > max_chars:
            continue                                     # a later, smaller chunk may still fit
        out.append(c)
        used += size
    return out


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens for keyword search, minus stopwords. A course code
    also becomes one token ('biit4321'), so it matches as a unit."""
    words = [w for w in re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", text.lower()) if w not in STOPWORDS]
    return [singular(w) for w in words] + [c.replace(" ", "").lower() for c in codes_in(text)]


def singular(word: str) -> str:
    """Crude plural stripping so 'prerequisites' matches 'prerequisite' and
    'courses' matches 'course'. Applied to queries and documents alike, so
    it only has to be consistent, not correct English."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


class BM25:
    """Okapi BM25 over a fixed list of documents. Rewards rare words that
    appear in both the query and a document, so exact names and course codes
    ('Final Year Project I', 'BIIT 4321') count for more than in embeddings."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        import math
        from collections import Counter

        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = (sum(self.lengths) / len(docs)) if docs else 0.0
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = set(tokenize(query))
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len) if self.avg_len else self.k1
            for t in terms:
                if t in tf:
                    s += self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm)
            out.append(s)
        return out

    def top(self, query: str, n: int) -> list[int]:
        """Indexes of the n best documents with a score above zero."""
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        return [i for _, i in sorted(scored, reverse=True)[:n]]


def rrf(rankings: list[list[Hashable]], k: int = 60) -> list[Hashable]:
    """Reciprocal rank fusion: merge ranked id lists; an id high in either
    list ends up high in the result."""
    score: dict[Hashable, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] = score.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(score, key=lambda i: -score[i])


def fuse(rankings: list[list[Hashable]], k: int, guaranteed: int = 0) -> list[Hashable]:
    """Top k of the rank-fused lists, where the first `guaranteed` items of
    every list are always included. Plain fusion gives an item nothing for
    lists it is missing from, so a course table that only keyword search
    finds (rank 2 there, absent from the embedding list) ends up 6th or 7th;
    guaranteed slots keep each method's best hits. Result is in fused order."""
    order = rrf(rankings)
    must = list(dict.fromkeys(i for ranking in rankings for i in ranking[:guaranteed]))[:k]
    chosen = set(must)
    for i in order:
        if len(chosen) >= k:
            break
        chosen.add(i)
    return [i for i in order if i in chosen]


def matches_filter(meta: dict, plan: str | None, programme: str | None) -> bool:
    """Python version of plan_filter, for searches done outside Chroma."""
    if plan is None:
        return True
    return meta.get("plan") == "all" or (meta.get("plan") == plan and meta.get("programme") == programme)


# --------------------------------------------------------- course tables --

def course_line(row: dict) -> str:
    """One CSV row as a sentence the embedder and the LLM can both read."""
    code = f"{row['course_code']} " if row.get("course_code") else ""
    parts = [f"- {code}{row['course_name']}: {row['credit_hours'] or '?'} credit hours",
             f"prerequisite: {row.get('prerequisite') or 'none'}"]
    if row.get("offering_dept"):
        parts.append(f"offered by {row['offering_dept']}")
    return "; ".join(parts)


def course_chunks(rows: list[dict], source: str, size: int = 900) -> list[Chunk]:
    """Turn a plan's course CSV rows into chunks: one or more per category
    and page, each starting with a header so it makes sense on its own."""
    chunks: list[Chunk] = []
    groups: list[tuple[tuple, list[dict]]] = []
    for row in rows:
        key = (row["plan"], row["programme"], row["category"], int(row["page"]))
        if groups and groups[-1][0] == key:
            groups[-1][1].append(row)
        else:
            groups.append((key, [row]))

    for (plan, programme, category, page), group in groups:
        header = f"{programme} study plan {plan}, {category}:"
        body: list[str] = []
        for line in map(course_line, group):
            if body and len(header) + sum(len(l) + 1 for l in body) + len(line) + 1 > size:
                chunks.append(Chunk("\n".join([header, *body]), source, page, plan, programme))
                body = []
            body.append(line)
        chunks.append(Chunk("\n".join([header, *body]), source, page, plan, programme))
    return chunks


def normalize(text: str) -> str:
    """Collapse whitespace that PDF extraction tends to leave behind."""
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into ~size-char chunks with overlap, preferring paragraph,
    then sentence, then word boundaries."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    text = normalize(text)
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            # Prefer breaking at a paragraph, then a sentence, then a space —
            # but only if that keeps the chunk reasonably large.
            for sep in ("\n\n", ". ", " "):
                cut = window.rfind(sep)
                if cut > size * 0.5:
                    end = start + cut + len(sep)
                    break
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_json(raw: str) -> dict:
    """Pull the first JSON object out of an LLM reply.

    Small local models often wrap JSON in ```json fences or add chatter
    around it; this tolerates both. Returns {} if nothing parses.
    """
    raw = re.sub(r"```(?:json)?", "", raw)
    depth, start = 0, None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(raw[start : i + 1])
                except json.JSONDecodeError:
                    start = None
    return {}


def format_context(chunks: list[Chunk]) -> str:
    """Number excerpts so the model can cite them as [1], [2], ..."""
    return "\n\n".join(f"[{i}] ({c.citation()})\n{c.text}" for i, c in enumerate(chunks, 1))


def dedupe(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, int | None, str]] = set()
    out: list[Chunk] = []
    for c in chunks:
        key = (c.source, c.page, c.text[:80])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
