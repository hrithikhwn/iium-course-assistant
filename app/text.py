"""Pure helpers: chunking, batch tags, course tables, JSON extraction,
context formatting.

No network or database access here, so everything is easy to unit-test.
"""
from __future__ import annotations

import json
import re
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
