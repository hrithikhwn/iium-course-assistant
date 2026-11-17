"""Pure helpers: chunking, JSON extraction, context formatting.

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

    def citation(self) -> str:
        return f"{self.source}, p.{self.page}" if self.page else self.source


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
