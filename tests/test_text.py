import pytest

from app.text import Chunk, chunk_text, dedupe, extract_json, format_context


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_empty_text_gives_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_chunks_respect_size_and_cover_text():
    text = " ".join(f"Sentence number {i} about academic rules." for i in range(200))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    assert "Sentence number 0" in chunks[0]
    assert "Sentence number 199" in chunks[-1]


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_text("abc", size=10, overlap=10)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"query": "add drop"}', {"query": "add drop"}),
        ('```json\n{"sufficient": true}\n```', {"sufficient": True}),
        ('Sure! Here it is: {"a": {"b": 1}} hope that helps', {"a": {"b": 1}}),
        ("no json here", {}),
    ],
)
def test_extract_json(raw, expected):
    assert extract_json(raw) == expected


def test_format_context_numbers_and_cites():
    out = format_context([Chunk("rule A", "handbook.pdf", 4), Chunk("rule B", "notes.txt")])
    assert "[1] (handbook.pdf, p.4)" in out
    assert "[2] (notes.txt)" in out


def test_dedupe_keeps_order():
    a, b = Chunk("x", "f.pdf", 1), Chunk("y", "f.pdf", 2)
    assert dedupe([a, b, Chunk("x", "f.pdf", 1)]) == [a, b]
