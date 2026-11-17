"""Ingest and retriever tests with temporary folders and fakes — no Ollama or Chroma needed."""
import csv

import pytest

from app import ingest
from app.retriever import ChromaRetriever
from app.text import Chunk


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def docs(tmp_path, monkeypatch):
    raw, courses = tmp_path / "raw", tmp_path / "courses"
    (raw / "all").mkdir(parents=True)
    (raw / "bcs-241").mkdir()
    courses.mkdir()
    (raw / "all" / "rules.txt").write_text("Students may take at most 20 credit hours.")
    (raw / "bcs-241" / "plan.pdf").write_bytes(b"")          # read through the fake load_file
    write_csv(courses / "bcs-241.csv", [{
        "plan": "241", "programme": "BCS", "category": "Final Year Project",
        "course_code": "BICS 4321", "course_name": "Final Year Project I", "credit_hours": "3",
        "prerequisite": ">=90 Credits", "offering_dept": "DCS", "page": "1"}])

    real_load = ingest.load_file

    def fake_load(path):
        if path.name == "plan.pdf":
            return [("scrambled course table text", 1), ("YEAR 1 SEMESTER 1 BICS 1301", 3)]
        return real_load(path)

    monkeypatch.setattr(ingest, "load_file", fake_load)
    monkeypatch.setitem(ingest.STUDY_PLANS, "bcs-241", "plan.pdf")
    return raw, courses


def test_chunks_are_tagged_from_their_folder(docs):
    raw, courses = docs
    chunks = ingest.build_chunks(raw, courses)
    tags = {(c.source, c.plan, c.programme) for c in chunks}
    assert tags == {("rules.txt", "all", "all"), ("plan.pdf", "241", "BCS")}


def test_study_plan_pages_come_from_csv_other_pages_from_pdf(docs):
    raw, courses = docs
    plan = [c for c in ingest.build_chunks(raw, courses) if c.source == "plan.pdf"]
    texts = [(c.page, c.text) for c in plan]
    assert (1, "BCS study plan 241, Final Year Project:\n"
               "- BICS 4321 Final Year Project I: 3 credit hours; prerequisite: >=90 Credits; offered by DCS") in texts
    assert (3, "YEAR 1 SEMESTER 1 BICS 1301") in texts
    assert not any("scrambled" in t for _, t in texts)         # page 1 PDF text replaced by the CSV


def test_missing_csv_falls_back_to_pdf_text(docs):
    raw, courses = docs
    (courses / "bcs-241.csv").unlink()
    plan = [c for c in ingest.build_chunks(raw, courses) if c.source == "plan.pdf"]
    assert [c.text for c in plan] == ["scrambled course table text", "YEAR 1 SEMESTER 1 BICS 1301"]


def test_file_outside_a_folder_is_an_error(docs):
    raw, courses = docs
    (raw / "loose.txt").write_text("not tagged")
    with pytest.raises(SystemExit):
        ingest.build_chunks(raw, courses)


class FakeCollection:
    def __init__(self):
        self.where = "unset"

    def count(self):
        return 1

    def query(self, query_embeddings, n_results, where=None):
        self.where = where
        return {"documents": [["FYP needs 90 credits"]],
                "metadatas": [[{"source": "plan.pdf", "page": 1, "plan": "241", "programme": "BCS"}]]}


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0] * 3


def test_search_filters_by_plan_and_programme():
    col = FakeCollection()
    chunks = ChromaRetriever(col, FakeEmbedder()).search("FYP", plan="241", programme="bcs")
    assert col.where == {"$or": [{"$and": [{"plan": "241"}, {"programme": "BCS"}]}, {"plan": "all"}]}
    assert chunks == [Chunk("FYP needs 90 credits", "plan.pdf", 1, "241", "BCS")]


def test_search_without_plan_searches_everything():
    col = FakeCollection()
    ChromaRetriever(col, FakeEmbedder()).search("FYP")
    assert col.where is None


def test_search_needs_both_plan_and_programme():
    with pytest.raises(ValueError):
        ChromaRetriever(FakeCollection(), FakeEmbedder()).search("FYP", plan="241")


def test_embedder_adds_task_prefixes(monkeypatch):
    from app.embeddings import OllamaEmbedder

    emb = OllamaEmbedder(doc_prefix="search_document: ", query_prefix="search_query: ")
    seen = []
    monkeypatch.setattr(emb, "embed", lambda texts: seen.extend(texts) or [[0.0]] * len(texts))
    emb.embed_documents(["FYP rules"])
    emb.embed_query("When can I take FYP?")
    assert seen == ["search_document: FYP rules", "search_query: When can I take FYP?"]
