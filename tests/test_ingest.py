"""Ingest and retriever tests with temporary folders and fakes — no Ollama or Chroma needed."""
import csv
from pathlib import Path

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
        if path.suffix == ".pdf":              # every test PDF is an empty file
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
    """An 'all' FAQ page and two course tables (BCS 241, BIT 241). query()
    always ranks the FAQ first, like embeddings did for course questions."""
    DOCS = {
        "faq": ("Registration FAQ: final year project placement and deadlines",
                {"source": "faq.pdf", "page": 7, "plan": "all", "programme": "all"}),
        "fyp": ("BCS study plan 241, Final Year Project:\n- BICS 4321 Final Year Project I: 3 credit hours",
                {"source": "plan.pdf", "page": 1, "plan": "241", "programme": "BCS"}),
        "bit": ("BIT study plan 241, Final Year Project:\n- BIIT 4321 Final Year Project I: 3 credit hours",
                {"source": "bit.pdf", "page": 1, "plan": "241", "programme": "BIT"}),
    }

    def __init__(self):
        self.where = "unset"

    def count(self):
        return len(self.DOCS)

    def query(self, query_embeddings, n_results, where=None):
        self.where = where
        # like the real q09: the course table is not among the embedding hits at all
        ids = ["faq"] if where else ["faq", "bit"]
        ids = ids[:n_results]
        return {"ids": [ids], "documents": [[self.DOCS[i][0] for i in ids]],
                "metadatas": [[self.DOCS[i][1] for i in ids]]}

    def get(self, include):
        return {"ids": list(self.DOCS), "documents": [d for d, _ in self.DOCS.values()],
                "metadatas": [m for _, m in self.DOCS.values()]}


class FakeEmbedder:
    def embed_query(self, text):
        return [0.0] * 3


def test_search_filters_by_plan_and_programme():
    col = FakeCollection()
    chunks = ChromaRetriever(col, FakeEmbedder()).search("BICS 4321", plan="241", programme="bcs")
    assert col.where == {"$or": [{"$and": [{"plan": "241"}, {"programme": "BCS"}]}, {"plan": "all"}]}
    assert {c.source for c in chunks} == {"plan.pdf", "faq.pdf"}      # BIT chunk filtered out of both searches


def test_keyword_search_finds_course_chunk_that_embeddings_miss():
    r = ChromaRetriever(FakeCollection(), FakeEmbedder())
    query = dict(query="Final Year Project I BICS 4321", k=2, plan="241", programme="BCS")
    assert [c.source for c in r.vector_search(**query)] == ["faq.pdf"]
    hybrid = r.search(**query)
    assert Chunk(FakeCollection.DOCS["fyp"][0], "plan.pdf", 1, "241", "BCS") in hybrid
    assert all(c.programme in ("BCS", "all") for c in hybrid)


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


# ------------------------------------------------------------------- sync --

class SyncCollection:
    def __init__(self, ids):
        self.ids = set(ids)
        self.deleted, self.upserted = [], []

    def get(self, include):
        return {"ids": sorted(self.ids)}

    def delete(self, ids):
        self.deleted += ids
        self.ids -= set(ids)

    def upsert(self, ids, documents, embeddings, metadatas):
        self.upserted += ids
        self.ids |= set(ids)


def test_sync_removes_chunks_that_no_longer_exist():
    new = [Chunk("FYP I needs 90 credits", "plan.pdf", 1, "241", "BCS")]
    col = SyncCollection(["old-version-of-an-edited-chunk"])
    removed = ingest.sync_collection(col, new, [[0.0]])
    assert removed == 1 and col.deleted == ["old-version-of-an-edited-chunk"]
    assert col.ids == {ingest.chunk_id(new[0])}


def test_sync_upserts_in_batches():
    chunks = [Chunk(f"chunk {i}", "a.pdf", 1) for i in range(7)]
    col = SyncCollection([])
    ingest.sync_collection(col, chunks, [[0.0]] * 7, batch=3)
    assert len(col.upserted) == 7 and len(col.ids) == 7


def test_same_file_in_two_folders_gets_two_ids():
    a = Chunk("same text", "faq.pdf", 1, "241", "BCS")
    b = Chunk("same text", "faq.pdf", 1, "241", "BIT")
    assert ingest.chunk_id(a) != ingest.chunk_id(b)
    assert ingest.unique_chunks([a, a, b]) == [a, b]


# ------------------------------------------------------------- CSV checks --

def test_csv_saved_by_excel_with_bom_still_loads(docs):
    raw, courses = docs
    path = courses / "bcs-241.csv"
    path.write_bytes(b"\xef\xbb\xbf" + path.read_bytes())      # UTF-8 byte-order mark
    plan = [c for c in ingest.build_chunks(raw, courses) if c.source == "plan.pdf"]
    assert any("BICS 4321" in c.text for c in plan)


@pytest.mark.parametrize("field, value, message", [
    ("programme", "BIT", "doesn't match its folder"),
    ("page", "", "page must be a page number"),
])
def test_bad_csv_rows_are_reported(docs, field, value, message):
    raw, courses = docs
    path = courses / "bcs-241.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0][field] = value
    write_csv(path, rows)
    with pytest.raises(SystemExit, match=message):
        ingest.build_chunks(raw, courses)


def test_csv_missing_a_column_is_reported(docs):
    raw, courses = docs
    path = courses / "bcs-241.csv"
    with path.open(encoding="utf-8") as f:
        rows = [{k: v for k, v in r.items() if k != "page"} for r in csv.DictReader(f)]
    write_csv(path, rows)
    with pytest.raises(SystemExit, match="missing columns"):
        ingest.build_chunks(raw, courses)


def write_semesters(courses):
    write_csv(courses / "bcs-241-semesters.csv", [
        {"plan": "241", "programme": "BCS", "year": "1", "semester": "1", "course_code": "BICS 1301",
         "course_name": "Elements of Programming", "credit_hours": "3", "prerequisite": "", "page": "3"},
        {"plan": "241", "programme": "BCS", "year": "", "semester": "", "course_code": "",
         "course_name": "Total Graduating Credits (Local Students)", "credit_hours": "125",
         "prerequisite": "", "page": "3"}])


def test_semester_page_comes_from_its_csv(docs):
    raw, courses = docs
    write_semesters(courses)
    plan = [c for c in ingest.build_chunks(raw, courses) if c.source == "plan.pdf"]
    texts = [(c.page, c.text) for c in plan]
    assert (3, "BCS study plan 241, Year 1 Semester 1 courses:\n"
               "- BICS 1301 Elements of Programming: 3 credit hours") in texts
    assert (3, "BCS study plan 241, total graduating credits:\n"
               "- Total Graduating Credits (Local Students): 125 credit hours") in texts
    assert (3, "YEAR 1 SEMESTER 1 BICS 1301") not in texts      # page 3's PDF text replaced
    assert any(page == 1 for page, _ in texts)                   # the course CSV still used


@pytest.mark.parametrize("year, semester", [("1", ""), ("", "2"), ("one", "1")])
def test_semester_csv_needs_both_year_and_semester_or_neither(docs, year, semester):
    raw, courses = docs
    write_semesters(courses)
    path = courses / "bcs-241-semesters.csv"
    with path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    rows[0].update(year=year, semester=semester)
    write_csv(path, rows)
    with pytest.raises(SystemExit, match="year and semester"):
        ingest.build_chunks(raw, courses)


def test_semester_csv_missing_a_column_names_its_script(docs):
    raw, courses = docs
    write_csv(courses / "bcs-241-semesters.csv", [{"plan": "241", "programme": "BCS", "page": "3"}])
    with pytest.raises(SystemExit, match="extract_semesters.py"):
        ingest.build_chunks(raw, courses)


def test_renamed_study_plan_pdf_is_warned_about(docs, capsys):
    raw, courses = docs
    (raw / "bcs-241" / "plan.pdf").rename(raw / "bcs-241" / "plan-v2.pdf")
    ingest.build_chunks(raw, courses)
    assert "plan.pdf not found" in capsys.readouterr().out


COURSES = Path(__file__).resolve().parent.parent / "data" / "courses"


@pytest.mark.parametrize("folder", sorted(ingest.STUDY_PLANS))
def test_study_plan_csvs_in_the_repo_load(folder):
    """The CSVs are checked and edited by hand: a broken row should fail here,
    not at ingest."""
    courses = ingest.load_course_csv(folder, COURSES)
    semesters = ingest.load_semester_csv(folder, COURSES)
    assert courses and semesters
    assert {r["year"] for r in semesters if r["year"]} == {"1", "2", "3", "4"}


# ------------------------------------------------------- keyword index age --

def test_keyword_index_is_rebuilt_after_reingest():
    col = FakeCollection()
    r = ChromaRetriever(col, FakeEmbedder())
    r.search("BICS 4321", k=3, plan="241", programme="BCS")           # builds the keyword index
    col.DOCS = {**FakeCollection.DOCS, "new": (
        "BCS study plan 241, Industrial Attachment:\n- BICS 4901 Industrial Attachment: 9 credit hours",
        {"source": "plan.pdf", "page": 1, "plan": "241", "programme": "BCS"})}
    hits = r.search("BICS 4901 Industrial Attachment", k=3, plan="241", programme="BCS")
    assert any("BICS 4901" in c.text for c in hits)
