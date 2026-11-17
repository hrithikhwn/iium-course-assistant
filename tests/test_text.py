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


# ------------------------------------------------------------ batch tags --

from app.text import course_chunks, course_line, folder_tags, plan_filter, plan_for  # noqa: E402


@pytest.mark.parametrize("folder, tags", [
    ("all", ("all", "all")),
    ("bcs-201", ("201", "BCS")),
    ("bit-261", ("261", "BIT")),
])
def test_folder_tags(folder, tags):
    assert folder_tags(folder) == tags


@pytest.mark.parametrize("folder", ["misc", "bcs241", "law-241"])
def test_folder_tags_rejects_unknown_folders(folder):
    with pytest.raises(ValueError):
        folder_tags(folder)


@pytest.mark.parametrize("batch, programme, plan", [
    (201, "BCS", "201"), (231, "BIT", "201"),
    (241, "BCS", "241"), ("251", "bit", "241"),
    (261, "BIT", "261"), (271, "BIT", "261"),
    (261, "BCS", "241"),                 # no BCS 261 plan yet: follows 241
    ("2510123", "BCS", "241"),           # matric number
])
def test_plan_for(batch, programme, plan):
    assert plan_for(batch, programme) == plan


@pytest.mark.parametrize("batch, programme", [(191, "BCS"), (241, "LAW"), ("x", "BCS")])
def test_plan_for_rejects_bad_input(batch, programme):
    with pytest.raises(ValueError):
        plan_for(batch, programme)


def test_plan_filter_includes_all_documents():
    assert plan_filter("241", "BCS") == {"$or": [
        {"$and": [{"plan": "241"}, {"programme": "BCS"}]},
        {"plan": "all"},
    ]}


# --------------------------------------------------------- course tables --

def row(code, name, page="1", category="Core Computing Courses", prereq="", dept="DCS"):
    return {"plan": "241", "programme": "BCS", "category": category, "course_code": code,
            "course_name": name, "credit_hours": "3", "prerequisite": prereq,
            "offering_dept": dept, "page": page}


def test_course_line():
    assert course_line(row("BICS 2302", "Data Structures and Algorithms", prereq="BICS 1304, BICS 1305")) == \
        "- BICS 2302 Data Structures and Algorithms: 3 credit hours; prerequisite: BICS 1304, BICS 1305; offered by DCS"
    assert course_line(row("", "Free Elective I", prereq="Course Specific", dept="")) == \
        "- Free Elective I: 3 credit hours; prerequisite: Course Specific"


def test_course_chunks_group_by_category_and_page():
    rows = [row("BICS 1301", "Elements of Programming"),
            row("BICS 1302", "Introduction to Computer Organization"),
            row("BICS 4321", "Final Year Project I", category="Final Year Project"),
            row("BICS 4340", "Machine Learning", page="2", category="Final Year Project")]
    chunks = course_chunks(rows, "plan.pdf")
    assert [(c.page, c.text.splitlines()[0]) for c in chunks] == [
        (1, "BCS study plan 241, Core Computing Courses:"),
        (1, "BCS study plan 241, Final Year Project:"),
        (2, "BCS study plan 241, Final Year Project:"),
    ]
    assert all((c.source, c.plan, c.programme) == ("plan.pdf", "241", "BCS") for c in chunks)
    assert "BICS 1302 Introduction to Computer Organization" in chunks[0].text


def test_course_chunks_split_long_categories_and_repeat_header():
    rows = [row(f"BICS {1300 + i}", f"Course number {i}") for i in range(30)]
    chunks = course_chunks(rows, "plan.pdf", size=400)
    assert len(chunks) > 1
    assert all(len(c.text) <= 400 for c in chunks)
    assert all(c.text.startswith("BCS study plan 241, Core Computing Courses:") for c in chunks)
    assert sum(c.text.count("\n- ") for c in chunks) == 30
