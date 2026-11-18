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


# ---------------------------------------------------------- keyword search --

from app.text import BM25, codes_in, matches_filter, rrf, singular, tokenize  # noqa: E402


def test_codes_in_normalises_spacing():
    assert codes_in("needs BIIT2305 and bics 1304, not 12345") == ["BIIT 2305", "BICS 1304"]


def test_tokenize_drops_stopwords_strips_plurals_and_joins_codes():
    assert tokenize("What are the prerequisites for BIIT 4321?") == ["prerequisite", "biit", "4321", "biit4321"]


@pytest.mark.parametrize("word, base", [
    ("prerequisites", "prerequisite"), ("courses", "course"), ("studies", "study"),
    ("class", "class"), ("status", "status"), ("analysis", "analysis"), ("gps", "gps"),
])
def test_singular(word, base):
    assert singular(word) == base


def test_bm25_prefers_rare_exact_terms():
    docs = ["Final Year Project rules and supervision in general",
            "BIIT 4321 Final Year Project I: prerequisite BIIT 1301",
            "Industrial attachment rules"]
    bm = BM25(docs)
    assert bm.top("Final Year Project I prerequisites", 3)[0] == 1
    assert bm.top("BIIT 4321", 3) == [1]           # only documents that match at all
    assert bm.top("tuition fees", 3) == []


def test_rrf_rewards_items_high_in_either_list():
    assert rrf([["a", "b", "c"], ["c", "d"]]) == ["c", "a", "b", "d"]


def test_matches_filter_mirrors_plan_filter():
    assert matches_filter({"plan": "all", "programme": "all"}, "241", "BCS")
    assert matches_filter({"plan": "241", "programme": "BCS"}, "241", "BCS")
    assert not matches_filter({"plan": "241", "programme": "BIT"}, "241", "BCS")
    assert not matches_filter({"plan": "201", "programme": "BCS"}, "241", "BCS")
    assert matches_filter({"plan": "201", "programme": "BCS"}, None, None)


@pytest.mark.parametrize("query, question, out", [
    ("BICS 3062 prerequisite", "What's the prereq for Data Structures?", "prerequisite"),
    ("BICS 3062 Data Structures prerequisite", "prereq for Data Structures?", "Data Structures prerequisite"),
    ("final exam Semester II 2026/2027", "When is the final exam?", "final exam Semester II 2026/2027"),
    ("exam 2026 period", "exam period?", "exam 2026 period"),     # lower case: not treated as a code
    ("BICS 2302 prerequisite", "prereq for BICS2302?", "BICS 2302 prerequisite"),
    ("BICS 3062", "prereq for x", ""),
])
def test_drop_invented_codes(query, question, out):
    from app.text import drop_invented_codes
    assert drop_invented_codes(query, question) == out


def test_codes_in_ignores_semesters_months_and_years():
    text = "Semester II 2026/2027, from Dec 2025 to Year 2027; LM 2026 and BIIT4321"
    assert codes_in(text) == ["LM 2026", "BIIT 4321"]


# ------------------------------------------------------------ context fit --

from app.text import fit_context  # noqa: E402


def test_fit_context_keeps_order_and_stays_under_budget():
    chunks = [Chunk("a" * 400, "x.pdf", 1), Chunk("b" * 400, "x.pdf", 2),
              Chunk("c" * 400, "x.pdf", 3), Chunk("d" * 50, "x.pdf", 4)]
    kept = fit_context(chunks, max_chars=1000)
    assert [c.page for c in kept] == [1, 2, 4]            # 3 doesn't fit; the small 4 still does
    assert sum(len(c.text) + len(c.citation()) + 10 for c in kept) <= 1000


def test_fit_context_always_keeps_one_chunk():
    assert len(fit_context([Chunk("z" * 5000, "x.pdf", 1)], max_chars=100)) == 1


# ------------------------------------------------------------- fused search --

from app.text import fuse  # noqa: E402


def test_fuse_without_guarantees_is_plain_rank_fusion():
    vector, keyword = ["faq", "a", "b", "c"], ["fyp", "faq", "a", "b"]
    assert fuse([vector, keyword], k=3) == ["faq", "a", "b"]          # fyp, only in one list, drops out


def test_fuse_keeps_each_lists_best_hits():
    vector, keyword = ["faq", "a", "b", "c"], ["fyp", "faq", "a", "b"]
    out = fuse([vector, keyword], k=3, guaranteed=1)
    assert "fyp" in out and "faq" in out and len(out) == 3
    assert out == [i for i in ["faq", "a", "b", "fyp"] if i in out]   # still in fused order


# --------------------------------------------------------------- citations --

from app.text import renumber_citations  # noqa: E402


def excerpts(n):
    return [Chunk(f"excerpt {i}", "doc.pdf", i) for i in range(1, n + 1)]


def test_only_cited_excerpts_are_kept_and_renumbered_in_order_of_use():
    chunks = excerpts(5)
    cited = renumber_citations("FYP I needs 80 credits [4]. It is run by DCS [2, 4]. Again [4].", chunks)
    assert cited.answer == "FYP I needs 80 credits [1]. It is run by DCS [1, 2]. Again [1]."
    assert cited.sources == [chunks[3], chunks[1]]
    assert cited.numbers == [4, 2]
    assert cited.unknown == []


def test_numbers_that_are_not_excerpts_are_removed_and_reported():
    chunks = excerpts(3)
    cited = renumber_citations("Add/drop ends in week 2 [2]. Fees are due [7]. See also [0, 3].", chunks)
    assert cited.answer == "Add/drop ends in week 2 [1]. Fees are due. See also [2]."
    assert cited.sources == [chunks[1], chunks[2]]
    assert cited.unknown == ["7", "0"]


def test_section_numbers_are_not_excerpt_numbers():
    """Seen live: SAPER's rule numbers cited instead of excerpt numbers."""
    chunks = excerpts(3)
    cited = renumber_citations("Probation starts below a CGPA of 2.00 [35.1]. "
                               "The load is then at most 18 credits [2, 35.2].", chunks)
    assert cited.answer == "Probation starts below a CGPA of 2.00. The load is then at most 18 credits [1]."
    assert cited.sources == [chunks[1]]
    assert cited.unknown == ["35.1", "35.2"]


def test_repeated_and_adjacent_citations():
    chunks = excerpts(3)
    cited = renumber_citations("A [3][1]. B [3, 3].", chunks)
    assert cited.answer == "A [1][2]. B [1]."
    assert cited.sources == [chunks[2], chunks[0]]


@pytest.mark.parametrize("answer", ["No citation here.", "Week 2 [of Semester I] or [2026/27].", ""])
def test_text_without_citations_is_unchanged(answer):
    cited = renumber_citations(answer, excerpts(2))
    assert (cited.answer, cited.sources, cited.unknown) == (answer, [], [])


# ------------------------------------------------------- not-found marker --

from app.text import strip_marker  # noqa: E402


@pytest.mark.parametrize("reply", [
    "NOT_FOUND", "not_found", "  NOT_FOUND\n", "**NOT_FOUND**",
    "[2] NOT_FOUND",                                    # after a citation
    "NOT_FOUND. The excerpts only list 2025 fees.",     # starts with it: still no answer
    "Unfortunately, NOT_FOUND", "Unfortunately, NOT_FOUND.",   # seen live: a lead-in first
    "Sorry, but NOT_FOUND [2]",
    "FYP I needs BIIT 1301 [3]. Hostel fees: NOT_FOUND.",     # inside a sentence, even after an answer
    "", "[3].",                                         # nothing but citations left
])
def test_replies_that_mean_no_answer(reply):
    assert strip_marker(reply, "NOT_FOUND") == ""


@pytest.mark.parametrize("reply, answer", [
    # seen live: answered from one excerpt, then judged another one
    ("[3] FYP I needs BIIT 1301. [7] NOT_FOUND", "[3] FYP I needs BIIT 1301."),
    ("FYP I needs BIIT 1301 [3]. NOT_FOUND.", "FYP I needs BIIT 1301 [3]."),   # keeps the answer's [3]
    ("FYP I needs BIIT 1301 [3] NOT_FOUND", "FYP I needs BIIT 1301 [3]"),
    ("FYP I needs BIIT 1301 [3].\n[7] NOT_FOUND", "FYP I needs BIIT 1301 [3]."),
    ("No marker here [1].", "No marker here [1]."),
])
def test_marker_after_a_real_answer_is_cut_off(reply, answer):
    assert strip_marker(reply, "NOT_FOUND") == answer
