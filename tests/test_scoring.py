import pytest

from app.scoring import (missing_facts, missing_from, normalise, parse_facts, question_facts,
                         rank_covering, score_answer, summarise)
from app.text import Chunk


def test_parse_facts_splits_facts_and_alternatives():
    assert parse_facts("1.67; 18 credit|eighteen credit; ") == [["1.67"], ["18 credit", "eighteen credit"]]
    assert parse_facts("") == []


@pytest.mark.parametrize("text, norm", [
    ("A load of eighteen (18) credits.", "a load of eighteen 18 credits"),
    ("needs BIIT2305 first", "needs biit 2305 first"),
    ("1.67 ≤ CGPA < 2.00.", "1.67 cgpa 2.00"),
    ("Lectures start 22/02/2027 (Monday)", "lectures start 22/02/2027 monday"),
])
def test_normalise(text, norm):
    assert normalise(text) == norm


@pytest.mark.parametrize("text, facts, missing", [
    ("you may take eighteen (18) credits", [["18 credit"]], []),        # plural, brackets
    ("twenty credit hours", [["20 credit", "twenty credit"]], []),      # an alternative
    ("needs BIIT2305", [["BIIT 2305"]], []),
    ("from 10 to 18 April", [["10"], ["18"], ["apr"]], []),
    ("rules for batch 180", [["18"]], ["18"]),                          # 18 is not 180
    ("a CGPA of 21.67", [["1.67"]], ["1.67"]),
    ("CGPA below 1.67", [["1.67"], ["2.00"]], ["2.00"]),
    ("FYP I is 3 credit hours and FYP II 4 credit hours", [["7 credit", "3 credit & 4 credit"]], []),
    ("FYP I is 3 credit hours", [["7 credit", "3 credit & 4 credit"]], ["7 credit"]),   # both parts needed
    ("Exams run 14 January to 5 February 2027", [["14/01/2027", "14 jan"], ["05/02/2027", "5 feb"]], []),
])
def test_missing_facts(text, facts, missing):
    assert missing_facts(text, facts) == missing


def test_facts_can_come_from_different_chunks():
    chunks = [Chunk("Probation: CGPA from 1.67 to below 2.00", "saper.pdf", 17),
              Chunk("A student on probation takes at most eighteen (18) credits", "saper.pdf", 17)]
    assert missing_from(chunks, [["1.67"], ["2.00"], ["18 credit"]]) == []
    assert missing_from(chunks[:1], [["1.67"], ["18 credit"]]) == ["18 credit"]


def test_rank_covering_counts_only_the_expected_file():
    chunks = [Chunk("1.67 and 18 credits", "handbook.pdf", 3),
              Chunk("CGPA 1.67", "saper.pdf", 17),
              Chunk("at most 18 credits", "saper.pdf", 17)]
    facts = [["1.67"], ["18 credit"]]
    assert rank_covering(chunks, facts) == 1
    assert rank_covering(chunks, facts, source="saper.pdf") == 3
    assert rank_covering(chunks, [["90"]]) is None


def test_question_facts_fall_back_to_codes_in_the_first_sentence():
    q = {"key_facts": "", "expected_answer": "BIIT 4321 requires BIIT 1301. The plan also lists BICS 4902."}
    assert question_facts(q) == [["BIIT 4321"], ["BIIT 1301"]]
    assert question_facts({"key_facts": "7 credit; 90"}) == [["7 credit"], ["90"]]


# ------------------------------------------------------------ score_answer --

FYP = {"type": "courses", "key_facts": "BIIT 1301; BIIT 2303", "source_file": "plan.pdf", "page": "1"}
LINE = Chunk("- BIIT 4321 Final Year Project I: prerequisite: BIIT 1301, BIIT 2303", "plan.pdf", 1)
OTHER = Chunk("Final Year Project rules", "handbook.pdf", 21)


def test_correct_supported_answer():
    s = score_answer(FYP, "You need BIIT 1301 and BIIT 2303 [1].", True, [LINE], [OTHER, LINE])
    assert s == {"correct": True, "missing": "", "supported": True, "retrieved": True, "cites_source": True}


def test_answer_missing_a_fact_and_citing_the_wrong_excerpt():
    s = score_answer(FYP, "You need BIIT 1301 [1].", True, [OTHER], [OTHER, LINE])
    assert (s["correct"], s["missing"], s["supported"], s["retrieved"], s["cites_source"]) == \
        (False, "BIIT 2303", False, True, False)


def test_not_found_on_an_answerable_question_is_wrong():
    s = score_answer(FYP, "I couldn't find that.", False, [], [LINE])
    assert (s["correct"], s["missing"], s["retrieved"]) == (False, "BIIT 1301; BIIT 2303", True)


@pytest.mark.parametrize("found, correct", [(False, True), (True, False)])
def test_refuse_questions_are_right_when_not_found(found, correct):
    assert score_answer({"type": "refuse"}, "x", found, [], [])["correct"] is correct


def test_citation_numbers_are_not_facts():
    """[4] and [5] cite excerpts; they aren't the weeks 4 and 5 the facts ask for."""
    q = {"type": "dates", "key_facts": "4|fourth; 5|fifth; 13|thirteenth", "source_file": ""}
    s = score_answer(q, "Classes end in week 13 [4][5].", True, [], [])
    assert (s["correct"], s["missing"]) == (False, "4; 5")


def test_questions_without_facts_are_not_scored():
    q = {"type": "rules", "key_facts": "", "expected_answer": "Twelve semesters.", "source_file": ""}
    assert score_answer(q, "12 semesters", True, [], [])["correct"] == ""


def test_summarise():
    rows = [
        {"type": "courses", "found": True, "correct": True, "supported": True, "retrieved": True,
         "seconds": 20, "llm_calls": 3, "error": ""},
        {"type": "rules", "found": False, "correct": False, "supported": False, "retrieved": True,
         "seconds": 40, "llm_calls": 7, "error": ""},
        {"type": "refuse", "found": False, "correct": True, "supported": "", "retrieved": "",
         "seconds": 30, "llm_calls": 5, "error": ""},
        {"type": "rules", "error": "timed out"},
    ]
    assert summarise(rows) == {
        "answers": 4, "errors": 1, "answerable": 2, "correct": 1, "false_refusals": 1,
        "supported": 1, "retrieved": 2, "should_refuse": 1, "refused": 1,
        "median_seconds": 30.0, "mean_llm_calls": 5.0,
    }
