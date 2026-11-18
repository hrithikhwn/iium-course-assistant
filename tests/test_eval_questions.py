"""eval/questions.csv is data the eval depends on: check every row is usable
before hours of model runs find out the hard way."""
import csv
from pathlib import Path

import pytest

from app.scoring import question_facts
from app.text import plan_for

PATH = Path(__file__).resolve().parent.parent / "eval" / "questions.csv"
with PATH.open(encoding="utf-8-sig", newline="") as f:
    QUESTIONS = list(csv.DictReader(f))
TYPES = {"rules", "registration", "courses", "dates", "refuse"}


def test_ids_are_unique():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("q", QUESTIONS, ids=[q["id"] for q in QUESTIONS])
def test_question_is_complete(q):
    assert q["question"].strip() and q["expected_answer"].strip()
    assert q["type"] in TYPES
    if q["type"] == "refuse":
        assert not q["key_facts"].strip() and not q["source_file"].strip(), "refusals have no answer to find"
    else:
        assert question_facts(q), "answerable questions need key facts"
        assert q["source_file"].strip() and q["page"].strip().isdigit()


@pytest.mark.parametrize("q", [q for q in QUESTIONS if q["plan"] != "all"],
                         ids=[q["id"] for q in QUESTIONS if q["plan"] != "all"])
def test_batch_belongs_to_the_plan(q):
    """The runner asks as this batch; the agent must land on the question's plan."""
    assert plan_for(q["batch"] or q["plan"], q["programme"]) == q["plan"]
