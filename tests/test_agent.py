"""Agent loop tests with a fake LLM and retriever — no Ollama needed."""
import pytest

from app.agent import ANSWER_SYSTEM, ASK_STUDENT, GRADE_SYSTEM, PLAN_SYSTEM, CourseAgent
from app.text import Chunk


class FakeLLM:
    def __init__(self, grades):
        self.grades = list(grades)
        self.calls = []

    def chat(self, system, user, temperature=0.0):
        self.calls.append(system)
        if system == PLAN_SYSTEM:
            return '{"query": "prerequisite INFO 4303"}'
        if system == GRADE_SYSTEM:
            return self.grades.pop(0)
        if system == ANSWER_SYSTEM:
            return "You need the prerequisite [1]."
        raise AssertionError("unexpected prompt")


class FakeRetriever:
    def __init__(self):
        self.queries = []
        self.filters = []          # (plan, programme) of each search

    def search(self, query, k=5, plan=None, programme=None):
        self.queries.append(query)
        self.filters.append((plan, programme))
        return [Chunk(f"text for {query}", "handbook.pdf", len(self.queries))]


def test_answers_after_one_search_when_sufficient():
    llm, ret = FakeLLM(['{"sufficient": true, "reason": "ok"}']), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("Can I skip the prereq?", batch=241, programme="BCS")
    assert ret.queries == ["prerequisite INFO 4303"]
    assert result.answer.endswith("[1].")
    assert [s.name for s in result.steps] == ["student", "plan", "retrieve", "grade", "answer"]


def test_retries_with_new_query_then_stops():
    grades = [
        '{"sufficient": false, "reason": "missing", "next_query": "course prerequisites policy"}',
        '{"sufficient": true, "reason": "found it"}',
    ]
    llm, ret = FakeLLM(grades), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("Can I skip the prereq?", batch=241, programme="BCS")
    assert ret.queries == ["prerequisite INFO 4303", "course prerequisites policy"]
    assert len(result.sources) == 2  # context accumulates across searches


def test_retry_loop_is_bounded():
    grades = [f'{{"sufficient": false, "next_query": "q{i}"}}' for i in range(10)]
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=2).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 3  # first search + 2 retries


def test_stops_if_model_repeats_query():
    grades = ['{"sufficient": false, "next_query": "prerequisite INFO 4303"}']
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 1


def test_unparseable_grade_does_not_loop():
    llm, ret = FakeLLM(["I think so?"]), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 1


# ------------------------------------------------------------ student info --

@pytest.mark.parametrize("batch, programme", [
    (None, None), (241, None), (None, "BCS"), (241, ""),     # missing
    (191, "BCS"), (241, "LAW"), ("abc", "BIT"),              # rejected by plan_for
])
def test_asks_for_student_info_without_searching(batch, programme):
    llm, ret = FakeLLM([]), FakeRetriever()
    result = CourseAgent(llm, ret).run("When is FYP?", batch=batch, programme=programme)
    assert result.needs_student_info
    assert result.answer == ASK_STUDENT
    assert result.sources == []
    assert llm.calls == [] and ret.queries == []             # no LLM call, no search
    assert [s.name for s in result.steps] == ["student"]


def test_trace_shows_plan_and_normalised_programme():
    llm, ret = FakeLLM(['{"sufficient": true, "reason": "ok"}']), FakeRetriever()
    result = CourseAgent(llm, ret).run("When is FYP?", batch="251", programme=" bcs ")
    assert not result.needs_student_info
    assert result.steps[0].detail == "BCS 251 -> plan 241"


def test_reason_for_rejection_is_in_the_trace():
    result = CourseAgent(FakeLLM([]), FakeRetriever()).run("q", batch=191, programme="BCS")
    assert "191" in result.steps[0].detail


def test_every_search_is_filtered_by_the_students_plan():
    grades = ['{"sufficient": false, "reason": "missing", "next_query": "FYP credit requirement"}',
              '{"sufficient": true, "reason": "found it"}']
    llm, ret = FakeLLM(grades), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("When can I take FYP?", batch=251, programme="bit")
    assert len(ret.queries) == 2                              # first search and the retry
    assert ret.filters == [("241", "BIT"), ("241", "BIT")]
    assert all("(plan 241, BIT)" in s.detail for s in result.steps if s.name == "retrieve")
