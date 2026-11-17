"""Agent loop tests with a fake LLM and retriever — no Ollama needed."""
from app.agent import ANSWER_SYSTEM, GRADE_SYSTEM, PLAN_SYSTEM, CourseAgent
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

    def search(self, query, k=5):
        self.queries.append(query)
        return [Chunk(f"text for {query}", "handbook.pdf", len(self.queries))]


def test_answers_after_one_search_when_sufficient():
    llm, ret = FakeLLM(['{"sufficient": true, "reason": "ok"}']), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("Can I skip the prereq?")
    assert ret.queries == ["prerequisite INFO 4303"]
    assert result.answer.endswith("[1].")
    assert [s.name for s in result.steps] == ["plan", "retrieve", "grade", "answer"]


def test_retries_with_new_query_then_stops():
    grades = [
        '{"sufficient": false, "reason": "missing", "next_query": "course prerequisites policy"}',
        '{"sufficient": true, "reason": "found it"}',
    ]
    llm, ret = FakeLLM(grades), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("Can I skip the prereq?")
    assert ret.queries == ["prerequisite INFO 4303", "course prerequisites policy"]
    assert len(result.sources) == 2  # context accumulates across searches


def test_retry_loop_is_bounded():
    grades = [f'{{"sufficient": false, "next_query": "q{i}"}}' for i in range(10)]
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=2).run("anything")
    assert len(ret.queries) == 3  # first search + 2 retries


def test_stops_if_model_repeats_query():
    grades = ['{"sufficient": false, "next_query": "prerequisite INFO 4303"}']
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything")
    assert len(ret.queries) == 1


def test_unparseable_grade_does_not_loop():
    llm, ret = FakeLLM(["I think so?"]), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything")
    assert len(ret.queries) == 1
