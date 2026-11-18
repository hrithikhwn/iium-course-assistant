"""Agent loop tests with a fake LLM and retriever — no Ollama needed."""
import pytest

from app.agent import (ANSWER_SYSTEM, ASK_STUDENT, GRADE_SYSTEM, NOT_FOUND, NOT_FOUND_MESSAGE,
                       PLAN_SYSTEM, CourseAgent)
from app.text import Chunk


class FakeLLM:
    def __init__(self, grades, answer="You need the prerequisite [1]."):
        self.grades = list(grades)
        self.reply = answer
        self.calls = []

    def chat(self, system, user, temperature=0.0):
        self.calls.append(system)
        if system == PLAN_SYSTEM:
            return '{"query": "prerequisite rules"}'
        if system == GRADE_SYSTEM:
            return self.grades.pop(0)
        if system == ANSWER_SYSTEM:
            return self.reply
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
    assert ret.queries == ["Can I skip the prereq?", "prerequisite rules"]   # question + rewrite
    assert result.answer.endswith("[1].")
    assert [s.name for s in result.steps] == ["student", "plan", "retrieve", "grade", "answer"]


def test_retries_with_new_query_then_stops():
    grades = [
        '{"sufficient": false, "reason": "missing", "next_query": "course prerequisites policy"}',
        '{"sufficient": true, "reason": "found it"}',
    ]
    llm, ret = FakeLLM(grades), FakeRetriever()
    result = CourseAgent(llm, ret, max_retries=2).run("Can I skip the prereq?", batch=241, programme="BCS")
    assert ret.queries == ["Can I skip the prereq?", "prerequisite rules", "course prerequisites policy"]
    assert len(result.context) == 3  # context accumulates across searches


def test_retry_loop_is_bounded():
    grades = [f'{{"sufficient": false, "next_query": "q{i}"}}' for i in range(10)]
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=2).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 4  # question + rewrite, then 2 retries


def test_stops_if_model_repeats_query():
    grades = ['{"sufficient": false, "next_query": "Prerequisite Rules"}']
    llm, ret = FakeLLM(grades), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 2  # the first search only


def test_unparseable_grade_does_not_loop():
    llm, ret = FakeLLM(["I think so?"]), FakeRetriever()
    CourseAgent(llm, ret, max_retries=3).run("anything", batch=241, programme="BCS")
    assert len(ret.queries) == 2  # the first search (question + rewrite) only


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
    assert len(ret.queries) == 3                              # question + rewrite, then the retry
    assert ret.filters == [("241", "BIT")] * 3
    assert all("(plan 241, BIT)" in s.detail for s in result.steps if s.name == "retrieve")



# --------------------------------------------------------------- not found --

class EmptyRetriever(FakeRetriever):
    def search(self, query, k=5, plan=None, programme=None):
        super().search(query, k, plan, programme)
        return []


def run_bcs(llm, ret, **kw):
    return CourseAgent(llm, ret, **kw).run("How much are the fees?", batch=241, programme="BCS")


def test_found_answer_keeps_sources():
    result = run_bcs(FakeLLM(['{"sufficient": true, "reason": "ok"}']), FakeRetriever())
    assert result.found and result.sources and result.answer.endswith("[1].")


@pytest.mark.parametrize("reply", [NOT_FOUND, "not_found", f"  {NOT_FOUND}\n", f"[2] {NOT_FOUND}",
                                   f"Unfortunately, {NOT_FOUND}"])
def test_model_not_found_becomes_fixed_message_without_sources(reply):
    result = run_bcs(FakeLLM(['{"sufficient": true, "reason": "ok"}'], answer=reply), FakeRetriever())
    assert not result.found
    assert result.answer == NOT_FOUND_MESSAGE
    assert result.sources == []
    assert len(result.context) == 2      # kept: tells a wrong refusal apart from a search miss
    assert result.steps[-1].detail == f"model replied {NOT_FOUND} from 2 excerpts"  # question + rewrite


def test_answer_step_still_runs_when_grade_never_passes():
    """The grader can be wrong (it rejected a correct course table in testing),
    so a failed grade alone doesn't stop the answer."""
    grades = ['{"sufficient": false, "next_query": "a"}', '{"sufficient": false, "next_query": "b"}']
    llm = FakeLLM(grades)
    result = run_bcs(llm, FakeRetriever(), max_retries=1)
    assert llm.calls[-1] == ANSWER_SYSTEM
    assert result.found


def test_nothing_retrieved_is_not_found_without_answer_call():
    llm = FakeLLM(['{"sufficient": false, "next_query": "x"}'])
    result = run_bcs(llm, EmptyRetriever(), max_retries=0)
    assert not result.found and result.answer == NOT_FOUND_MESSAGE
    assert ANSWER_SYSTEM not in llm.calls


def test_unparseable_grade_is_noted_in_trace():
    result = run_bcs(FakeLLM(["I think so?"]), FakeRetriever())
    assert "unparseable" in [s for s in result.steps if s.name == "grade"][0].detail


def test_invented_codes_are_dropped_from_plan_and_retry_queries():
    class InventingLLM(FakeLLM):
        def chat(self, system, user, temperature=0.0):
            if system == PLAN_SYSTEM:
                self.calls.append(system)
                return '{"query": "BICS 3062 Data Structures prerequisite"}'
            return super().chat(system, user, temperature)

    grades = ['{"sufficient": false, "next_query": "CSCI 9999 Data Structures"}',
              '{"sufficient": true, "reason": "ok"}']
    ret = FakeRetriever()
    CourseAgent(InventingLLM(grades), ret, max_retries=2).run(
        "What's the prereq for Data Structures?", batch=241, programme="BCS")
    assert ret.queries == ["What's the prereq for Data Structures?",
                           "Data Structures prerequisite", "Data Structures"]


# ---------------------------------------------------------- context budget --

class BigRetriever(FakeRetriever):
    """Every search returns three 1000-character chunks."""
    def search(self, query, k=5, plan=None, programme=None):
        super().search(query, k, plan, programme)
        n = len(self.queries)
        return [Chunk(f"{query} {n}-{i} " + "x" * 1000, "plan.pdf", n * 10 + i) for i in range(3)]


def test_context_sent_to_the_model_stays_within_budget():
    grades = ['{"sufficient": false, "next_query": "a"}', '{"sufficient": false, "next_query": "b"}',
              '{"sufficient": true, "reason": "ok"}']
    result = CourseAgent(FakeLLM(grades), BigRetriever(), max_retries=2, max_context_chars=3500).run(
        "When is FYP?", batch=241, programme="BCS")
    assert sum(len(c.text) + len(c.citation()) + 10 for c in result.context) <= 3500
    assert len(result.context) == 3
    assert all(c.text.startswith("b ") for c in result.context)   # the newest search comes first


def test_question_comes_after_the_excerpts():
    from app.agent import excerpts_then_question
    msg = excerpts_then_question("When is FYP?", [Chunk("FYP rules", "plan.pdf", 1)])
    assert msg.startswith("Excerpts:\n[1] (plan.pdf, p.1)\nFYP rules")
    assert msg.endswith("\n\nQuestion: When is FYP?")


def test_rewrite_adds_to_the_questions_results_without_removing_any():
    class TwoListRetriever(FakeRetriever):
        def search(self, query, k=5, plan=None, programme=None):
            super().search(query, k, plan, programme)
            tag = "q" if query.startswith("When") else "r"
            return [Chunk(f"{tag}{i}", "plan.pdf", i) for i in range(k)]

    agent = CourseAgent(FakeLLM([]), TwoListRetriever(), top_k=3)
    found = agent.retrieve(["When is FYP?", "FYP date"], "241", "BCS")
    assert [c.text for c in found] == ["q0", "q1", "q2", "r0", "r1", "r2"]


def test_grader_suggesting_the_question_itself_is_not_searched_again():
    grades = ['{"sufficient": false, "next_query": "when is fyp?"}']
    ret = FakeRetriever()
    CourseAgent(FakeLLM(grades), ret, max_retries=2).run("When is FYP?", batch=241, programme="BCS")
    assert ret.queries == ["When is FYP?", "prerequisite rules"]


def test_rewrite_off_searches_the_question_only():
    llm, ret = FakeLLM(['{"sufficient": true, "reason": "ok"}']), FakeRetriever()
    result = CourseAgent(llm, ret, rewrite=False).run("When is FYP?", batch=241, programme="BCS")
    assert llm.calls == [GRADE_SYSTEM, ANSWER_SYSTEM]          # no plan call
    assert ret.queries == ["When is FYP?"]
    assert [s.name for s in result.steps] == ["student", "plan", "retrieve", "grade", "answer"]
    assert "off" in result.steps[1].detail                    # the trace still says what happened


# --------------------------------------------------------------- citations --

def answer_with(reply):
    llm = FakeLLM(['{"sufficient": true, "reason": "ok"}'], answer=reply)
    return CourseAgent(llm, FakeRetriever()).run("Can I skip the prereq?", batch=241, programme="BCS")


def test_sources_are_only_the_excerpts_the_answer_cites():
    result = answer_with("You need BICS 1301 first [2].")
    assert len(result.context) == 2                          # question + rewrite searches
    assert result.sources == [result.context[1]]
    assert result.answer == "You need BICS 1301 first [1]."  # renumbered to match the sources list


def test_unknown_citations_are_removed_and_noted_in_the_trace():
    result = answer_with("You need BICS 1301 first [1]. Fees apply [9].")
    assert result.answer == "You need BICS 1301 first [1]. Fees apply."
    assert "[9]" in result.steps[-1].detail


def test_answer_without_citations_is_kept_but_has_no_sources():
    result = answer_with("You need BICS 1301 first.")
    assert result.found and result.answer == "You need BICS 1301 first."
    assert result.sources == [] and len(result.context) == 2
    assert "no citations" in result.steps[-1].detail


def test_not_found_after_a_real_answer_is_cut_off():
    """Seen live: the model answered from one excerpt, then judged another."""
    result = answer_with(f"[1] You need BICS 1301 first. [2] {NOT_FOUND}")
    assert result.found
    assert result.answer == "[1] You need BICS 1301 first."
    assert result.sources == [result.context[0]]        # [2] was only cited by the marker
    assert f"removed {NOT_FOUND}" in result.steps[-1].detail


def test_section_numbers_used_as_citations_are_removed():
    """Seen live: "[35.1]" for SAPER rule 35.1 instead of an excerpt number."""
    result = answer_with("Probation starts below a CGPA of 2.00 [35.1].")
    assert result.answer == "Probation starts below a CGPA of 2.00."
    assert result.sources == []
    assert "[35.1]" in result.steps[-1].detail


def test_empty_reply_is_not_found():
    result = answer_with("")
    assert not result.found and result.answer == NOT_FOUND_MESSAGE
    assert result.steps[-1].detail == "empty reply from 2 excerpts"


# ----------------------------------------------------- plain RAG and strict --

def test_grading_off_is_plain_rag():
    llm, ret = FakeLLM([]), FakeRetriever()
    result = CourseAgent(llm, ret, rewrite=False, grading=False, max_retries=2).run(
        "When is FYP?", batch=241, programme="BCS")
    assert llm.calls == [ANSWER_SYSTEM]                       # one LLM call: the answer
    assert ret.queries == ["When is FYP?"]                    # one search, no retries
    assert [s.name for s in result.steps] == ["student", "plan", "retrieve", "grade", "answer"]
    assert "off" in result.steps[3].detail
    assert result.found and result.answer.endswith("[1].")


def test_strict_says_not_found_when_the_last_grade_fails():
    grades = ['{"sufficient": false, "next_query": "a"}', '{"sufficient": false, "next_query": "b"}']
    llm = FakeLLM(grades)
    result = CourseAgent(llm, FakeRetriever(), max_retries=1, refuse_on_failed_grade=True).run(
        "When is FYP?", batch=241, programme="BCS")
    assert not result.found and result.answer == NOT_FOUND_MESSAGE
    assert ANSWER_SYSTEM not in llm.calls                     # the answer model isn't asked
    assert len(result.context) == 3                           # question + rewrite, then the retry
    assert "strict" in result.steps[-1].detail


def test_strict_answers_when_the_grade_passes():
    llm = FakeLLM(['{"sufficient": true, "reason": "ok"}'])
    result = CourseAgent(llm, FakeRetriever(), refuse_on_failed_grade=True).run(
        "When is FYP?", batch=241, programme="BCS")
    assert result.found and llm.calls[-1] == ANSWER_SYSTEM


def test_strict_mode_needs_grading():
    with pytest.raises(ValueError):
        CourseAgent(FakeLLM([]), FakeRetriever(), grading=False, refuse_on_failed_grade=True)
