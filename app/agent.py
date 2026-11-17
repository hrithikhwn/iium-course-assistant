"""The agentic RAG loop: plan -> retrieve -> grade -> (retry) -> answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings
from .llm import LLM
from .retriever import Retriever
from .text import (Chunk, dedupe, drop_invented_codes, extract_json, fit_context, format_context,
                   plan_for, renumber_citations, strip_marker)

ASK_STUDENT = "Which programme are you in (BCS or BIT), and what's your batch (e.g. 241)?"

# How every search query the model writes should look (plan step and retries).
# Found with `check_retrieval --planned`: the old prompt's example code (INFO 4303)
# was copied into queries, codes were invented, and batch, programme and "IIUM"
# were added although the search filter already handles them. A long list of
# rules made it worse (the 7B model appended "official documents" to queries);
# short rules plus examples gave cleaner queries, but still scored below the raw
# question (11/13 vs 12/13), and invented codes are only stopped by
# drop_invented_codes. The examples avoid the eval questions' topics.
QUERY_RULES = ("Keep only the topic words. Leave out the student's programme, batch, \"IIUM\" and "
               "question words. Copy course names and codes exactly and never add a code that is "
               "not in the question. Write FYP as Final Year Project and IA as Industrial Attachment.")

PLAN_SYSTEM = f"""You turn a student's question about IIUM courses, study plans and academic rules \
into ONE short keyword search query. {QUERY_RULES}

Examples:
Question: I'm BIT batch 231, how many credits is Web Technologies?
{{"query": "Web Technologies credit hours"}}
Question: can i defer my studies for one semester? im bcs 241
{{"query": "deferment of studies semester"}}
Question: What's the prereq for Operating Systems?
{{"query": "Operating Systems prerequisite"}}
Question: When does course add/drop end?
{{"query": "add drop courses period"}}

Reply with JSON only:
{{"query": "<search query>"}}"""

GRADE_SYSTEM = f"""You check whether document excerpts contain enough information to answer a question.
Some excerpts are course tables with lines like
"- BIIT 4321 Final Year Project I: 3 credit hours; prerequisite: BIIT 1301, BIIT 2303; offered by DIS".
Such a line fully answers questions about that course's credit hours, prerequisites or department.
Judge only whether the needed facts are present, not how clearly they are worded.
If not sufficient, suggest a different short keyword search query. {QUERY_RULES}
Reply with JSON only:
{{"sufficient": true|false, "reason": "<one short sentence>", "next_query": "<the different search query if not sufficient, else empty>"}}"""

NOT_FOUND = "NOT_FOUND"   # what the answer model replies when the excerpts don't answer the question

# In live runs the 7B model once cited a document's own section numbers ("[35.1]"
# for SAPER rule 35.1) instead of excerpt numbers, and once ended a correct answer
# with "[7] NOT_FOUND", judging the excerpts one by one. The citation and NOT_FOUND
# lines below target both. Old vs new wording on the same excerpts (5 eval
# questions x 3 replies): refused 6/6 both, cited a real excerpt 9/9 both, put the
# citation before the fact ("[3] You need...") 6/9 -> 0/9. Neither failure came
# back with either wording, so text.renumber_citations and text.strip_marker are
# what keep them from reaching the student.
ANSWER_SYSTEM = f"""You are an assistant for IIUM students. Answer the question ONLY from the numbered excerpts.
- Write the answer in your own words in 1-3 sentences. Do not copy whole excerpts.
- After each fact, put the number of its excerpt in brackets, like [2]: the number shown before the excerpt, never a section or page number from its text.
- Course tables list each course as "- CODE Name: credit hours; prerequisite: ...". If the question is about a course, use its line.
- If none of the excerpts answer the question, reply with exactly {NOT_FOUND} and nothing else. Do not guess."""


def excerpts_then_question(question: str, chunks: list[Chunk]) -> str:
    """The user message for grade and answer. The question goes last: with it
    first, the 7B model answered from excerpt [1] even when the right course
    line was excerpt [5] (0/3 correct); with it last, 3/3."""
    return f"Excerpts:\n{format_context(chunks)}\n\nQuestion: {question}"

NOT_FOUND_MESSAGE = ("I couldn't find that in the official KICT and IIUM documents I have. "
                     "Please check with your academic advisor or the KICT academic office.")


@dataclass
class Step:
    name: str
    detail: str


@dataclass
class Result:
    answer: str
    # the excerpts the answer cites, in the order of its [1], [2], ... citations
    sources: list[Chunk]
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    # True when the agent stopped to ask for the student's batch/programme
    # instead of answering; the CLI/UI should ask and call run() again
    needs_student_info: bool = False
    # False when the documents don't answer the question; `answer` is then
    # NOT_FOUND_MESSAGE and there are no sources to cite
    found: bool = True
    # every excerpt the model was shown, cited or not (also when not found), so
    # the eval can tell "retrieved but not cited" from "never retrieved"
    context: list[Chunk] = field(default_factory=list)


class CourseAgent:
    def __init__(self, llm: LLM, retriever: Retriever, top_k: int = settings.top_k,
                 max_retries: int = settings.max_retries, max_context_chars: int = settings.max_context_chars,
                 rewrite: bool = settings.query_rewrite):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_retries = max_retries
        self.max_context_chars = max_context_chars   # excerpts sent to grade/answer, see text.fit_context
        self.rewrite = rewrite                       # False: no plan-step LLM call, search the question only

    def plan(self, question: str) -> str:
        reply = extract_json(self.llm.chat(PLAN_SYSTEM, question))
        # the model sometimes invents course codes ("BICS 3062" for Data Structures);
        # drop any code the student didn't write, and fall back to the question if nothing is left
        return drop_invented_codes(str(reply.get("query") or ""), question) or question.strip()

    def grade(self, question: str, chunks: list[Chunk]) -> dict:
        user = excerpts_then_question(question, chunks)
        reply = extract_json(self.llm.chat(GRADE_SYSTEM, user))
        # If the model returns something unparseable, assume it's fine and move on
        # rather than looping forever; the answer step can still say NOT_FOUND.
        if not reply:
            return {"sufficient": True, "reason": "unparseable grade reply, assumed sufficient", "next_query": ""}
        return {
            "sufficient": bool(reply.get("sufficient", True)),
            "reason": str(reply.get("reason", "")),
            "next_query": str(reply.get("next_query", "")).strip(),
        }

    def retrieve(self, queries: list[str], plan: str, programme: str) -> list[Chunk]:
        """Search each query and keep every query's top_k, in order, without
        duplicates. The first search uses the student's own words and the
        rewrite, so a bad rewrite can add excerpts but can't remove any the
        question finds. (Merging the two lists and cutting back to top_k did
        remove them: the BIT FYP course line was lost that way.) fit_context
        keeps the total within the model's context window."""
        found: list[Chunk] = []
        for q in dict.fromkeys(q for q in queries if q):
            found += self.retriever.search(q, k=self.top_k, plan=plan, programme=programme)
        return dedupe(found)

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        user = excerpts_then_question(question, chunks)
        return self.llm.chat(ANSWER_SYSTEM, user, temperature=0.2)

    def run(self, question: str, batch: int | str | None = None, programme: str | None = None) -> Result:
        start = time.perf_counter()
        steps: list[Step] = []

        # 1. Which study plan does this student follow? Ask before doing any work.
        if batch is None or not programme:
            return self._ask_for_student("missing batch/programme", steps, start)
        programme = programme.strip().upper()
        try:
            plan = plan_for(batch, programme)
        except ValueError as e:
            return self._ask_for_student(str(e), steps, start)
        steps.append(Step("student", f"{programme} {batch} -> plan {plan}"))

        # The plan step: rewrite the question into a keyword query. Switchable because
        # the question alone found as many answers in testing; the Phase 4 eval decides.
        if self.rewrite:
            query = self.plan(question)
            steps.append(Step("plan", query))
        else:
            query = question.strip()
            steps.append(Step("plan", "rewrite off: searching the question only"))

        context: list[Chunk] = []
        tried = {query.lower(), question.strip().lower()}
        # first search: the student's own words, plus the rewrite if there is one
        queries = [question, query] if self.rewrite else [question]
        for attempt in range(self.max_retries + 1):
            # 2. Search only this student's plan and programme, plus the 'all' documents
            found = self.retrieve(queries, plan, programme)
            # newest results first, then earlier ones, within what the model's context window holds
            context = fit_context(dedupe(found + context), self.max_context_chars)
            searched = " + ".join(f"'{q}'" for q in dict.fromkeys(queries))
            steps.append(Step("retrieve", f"{len(found)} chunks for {searched} (plan {plan}, {programme}); "
                                          f"{len(context)} in context"))

            verdict = self.grade(question, context)
            steps.append(Step("grade", f"sufficient={verdict['sufficient']} — {verdict['reason']}"))

            if verdict["sufficient"] or attempt == self.max_retries:
                break
            next_q = drop_invented_codes(verdict["next_query"], question)
            if not next_q or next_q.lower() in tried:
                break  # nothing new to try
            tried.add(next_q.lower())
            queries = [next_q]

        # 3. Answer only from what was found, or say so.
        if not context:                                   # nothing retrieved: don't ask the LLM at all
            steps.append(Step("answer", "no excerpts found -> not found"))
            return self._not_found(steps, start, context)
        reply = self.answer(question, context)
        # "" when the model says the excerpts don't answer it (or wrote nothing);
        # a NOT_FOUND it added after a real answer is cut off here
        text = strip_marker(reply, NOT_FOUND)
        if not text:
            why = f"model replied {NOT_FOUND}" if NOT_FOUND in reply.upper() else "empty reply"
            steps.append(Step("answer", f"{why} from {len(context)} excerpts"))
            return self._not_found(steps, start, context)

        # 4. Sources are only the excerpts the answer cites, renumbered [1], [2], ...
        # to match. An answer without citations is kept (the model sometimes drops
        # them from a correct answer); the trace says so and the eval counts it.
        cited = renumber_citations(text, context)
        detail = f"{len(context)} excerpts in context; "
        detail += f"cited excerpts {cited.numbers}" if cited.sources else "no citations"
        if cited.unknown:                                 # numbers that weren't excerpts: [9], [35.1]
            detail += f"; removed unknown citations [{', '.join(cited.unknown)}]"
        if NOT_FOUND in reply.upper():
            detail += f"; removed {NOT_FOUND} after the answer"
        steps.append(Step("answer", detail))
        return Result(answer=cited.answer, sources=cited.sources, steps=steps,
                      seconds=time.perf_counter() - start, context=context)

    def _not_found(self, steps: list[Step], start: float, context: list[Chunk]) -> Result:
        return Result(answer=NOT_FOUND_MESSAGE, sources=[], steps=steps,
                      seconds=time.perf_counter() - start, found=False, context=context)

    def _ask_for_student(self, reason: str, steps: list[Step], start: float) -> Result:
        steps.append(Step("student", reason))
        return Result(answer=ASK_STUDENT, sources=[], steps=steps,
                      seconds=time.perf_counter() - start, needs_student_info=True)
