"""The agentic RAG loop: plan -> retrieve -> grade -> (retry) -> answer."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import settings
from .llm import LLM
from .retriever import Retriever
from .text import Chunk, dedupe, extract_json, format_context, plan_for

ASK_STUDENT = "Which programme are you in (BCS or BIT), and what's your batch (e.g. 241)?"

PLAN_SYSTEM = """You turn a student's question about IIUM (International Islamic University Malaysia) \
courses, handbooks and academic rules into ONE short search query for a document search engine.
Keep course codes (e.g. INFO 4303) and key terms exactly. Reply with JSON only:
{"query": "<search query>"}"""

GRADE_SYSTEM = """You check whether document excerpts contain enough information to answer a question.
Reply with JSON only:
{"sufficient": true|false, "reason": "<one short sentence>", "next_query": "<a different search query to try if not sufficient, else empty>"}"""

ANSWER_SYSTEM = """You are an assistant for IIUM students. Answer ONLY from the numbered excerpts provided.
- Cite excerpts inline like [1] or [2][3] after the sentences they support.
- If the excerpts do not contain the answer, say you couldn't find it in the documents and suggest who to ask (e.g. the Kulliyyah office or academic advisor). Do not guess.
- Be concise and direct."""


@dataclass
class Step:
    name: str
    detail: str


@dataclass
class Result:
    answer: str
    sources: list[Chunk]
    steps: list[Step] = field(default_factory=list)
    seconds: float = 0.0
    # True when the agent stopped to ask for the student's batch/programme
    # instead of answering; the CLI/UI should ask and call run() again
    needs_student_info: bool = False


class CourseAgent:
    def __init__(self, llm: LLM, retriever: Retriever, top_k: int = settings.top_k, max_retries: int = settings.max_retries):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k
        self.max_retries = max_retries

    def plan(self, question: str) -> str:
        reply = extract_json(self.llm.chat(PLAN_SYSTEM, question))
        return (reply.get("query") or question).strip()

    def grade(self, question: str, chunks: list[Chunk]) -> dict:
        user = f"Question: {question}\n\nExcerpts:\n{format_context(chunks)}"
        reply = extract_json(self.llm.chat(GRADE_SYSTEM, user))
        # If the model returns something unparseable, assume it's fine and move on
        # rather than looping forever.
        return {
            "sufficient": bool(reply.get("sufficient", True)),
            "reason": str(reply.get("reason", "")),
            "next_query": str(reply.get("next_query", "")).strip(),
        }

    def answer(self, question: str, chunks: list[Chunk]) -> str:
        user = f"Question: {question}\n\nExcerpts:\n{format_context(chunks)}"
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

        query = self.plan(question)
        steps.append(Step("plan", query))

        context: list[Chunk] = []
        tried = {query.lower()}
        for attempt in range(self.max_retries + 1):
            # 2. Search only this student's plan and programme, plus the 'all' documents
            found = self.retriever.search(query, k=self.top_k, plan=plan, programme=programme)
            context = dedupe(context + found)
            steps.append(Step("retrieve", f"{len(found)} chunks for '{query}' (plan {plan}, {programme})"))

            verdict = self.grade(question, context)
            steps.append(Step("grade", f"sufficient={verdict['sufficient']} — {verdict['reason']}"))

            if verdict["sufficient"] or attempt == self.max_retries:
                break
            next_q = verdict["next_query"]
            if not next_q or next_q.lower() in tried:
                break  # nothing new to try
            tried.add(next_q.lower())
            query = next_q

        text = self.answer(question, context)
        steps.append(Step("answer", f"{len(context)} excerpts in context"))
        return Result(answer=text, sources=context, steps=steps, seconds=time.perf_counter() - start)

    def _ask_for_student(self, reason: str, steps: list[Step], start: float) -> Result:
        steps.append(Step("student", reason))
        return Result(answer=ASK_STUDENT, sources=[], steps=steps,
                      seconds=time.perf_counter() - start, needs_student_info=True)
