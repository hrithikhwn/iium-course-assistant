"""FastAPI server.

    uvicorn app.main:app --reload
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "When is the add/drop deadline?"}'
"""
from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import CourseAgent
from .config import settings
from .llm import OpenAICompatibleLLM
from .retriever import ChromaRetriever

app = FastAPI(title="IIUM Course Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    trace: bool = False


class Source(BaseModel):
    ref: int
    citation: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    steps: list[dict] | None = None
    seconds: float
    model: str


@lru_cache
def get_agent() -> CourseAgent:
    return CourseAgent(OpenAICompatibleLLM(), ChromaRetriever())


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = get_agent().run(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return AskResponse(
        answer=result.answer,
        sources=[Source(ref=i, citation=c.citation(), excerpt=c.text[:300]) for i, c in enumerate(result.sources, 1)],
        steps=[{"name": s.name, "detail": s.detail} for s in result.steps] if req.trace else None,
        seconds=round(result.seconds, 2),
        model=settings.llm_model,
    )
