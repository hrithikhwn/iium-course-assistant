"""FastAPI server.

    uvicorn app.main:app --reload
    curl -X POST localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question": "When is the add/drop deadline?", "programme": "BCS", "batch": 241}'

Without programme and batch the agent doesn't search; the reply has
needs_student_info=true and asks for them, and the client asks the student
and sends the question again.

Each source names its document and page, and GET /documents/<name> serves
that document from data/raw/, so the web UI can open it at the cited page
(/documents/<name>#page=42).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import CourseAgent
from .config import settings
from .ingest import SUPPORTED
from .llm import OpenAICompatibleLLM
from .retriever import ChromaRetriever

app = FastAPI(title="IIUM Course Assistant", version="0.1.0")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    programme: str | None = Field(default=None, description="BCS or BIT")
    batch: int | str | None = Field(default=None, description="e.g. 241, or a matric number")
    trace: bool = False


class Source(BaseModel):
    ref: int
    citation: str
    source: str                # the document's file name: GET /documents/<source> serves it
    page: int | None = None    # the page in that document, from 1; None for files without pages
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]
    needs_student_info: bool = False   # true: ask for programme and batch, then send again
    found: bool = True                 # false: the documents don't answer this question
    steps: list[dict] | None = None
    seconds: float
    model: str


@lru_cache
def get_agent() -> CourseAgent:
    return CourseAgent(OpenAICompatibleLLM(), ChromaRetriever())


def documents(raw_dir: Path) -> dict[str, Path]:
    """Every file under raw_dir that ingest reads, by file name: the name
    sources cite. Requests are looked up here, so a name like '../.env'
    finds nothing, and no path from a request reaches the disk."""
    return {p.name: p for p in sorted(raw_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in SUPPORTED}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = get_agent().run(req.question, batch=req.batch, programme=req.programme)
    except RuntimeError as e:          # the vector store is empty
        raise HTTPException(status_code=503, detail=str(e))
    except httpx.HTTPError as e:       # Ollama or the LLM server is down, timed out or refused the request
        why = f"{type(e).__name__}: {e}" if str(e) else type(e).__name__
        raise HTTPException(status_code=503, detail=f"No answer from the model server ({why}). "
                                                    "Check that Ollama is running, or the LLM settings in .env.")
    return AskResponse(
        answer=result.answer,
        # the whole excerpt (up to CHUNK_SIZE characters): the cited fact can be
        # anywhere in it, and in the eval most cited excerpts were over 300
        sources=[Source(ref=i, citation=c.citation(), source=c.source, page=c.page, excerpt=c.text)
                 for i, c in enumerate(result.sources, 1)],
        needs_student_info=result.needs_student_info,
        found=result.found,
        steps=[{"name": s.name, "detail": s.detail} for s in result.steps] if req.trace else None,
        seconds=round(result.seconds, 2),
        model=settings.llm_model,
    )


@app.get("/documents/{name}")
def document(name: str) -> FileResponse:
    """A cited document, shown in the browser rather than downloaded. The UI
    links to /documents/<name>#page=<page>, which PDF viewers open at that page."""
    path = documents(settings.raw_dir).get(name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"No document named '{name}' in {settings.raw_dir}")
    return FileResponse(path, filename=name, content_disposition_type="inline")
