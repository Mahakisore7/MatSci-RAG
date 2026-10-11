"""REST API for MatSci-RAG — a thin HTTP layer over src.rag.rag_query.

Run from the project root:
    uv run uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Interactive docs are served at /docs (Swagger UI).
"""

import threading
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.core.config import active_generator

ABSTENTION_MARKER = "Multiple candidate materials found"

app = FastAPI(
    title="MatSci-RAG API",
    description="Grounded question answering over 400 Solid Oxide Fuel Cell papers.",
    version="1.0.0",
)

# The generator is a single in-process model; serialize access so concurrent
# requests don't interleave generate() calls on the same weights.
_pipeline_lock = threading.Lock()


@lru_cache(maxsize=1)
def get_pipeline():
    # Deferred so importing this module (e.g. in tests) doesn't load ChromaDB
    # and the embedding/generator models.
    from src.rag.rag_query import answer_query, verify_answer

    return answer_query, verify_answer


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    use_hybrid: bool = True
    expand_query: bool = False
    verify: bool = True
    chat_history: list[ChatMessage] = Field(default_factory=list, max_length=20)


class Source(BaseModel):
    rank: int
    title: str
    paper_id: str
    section: str | None
    text: str


class QueryResponse(BaseModel):
    answer: str
    abstained: bool
    verification: str | None
    sources: list[Source]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "generator": active_generator()}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    answer_query, verify_answer = get_pipeline()
    history = [m.model_dump() for m in req.chat_history]

    with _pipeline_lock:
        answer, docs = answer_query(
            req.question,
            use_hybrid=req.use_hybrid,
            chat_history=history,
            expand_q=req.expand_query,
        )
        abstained = ABSTENTION_MARKER in answer
        verification = None
        if req.verify and not abstained:
            verification = verify_answer(req.question, answer, docs)

    sources = [
        Source(
            rank=i,
            title=d.metadata.get("title", "Unknown Title"),
            paper_id=d.metadata.get("paper_id", ""),
            section=d.metadata.get("section"),
            text=d.page_content,
        )
        for i, d in enumerate(docs, 1)
    ]
    return QueryResponse(
        answer=answer, abstained=abstained, verification=verification, sources=sources
    )
