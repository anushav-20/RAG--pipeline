from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline

app = FastAPI(title="ragpipe", version="0.1.0")
_pipe: RAGPipeline | None = None


def get_pipe() -> RAGPipeline:
    global _pipe
    if _pipe is None:
        _pipe = RAGPipeline()
        try:
            _pipe.load()
        except FileNotFoundError:
            raise HTTPException(503, "No index found. Run `ragpipe ingest <path>` first.")
    return _pipe


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    a = get_pipe().ask(req.question, req.k, req.mode)
    return {
        "answer": a.answer,
        "grounded": a.grounded,
        "citations": [
            {"n": i, "source": h.chunk.source, "score": round(h.score, 4), "text": h.chunk.text, **h.sources}
            for i, h in enumerate(a.hits, 1)
        ],
    }
