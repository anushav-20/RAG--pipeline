from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .loaders import SUPPORTED
from .pipeline import RAGPipeline

SAMPLE_DIR = os.getenv("RAG_SAMPLE_DIR", "data/docs")
UPLOAD_DIR = os.getenv("RAG_UPLOAD_DIR", "data/uploads")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="ragpipe", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("RAG_CORS", "http://localhost:5173").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)
_pipe: RAGPipeline | None = None


def get_pipe() -> RAGPipeline:
    """Lazily load the index; build it from the bundled sample docs on first run."""
    global _pipe
    if _pipe is None:
        p = RAGPipeline()
        try:
            p.load()
        except FileNotFoundError:
            p.ingest([SAMPLE_DIR, UPLOAD_DIR])
        _pipe = p
    return _pipe


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(5, ge=1, le=20)
    mode: str = Field("hybrid", pattern="^(hybrid|dense|bm25)$")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/stats")
def stats():
    pipe = get_pipe()
    return {**pipe.stats(), "llm": type(pipe.generator).__name__}


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


@app.post("/upload")
async def upload(files: Annotated[list[UploadFile], File()]):
    updir = Path(UPLOAD_DIR)
    updir.mkdir(parents=True, exist_ok=True)
    saved = []
    for f in files:
        name = Path(f.filename or "").name  # strip any path components
        if Path(name).suffix.lower() not in SUPPORTED:
            raise HTTPException(400, f"Unsupported file type: {name}. Use {sorted(SUPPORTED)}")
        data = await f.read()
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{name} exceeds 20 MB")
        (updir / name).write_bytes(data)
        saved.append(name)
    pipe = get_pipe()
    pipe.ingest([SAMPLE_DIR, UPLOAD_DIR])
    return {"uploaded": saved, **pipe.stats()}


@app.delete("/documents/{name}")
def delete_document(name: str):
    target = Path(UPLOAD_DIR) / Path(name).name
    if not target.is_file():
        raise HTTPException(404, "Only uploaded documents can be removed")
    target.unlink()
    pipe = get_pipe()
    pipe.ingest([SAMPLE_DIR, UPLOAD_DIR])
    return pipe.stats()
