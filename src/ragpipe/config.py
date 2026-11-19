from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    index_dir: str = field(default_factory=lambda: os.getenv("RAG_INDEX_DIR", "index"))
    embedder: str = field(default_factory=lambda: os.getenv("RAG_EMBEDDER", "hashing"))
    st_model: str = field(default_factory=lambda: os.getenv("RAG_ST_MODEL", "BAAI/bge-small-en-v1.5"))
    llm_model: str = field(default_factory=lambda: os.getenv("RAG_MODEL", "claude-opus-5-5"))
    chunk_size: int = 800  # characters
    chunk_overlap: int = 120
    top_k: int = 5
    candidates: int = 20  # per retriever, before fusion/rerank
    rrf_k: int = 60
    max_tokens: int = 1024
