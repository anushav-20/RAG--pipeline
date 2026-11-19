from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import Chunk


class VectorStore:
    """Minimal persistent dense store (cosine via dot product on normalised vectors).
    Swap for FAISS / Qdrant / pgvector behind the same two methods when scaling."""

    def __init__(self, chunks: list[Chunk], vectors: np.ndarray, embedder_name: str):
        self.chunks, self.vectors, self.embedder_name = chunks, vectors, embedder_name

    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.chunks):
            return []
        sims = self.vectors @ qvec
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx]

    def save(self, directory: str) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectors.npy", self.vectors)
        meta = {"embedder": self.embedder_name, "chunks": [c.__dict__ for c in self.chunks]}
        (d / "chunks.json").write_text(json.dumps(meta))

    @classmethod
    def load(cls, directory: str) -> VectorStore:
        d = Path(directory)
        meta = json.loads((d / "chunks.json").read_text())
        chunks = [Chunk(**c) for c in meta["chunks"]]
        return cls(chunks, np.load(d / "vectors.npy"), meta["embedder"])
