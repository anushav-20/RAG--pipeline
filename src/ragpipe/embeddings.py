from __future__ import annotations

import hashlib
import re
from itertools import pairwise
from typing import Protocol

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...  # L2-normalised, shape (n, d)


class HashingEmbedder:
    """Dependency-free, deterministic embedder (hashed unigrams+bigrams, sublinear TF).
    Works fully offline — used for tests/CI and as a zero-setup default.
    For real semantic quality use SentenceTransformerEmbedder."""

    name = "hashing"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _vec(self, text: str) -> np.ndarray:
        toks = _TOKEN.findall(text.lower())
        feats = toks + [f"{a}_{b}" for a, b in pairwise(toks)]
        v = np.zeros(self.dim, dtype=np.float32)
        for f in feats:
            h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        v = np.sign(v) * np.log1p(np.abs(v))
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class SentenceTransformerEmbedder:
    def __init__(self, model: str):
        from sentence_transformers import SentenceTransformer  # pip install ragpipe[local]

        self.name = f"st:{model}"
        self._m = SentenceTransformer(model)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._m.encode(texts, normalize_embeddings=True, show_progress_bar=False), np.float32)


def get_embedder(kind: str, st_model: str = "") -> Embedder:
    if kind == "hashing":
        return HashingEmbedder()
    if kind in {"sentence-transformers", "st"}:
        return SentenceTransformerEmbedder(st_model)
    raise ValueError(f"Unknown embedder: {kind}")
