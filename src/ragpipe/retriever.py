from __future__ import annotations

from .bm25 import BM25, tokenize
from .embeddings import Embedder
from .store import VectorStore
from .types import Hit


def reciprocal_rank_fusion(rankings: dict[str, list[int]], k: int = 60) -> dict[int, float]:
    """RRF: score(d) = sum_r 1/(k + rank_r(d)). Robust fusion of incomparable score scales."""
    fused: dict[int, float] = {}
    for ranking in rankings.values():
        for rank, doc in enumerate(ranking, start=1):
            fused[doc] = fused.get(doc, 0.0) + 1.0 / (k + rank)
    return fused


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, candidates: int = 20, rrf_k: int = 60):
        self.store, self.embedder = store, embedder
        self.bm25 = BM25([c.text for c in store.chunks])
        self.candidates, self.rrf_k = candidates, rrf_k

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        n = self.candidates
        dense = self.store.search(self.embedder.embed([query])[0], n)
        sparse_scores = self.bm25.scores(query)
        sparse = [(int(i), float(sparse_scores[i])) for i in sparse_scores.argsort()[::-1][:n] if sparse_scores[i] > 0]
        if mode == "dense":
            fused = {i: s for i, s in dense}
        elif mode == "bm25":
            fused = {i: s for i, s in sparse}
        else:
            fused = reciprocal_rank_fusion(
                {"dense": [i for i, _ in dense], "bm25": [i for i, _ in sparse]}, self.rrf_k
            )
        d_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
        s_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
        top = sorted(fused.items(), key=lambda x: -x[1])[:k]
        return [
            Hit(self.store.chunks[i], score, {"dense_rank": d_rank.get(i), "bm25_rank": s_rank.get(i)})
            for i, score in top
        ]


class LexicalReranker:
    """Cheap, dependency-free reranker: blends fused score with query-term coverage.
    Replace with CrossEncoderReranker for production quality."""

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        q = set(tokenize(query))
        if not q:
            return hits[:k]
        mx = max((h.score for h in hits), default=1.0) or 1.0
        for h in hits:
            cov = len(q & set(tokenize(h.chunk.text))) / len(q)
            h.score = 0.5 * (h.score / mx) + 0.5 * cov
        return sorted(hits, key=lambda h: -h.score)[:k]


class CrossEncoderReranker:
    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder  # pip install ragpipe[local]

        self.model = CrossEncoder(model)

    def rerank(self, query: str, hits: list[Hit], k: int) -> list[Hit]:
        scores = self.model.predict([(query, h.chunk.text) for h in hits])
        for h, s in zip(hits, scores):
            h.score = float(s)
        return sorted(hits, key=lambda h: -h.score)[:k]
