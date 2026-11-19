"""Retrieval evaluation: Hit@k and MRR over a JSONL set of
{"question": ..., "relevant_sources": ["file.md"], "keywords": ["must-appear", ...]}.
A retrieved chunk is relevant if it comes from a relevant source AND contains a keyword
(when keywords are given), so scores reflect passage-level — not just file-level — retrieval."""
from __future__ import annotations

import json

from .pipeline import RAGPipeline


def _relevant(hit, item) -> bool:
    if hit.chunk.source not in item["relevant_sources"]:
        return False
    kws = item.get("keywords") or []
    return not kws or any(k.lower() in hit.chunk.text.lower() for k in kws)


def evaluate(pipe: RAGPipeline, dataset_path: str, k: int = 5, mode: str = "hybrid", rerank: bool = True) -> dict:
    with open(dataset_path) as f:
        items = [json.loads(line) for line in f if line.strip()]
    hits_at_k, rr = 0, 0.0
    for item in items:
        hits = pipe.retrieve(item["question"], k, mode, rerank)
        rank = next((i for i, h in enumerate(hits, 1) if _relevant(h, item)), None)
        if rank:
            hits_at_k += 1
            rr += 1 / rank
    n = len(items)
    return {"mode": mode, "rerank": rerank, "k": k, "n": n, f"hit@{k}": round(hits_at_k / n, 3), "mrr": round(rr / n, 3)}
