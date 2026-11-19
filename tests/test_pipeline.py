import numpy as np
import pytest
from fastapi.testclient import TestClient

from ragpipe.bm25 import BM25
from ragpipe.chunking import chunk_text
from ragpipe.config import Config
from ragpipe.evaluate import evaluate
from ragpipe.generator import NO_ANSWER, ExtractiveGenerator
from ragpipe.pipeline import RAGPipeline
from ragpipe.retriever import reciprocal_rank_fusion


@pytest.fixture()
def pipe(tmp_path):
    p = RAGPipeline(Config(index_dir=str(tmp_path / "idx"), embedder="hashing"), generator=ExtractiveGenerator())
    p.ingest("data/docs")
    return p


def test_chunking_respects_size_and_overlap():
    text = ("Sentence number one is here. " * 60).strip()
    chunks = chunk_text(text, size=200, overlap=40)
    assert len(chunks) > 3
    assert all(len(c) <= 260 for c in chunks)


def test_rrf_prefers_docs_ranked_by_both():
    fused = reciprocal_rank_fusion({"a": [1, 2, 3], "b": [3, 1, 4]})
    assert max(fused, key=fused.get) == 1


def test_bm25_ranks_exact_term_first():
    s = BM25(["cats purr loudly", "error code E4012 occurred", "dogs bark"]).scores("E4012")
    assert int(np.argmax(s)) == 1


def test_retrieval_quality_floor(pipe):
    r = evaluate(pipe, "eval/dataset.jsonl", k=5)
    assert r["hit@5"] >= 0.8 and r["mrr"] >= 0.5


def test_hybrid_not_worse_than_dense(pipe):
    d = evaluate(pipe, "eval/dataset.jsonl", 5, "dense", False)
    h = evaluate(pipe, "eval/dataset.jsonl", 5, "hybrid", False)
    assert h["mrr"] >= d["mrr"] - 0.05


def test_reload_roundtrip_and_ask(pipe, tmp_path):
    p2 = RAGPipeline(Config(index_dir=pipe.cfg.index_dir, embedder="hashing"), generator=ExtractiveGenerator())
    ans = p2.ask("What is Reciprocal Rank Fusion?")
    assert ans.hits[0].chunk.source == "retrieval.md" and ans.grounded


def test_abstains_with_no_hits():
    assert ExtractiveGenerator().generate("q", []) == NO_ANSWER


def test_api(pipe, monkeypatch):
    from ragpipe import api

    api._pipe = pipe
    c = TestClient(api.app)
    assert c.get("/health").json() == {"status": "ok"}
    r = c.post("/ask", json={"question": "What is MRR?"}).json()
    assert r["citations"] and r["citations"][0]["source"] == "evaluation.md"
    assert c.post("/ask", json={"question": ""}).status_code == 422
