from __future__ import annotations

from .chunking import chunk_text
from .config import Config
from .embeddings import get_embedder
from .generator import NO_ANSWER, get_generator
from .loaders import iter_documents
from .retriever import HybridRetriever, LexicalReranker
from .store import VectorStore
from .types import Answer, Chunk, Hit


class RAGPipeline:
    def __init__(self, config: Config | None = None, generator=None, reranker=None):
        self.cfg = config or Config()
        self.embedder = get_embedder(self.cfg.embedder, self.cfg.st_model)
        self.generator = generator or get_generator(self.cfg.llm_model, self.cfg.max_tokens)
        self.reranker = reranker or LexicalReranker()
        self.retriever: HybridRetriever | None = None

    # ---- indexing -------------------------------------------------------
    def ingest(self, path: str | list[str]) -> int:
        paths = [path] if isinstance(path, str) else path
        chunks: list[Chunk] = []
        for root in paths:
            for source, text, meta in iter_documents(root):
                for piece in chunk_text(text, self.cfg.chunk_size, self.cfg.chunk_overlap):
                    chunks.append(Chunk(len(chunks), piece, source, dict(meta)))
        if not chunks:
            raise ValueError(f"No supported documents found in {paths}")
        store = VectorStore(chunks, self.embedder.embed([c.text for c in chunks]), self.embedder.name)
        store.save(self.cfg.index_dir)
        self._attach(store)
        return len(chunks)

    def load(self) -> None:
        store = VectorStore.load(self.cfg.index_dir)
        if store.embedder_name != self.embedder.name:
            raise RuntimeError(
                f"Index built with '{store.embedder_name}' but pipeline uses '{self.embedder.name}'. Re-ingest."
            )
        self._attach(store)

    def _attach(self, store: VectorStore) -> None:
        self.retriever = HybridRetriever(store, self.embedder, self.cfg.candidates, self.cfg.rrf_k)

    def stats(self) -> dict:
        if self.retriever is None:
            return {"chunks": 0, "documents": []}
        counts: dict[str, int] = {}
        for c in self.retriever.store.chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return {"chunks": sum(counts.values()), "embedder": self.embedder.name,
                "documents": [{"name": n, "chunks": k} for n, k in sorted(counts.items())]}

    # ---- querying -------------------------------------------------------
    def retrieve(self, question: str, k: int | None = None, mode: str = "hybrid", rerank: bool = True) -> list[Hit]:
        if self.retriever is None:
            self.load()
        k = k or self.cfg.top_k
        hits = self.retriever.retrieve(question, k=self.cfg.candidates if rerank else k, mode=mode)
        return self.reranker.rerank(question, hits, k) if rerank else hits

    def ask(self, question: str, k: int | None = None, mode: str = "hybrid") -> Answer:
        hits = self.retrieve(question, k, mode)
        text = self.generator.generate(question, hits)
        return Answer(question, text, hits, grounded=text.strip() != NO_ANSWER)
