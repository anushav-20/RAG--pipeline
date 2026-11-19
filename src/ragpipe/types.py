from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: int
    text: str
    source: str
    meta: dict = field(default_factory=dict)


@dataclass
class Hit:
    chunk: Chunk
    score: float
    sources: dict = field(default_factory=dict)  # per-retriever rank/score, for debugging


@dataclass
class Answer:
    question: str
    answer: str
    hits: list[Hit]
    grounded: bool
