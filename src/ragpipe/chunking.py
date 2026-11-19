from __future__ import annotations

import re

_SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text: str, size: int, seps: list[str]) -> list[str]:
    if len(text) <= size:
        return [text]
    if not seps:
        return [text[i : i + size] for i in range(0, len(text), size)]
    sep, rest = seps[0], seps[1:]
    pieces = [p for p in text.split(sep) if p]
    out: list[str] = []
    for p in pieces:
        out.extend(_split(p, size, rest) if len(p) > size else [p])
    # re-add separator when merging later
    return out


def chunk_text(text: str, size: int = 800, overlap: int = 120) -> list[str]:
    """Recursive, structure-aware splitter: prefers paragraph > line > sentence > word
    boundaries, packs pieces up to `size` chars, and carries `overlap` chars of context."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    pieces = _split(text, size, _SEPARATORS)
    chunks: list[str] = []
    cur = ""
    for p in pieces:
        if cur and len(cur) + len(p) + 1 > size:
            chunks.append(cur.strip())
            tail = cur[-overlap:] if overlap else ""
            # start the overlap on a word boundary
            tail = tail[tail.find(" ") + 1 :] if " " in tail else tail
            cur = f"{tail} {p}"
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    return chunks
