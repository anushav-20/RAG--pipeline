from __future__ import annotations

from pathlib import Path

SUPPORTED = {".txt", ".md", ".pdf"}


def load_file(path: Path) -> list[tuple[str, dict]]:
    """Return [(text, meta)] — one entry per PDF page, or one for text files."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return [(p.extract_text() or "", {"page": i + 1}) for i, p in enumerate(reader.pages)]
    if suffix in {".txt", ".md"}:
        return [(path.read_text(encoding="utf-8", errors="ignore"), {})]
    raise ValueError(f"Unsupported file type: {path}")


def iter_documents(root: str | Path):
    root = Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED)
    for f in files:
        for text, meta in load_file(f):
            if text.strip():
                yield f.name, text, meta
