"""JSONL serialisation for corpora of :class:`Document`."""

from __future__ import annotations

import json
from pathlib import Path

from dp_rag_eval.core.types import Document


def save_documents_jsonl(documents: list[Document], path: str | Path) -> Path:
    """Write a corpus to a JSONL file (one document per line).

    Args:
        documents: The corpus to serialise.
        path: Destination file. Parent directories are created if missing.

    Returns:
        The resolved destination path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False))
            fh.write("\n")
    return path


def load_documents_jsonl(path: str | Path) -> list[Document]:
    """Read a corpus previously written by :func:`save_documents_jsonl`."""
    path = Path(path)
    documents: list[Document] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                documents.append(Document.from_dict(json.loads(line)))
    return documents


def to_texts(documents: list[Document]) -> list[str]:
    """Return just the document texts.

    Convenience for pipelines that ingest raw strings (e.g. Grislain's
    ``DPRAGEngine.add``).
    """
    return [doc.text for doc in documents]
