"""Tests for JSONL serialisation of corpora."""

from __future__ import annotations

from dp_rag_eval.datasets.io import (
    load_documents_jsonl,
    save_documents_jsonl,
    to_texts,
)


def test_jsonl_roundtrip(sample_documents, tmp_path):
    path = tmp_path / "corpus.jsonl"
    save_documents_jsonl(sample_documents, path)
    loaded = load_documents_jsonl(path)
    assert loaded == sample_documents


def test_save_creates_parent_dirs(sample_documents, tmp_path):
    path = tmp_path / "nested" / "deep" / "corpus.jsonl"
    save_documents_jsonl(sample_documents, path)
    assert path.exists()


def test_to_texts(sample_documents):
    texts = to_texts(sample_documents)
    assert texts == [d.text for d in sample_documents]
