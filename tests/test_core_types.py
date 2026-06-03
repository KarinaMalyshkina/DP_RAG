"""Tests for core data types."""

from __future__ import annotations

from dp_rag_eval.core.types import Document


def test_document_roundtrip():
    doc = Document(id="a", text="hello", metadata={"k": 1, "nested": [1, 2]})
    restored = Document.from_dict(doc.to_dict())
    assert restored == doc


def test_from_dict_defaults_metadata():
    doc = Document.from_dict({"id": "a", "text": "hello"})
    assert doc.metadata == {}


def test_copy_is_independent():
    doc = Document(id="a", text="hello", metadata={"k": 1})
    clone = doc.copy()
    clone.metadata["k"] = 999
    clone.text = "changed"
    # Mutating the clone must not touch the original.
    assert doc.metadata["k"] == 1
    assert doc.text == "hello"


def test_copy_overrides():
    doc = Document(id="a", text="hello", metadata={"k": 1})
    clone = doc.copy(id="b", text="world")
    assert clone.id == "b"
    assert clone.text == "world"
    assert clone.metadata == {"k": 1}
