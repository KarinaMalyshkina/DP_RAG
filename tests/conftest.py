"""Shared pytest fixtures for the dp-rag-eval test suite.

Tests here are deliberately lightweight: no GPU, no model downloads, no network.
The HealthCareMagic loader test mocks ``datasets.load_dataset``.
"""

from __future__ import annotations

import pytest

from dp_rag_eval.core.types import Document


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small synthetic corpus with multi-sentence texts and unique ids."""
    return [
        Document(
            id=f"doc-{i:03d}",
            text=(
                f"This is the first sentence of document {i}. "
                f"Here is a second, longer sentence describing topic number {i} in some detail. "
                f"A third short note {i}."
            ),
            metadata={"source": "synthetic", "query": f"question {i}", "variant": "original"},
        )
        for i in range(10)
    ]
