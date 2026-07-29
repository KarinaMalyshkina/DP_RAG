"""Regression guard: verbatim duplicates must reach Grislain's vector store.

Upstream's ``PUPVectorStore.add`` is keyed on the text and silently drops an exact
repeat. Routing the corpus through it made the ``full_duplicate`` variant produce a
corpus identical to ``original`` — same AUC, same per-target scores across all 500
targets — which quietly emptied the duplication axis this thesis is about.

These tests need no model: they exercise the store's bookkeeping only.
"""

from __future__ import annotations

import pytest

from dp_rag_eval.datasets.variants import FullDuplicateVariant
from dp_rag_eval.wrappers.grislain_imports import PUPVectorStore, PUPVectorStoreConfig


def make_store() -> PUPVectorStore:
    return PUPVectorStore(PUPVectorStoreConfig(top_k=5, epsilon=1.0, max_retrieve=8, differential_pivacy=True))


def index_like_pipeline(store: PUPVectorStore, texts: list[str]) -> None:
    """The same bookkeeping as ``GrislainPipeline.index``, without the embedder."""
    for text in texts:
        store.store.append(text)
        store.index.setdefault(text, len(store.store) - 1)
    store._embeddings = None


def test_upstream_add_still_deduplicates():
    """Pin the behaviour being worked around, so an upstream change is visible."""
    store = make_store()
    store.add("a")
    store.add("a")
    store.add("b")

    assert store.store == ["a", "b"], "upstream no longer dedups — the workaround may be removable"


def test_direct_append_keeps_every_copy():
    store = make_store()

    index_like_pipeline(store, ["a", "a", "b"])

    assert store.store == ["a", "a", "b"]


def test_a_duplicated_corpus_keeps_its_duplicates(sample_documents):
    corpus = FullDuplicateVariant(duplication_rate=1.0, n_copies=1, seed=7).apply(sample_documents)
    assert len(corpus) == 2 * len(sample_documents)  # the variant did duplicate

    store = make_store()
    index_like_pipeline(store, [d.text for d in corpus])

    # The whole point: the store holds as many records as the variant produced.
    assert len(store.store) == len(corpus)
    assert len(set(store.store)) == len(sample_documents)  # and they are verbatim copies


def test_upstream_add_would_have_neutralised_the_variant(sample_documents):
    """Shows what the bug looked like: the duplicated corpus collapses to the original."""
    corpus = FullDuplicateVariant(duplication_rate=1.0, n_copies=1, seed=7).apply(sample_documents)

    store = make_store()
    for doc in corpus:
        store.add(doc.text)

    assert len(store.store) == len(sample_documents)
    assert len(store.store) != len(corpus)


def test_appending_invalidates_the_embedding_cache():
    store = make_store()
    store._embeddings = "stale"

    index_like_pipeline(store, ["a"])

    assert store._embeddings is None


@pytest.mark.parametrize("n_copies", [1, 2, 5])
def test_copy_count_reaches_the_store(sample_documents, n_copies):
    corpus = FullDuplicateVariant(duplication_rate=1.0, n_copies=n_copies, seed=3).apply(sample_documents)

    store = make_store()
    index_like_pipeline(store, [d.text for d in corpus])

    assert len(store.store) == len(sample_documents) * (1 + n_copies)
