"""Tests for the four controlled dataset variants.

Cover: determinism, document counts, provenance metadata, input immutability,
and the specific transformation each variant performs.
"""

from __future__ import annotations

from dp_rag_eval.datasets.variants import (
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
    split_sentences,
)


def _texts(docs):
    return [d.text for d in docs]


# --- split_sentences ---------------------------------------------------------


def test_split_sentences():
    s = split_sentences("First sentence. Second one! A third? ")
    assert s == ["First sentence.", "Second one!", "A third?"]


# --- OriginalVariant ---------------------------------------------------------


def test_original_is_control(sample_documents):
    out = OriginalVariant().apply(sample_documents)
    assert len(out) == len(sample_documents)
    assert all(d.metadata["variant"] == "original" for d in out)
    assert all(d.metadata["is_duplicate"] is False for d in out)
    # Same texts, but does not mutate input objects.
    assert _texts(out) == _texts(sample_documents)


def test_original_does_not_mutate_input(sample_documents):
    before = _texts(sample_documents)
    OriginalVariant().apply(sample_documents)
    assert _texts(sample_documents) == before


# --- FullDuplicateVariant ----------------------------------------------------


def test_full_duplicate_counts_and_verbatim(sample_documents):
    n = len(sample_documents)
    variant = FullDuplicateVariant(duplication_rate=0.3, n_copies=2, seed=42)
    out = variant.apply(sample_documents)

    expected_sources = max(1, round(0.3 * n))
    assert len(out) == n + expected_sources * 2

    dups = [d for d in out if d.metadata.get("is_duplicate")]
    assert len(dups) == expected_sources * 2
    by_id = {d.id: d for d in out}
    for dup in dups:
        source = by_id[dup.metadata["duplicate_of"]]
        assert dup.text == source.text  # verbatim copy
        assert dup.id != source.id


def test_full_duplicate_deterministic(sample_documents):
    a = FullDuplicateVariant(0.3, 1, seed=7).apply(sample_documents)
    b = FullDuplicateVariant(0.3, 1, seed=7).apply(sample_documents)
    assert [(d.id, d.text) for d in a] == [(d.id, d.text) for d in b]


def test_full_duplicate_does_not_mutate_input(sample_documents):
    before = _texts(sample_documents)
    FullDuplicateVariant(0.5, 3, seed=1).apply(sample_documents)
    assert _texts(sample_documents) == before


# --- PartialOverlapVariant ---------------------------------------------------


def test_partial_overlap_shares_fragments(sample_documents):
    n = len(sample_documents)
    out = PartialOverlapVariant(overlap_rate=0.3, n_hosts=3, seed=42).apply(sample_documents)
    # Partial overlap does not change the corpus size.
    assert len(out) == n

    hosts = [d for d in out if d.metadata.get("shared_fragment_from")]
    assert hosts, "expected at least one host carrying a shared fragment"
    for host in hosts:
        for fragment in host.metadata["shared_fragment"]:
            assert fragment in host.text  # fragment really embedded


def test_partial_overlap_deterministic(sample_documents):
    a = PartialOverlapVariant(0.3, 3, seed=5).apply(sample_documents)
    b = PartialOverlapVariant(0.3, 3, seed=5).apply(sample_documents)
    assert _texts(a) == _texts(b)


def test_partial_overlap_does_not_mutate_input(sample_documents):
    before = _texts(sample_documents)
    PartialOverlapVariant(0.5, 2, seed=3).apply(sample_documents)
    assert _texts(sample_documents) == before


# --- CanaryInjectionVariant --------------------------------------------------


def test_canary_ground_truth_matches_documents(sample_documents):
    variant = CanaryInjectionVariant(n_canaries=4, n_repeats=2, seed=42)
    out = variant.apply(sample_documents)

    assert len(variant.injected_canaries) == 4
    by_id = {d.id: d for d in out}
    for canary in variant.injected_canaries:
        assert canary["n_repeats"] == 2
        assert len(canary["host_doc_ids"]) == 2
        for host_id in canary["host_doc_ids"]:
            host = by_id[host_id]
            assert canary["text"] in host.text
            assert host.metadata["is_canary_host"] is True
            assert canary["canary_id"] in host.metadata["canary_ids"]


def test_canary_secrets_are_unique(sample_documents):
    variant = CanaryInjectionVariant(n_canaries=5, n_repeats=1, seed=42)
    variant.apply(sample_documents)
    codes = [c["secret_code"] for c in variant.injected_canaries]
    assert len(set(codes)) == len(codes)


def test_canary_deterministic(sample_documents):
    a = CanaryInjectionVariant(3, 1, seed=9)
    b = CanaryInjectionVariant(3, 1, seed=9)
    a.apply(sample_documents)
    b.apply(sample_documents)
    assert [c["secret_code"] for c in a.injected_canaries] == [c["secret_code"] for c in b.injected_canaries]
    assert [c["host_doc_ids"] for c in a.injected_canaries] == [c["host_doc_ids"] for c in b.injected_canaries]


def test_canary_does_not_mutate_input(sample_documents):
    before = _texts(sample_documents)
    CanaryInjectionVariant(3, 2, seed=1).apply(sample_documents)
    assert _texts(sample_documents) == before
