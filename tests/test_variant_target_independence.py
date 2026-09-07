"""The corpus a variant builds must not depend on which documents the attack scores.

Both draws used ``random.Random(seed).sample`` over populations of the same size.
``random.sample`` takes the same algorithmic branch for a given population size, so
the two sequences agreed on a prefix and every duplicated document was one of the
first scored targets: at a duplication rate of 0.1, 100 of 250 scored members were
duplicated where ~25 were expected, and from 0.25 upward all of them were. Every
rate at or above 0.3 therefore measured the same treated population, which is why
the dose-response curve flattened and then fell — the fall was the corpus growing,
not duplication ceasing to matter.

These tests pin the separation. If one fails, the duplication axis is measuring the
adversary's aim rather than the corpus's redundancy again.
"""

from __future__ import annotations

import random

import pytest

from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.variants import FullDuplicateVariant, PartialOverlapVariant, treated_doc_ids
from dp_rag_eval.runner.sweep import make_variant, seeded_sample, variant_seed

POPULATION = 1000
TARGETS = 250


def corpus(n: int = POPULATION) -> list[Document]:
    return [
        Document(id=f"doc-{i:04d}", text=f"Document number {i}. A second sentence about topic {i}.") for i in range(n)
    ]


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_variant_seed_differs_from_the_experiment_seed(seed: int) -> None:
    assert variant_seed(seed) != seed


@pytest.mark.parametrize("seed", [42, 43, 44])
@pytest.mark.parametrize("rate", [0.1, 0.2, 0.3, 0.5, 0.9])
def test_treated_target_fraction_tracks_the_duplication_rate(seed: int, rate: float) -> None:
    """The property the phrase "duplication rate" promises a reader."""
    base = corpus()
    targets = seeded_sample(base, TARGETS, seed)
    built = make_variant("full_duplicate", {"duplication_rate": rate, "n_copies": 1}, seed)
    treated = treated_doc_ids(built.apply(base))

    hits = sum(1 for doc in targets if doc.id in treated)
    expected = rate * TARGETS
    # Hypergeometric; three standard deviations is ~20 targets at the worst rate, and
    # the failure this guards against overshoots by 75-225, not by 20.
    assert abs(hits - expected) < 0.25 * TARGETS, f"{hits} of {TARGETS} treated, expected about {expected:.0f}"


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_the_old_coupling_would_have_failed_this(seed: int) -> None:
    """Documents the defect, so the test above cannot be weakened without noticing.

    Reproduces the two draws as they were: same seed, same population size.
    """
    base = corpus()
    targets = {doc.id for doc in seeded_sample(base, TARGETS, seed)}
    coupled = {base[i].id for i in sorted(random.Random(seed).sample(range(POPULATION), 100))}
    assert len(coupled & targets) == 100  # every duplicated document was a scored target


@pytest.mark.parametrize("rate", [0.1, 0.5])
def test_corpus_side_is_unchanged(rate: float) -> None:
    """The definition was never wrong — the rate still means "this share of the corpus"."""
    base = corpus()
    out = make_variant("full_duplicate", {"duplication_rate": rate, "n_copies": 1}, 42).apply(base)
    duplicates = [doc for doc in out if doc.metadata.get("is_duplicate")]
    assert len(duplicates) == round(rate * POPULATION)
    assert len(out) == POPULATION + len(duplicates)


def test_treated_ids_exclude_the_copies_themselves() -> None:
    base = corpus(100)
    out = FullDuplicateVariant(duplication_rate=0.5, n_copies=3, seed=1).apply(base)
    treated = treated_doc_ids(out)
    assert all(not doc_id.endswith(("-dup0", "-dup1", "-dup2")) for doc_id in treated)
    assert len(treated) == 50


def test_treated_ids_cover_fragment_donors() -> None:
    base = corpus(100)
    out = PartialOverlapVariant(overlap_rate=0.3, n_hosts=3, seed=1).apply(base)
    assert len(treated_doc_ids(out)) == pytest.approx(30, abs=2)


def test_variants_stay_deterministic() -> None:
    """Decoupling must not have made the corpus depend on anything but the seed."""
    base = corpus(200)
    first = make_variant("full_duplicate", {"duplication_rate": 0.3, "n_copies": 2}, 42).apply(base)
    second = make_variant("full_duplicate", {"duplication_rate": 0.3, "n_copies": 2}, 42).apply(base)
    assert [d.id for d in first] == [d.id for d in second]
