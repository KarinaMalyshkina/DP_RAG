"""Retrieval must not collapse when a document has many byte-identical copies.

Upstream's non-DP threshold is the ``(top_k + 1)``-th highest score, and selection keeps
scores **strictly** above it. Identical documents score identically, so at ``top_k + 1``
copies the threshold equals the top score and the tied block is rejected along with
everything below it — retrieval returns nothing and the answer comes from the public
prior alone.

That produced an AUC of 0.26 at ``n_copies=5`` in ``runs/mia_main``: below chance, which
reads as duplication *protecting* the corpus. The signal had simply been deleted. Since
duplicated corpora are the object of study here, this path has to be exercised by a test
rather than trusted.

No model is loaded: only the threshold arithmetic and the upstream selection rule are
exercised.
"""

from __future__ import annotations

import numpy as np
import pytest

from dp_rag_eval.pipelines.grislain_pipeline import _TieSafePUPVectorStore

TOP_K = 5


class _Bare:
    """Just enough of the store to call the threshold helper."""

    def __init__(self, top_k: int | None = TOP_K) -> None:
        self.top_k = top_k

    threshold = _TieSafePUPVectorStore._non_dp_top_k_threshold


def upstream_threshold(scores: np.ndarray, top_k: int = TOP_K) -> float:
    """Verbatim from ``baselines/grislain/pup_vector_store.py:146-149``."""
    return float(np.sort(scores)[-(top_k + 1)])


def n_selected(scores: np.ndarray, threshold: float) -> int:
    """Verbatim selection rule from ``pup_retrieve`` (line 187): strictly greater."""
    return int(np.sum(scores > threshold))


def corpus_with(copies: int, n_docs: int = 1000, top_score: float = 0.9) -> np.ndarray:
    """Scores for a corpus where one document is present ``copies`` times."""
    return np.concatenate([np.full(copies, top_score), np.linspace(0.1, 0.85, n_docs - copies)])


# ---------------------------------------------------------------------------
# The bug this guards against
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("copies", [TOP_K + 1, TOP_K + 2, 11, 40])
def test_upstream_collapses_once_copies_reach_top_k_plus_one(copies: int):
    """Documents the defect. If this ever stops holding, the fix can be dropped."""
    scores = corpus_with(copies)

    assert n_selected(scores, upstream_threshold(scores)) == 0


@pytest.mark.parametrize("copies", [TOP_K + 1, TOP_K + 2, 11, 40])
def test_the_fix_retrieves_the_tied_block(copies: int):
    scores = corpus_with(copies)

    selected = n_selected(scores, _Bare().threshold(scores))

    assert selected == copies, "every copy ties at the top, so all of them are admitted"


def test_the_boundary_is_exactly_top_k_plus_one():
    """Below the boundary upstream is already correct; at it, it breaks."""
    assert n_selected(corpus_with(TOP_K), upstream_threshold(corpus_with(TOP_K))) == TOP_K
    assert n_selected(corpus_with(TOP_K + 1), upstream_threshold(corpus_with(TOP_K + 1))) == 0


# ---------------------------------------------------------------------------
# The fix must not move any result that was not already broken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("copies", [1, 2, 3, TOP_K])
def test_no_op_below_the_boundary(copies: int):
    scores = corpus_with(copies)

    assert n_selected(scores, _Bare().threshold(scores)) == n_selected(scores, upstream_threshold(scores))


def test_no_op_on_distinct_scores():
    """The case every previously-computed cell falls into.

    This is what makes re-running only the collapsed cells sound: cells whose corpus had
    no exact score ties are unaffected, so they stay comparable with cells computed
    before the fix.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        scores = rng.random(1000)
        assert n_selected(scores, _Bare().threshold(scores)) == n_selected(scores, upstream_threshold(scores)) == TOP_K


def test_selects_top_k_exactly_when_scores_are_distinct():
    scores = np.linspace(0.1, 0.9, 50)

    assert n_selected(scores, _Bare().threshold(scores)) == TOP_K


# ---------------------------------------------------------------------------
# Degenerate shapes
# ---------------------------------------------------------------------------


def test_corpus_smaller_than_top_k_returns_everything():
    scores = np.array([0.3, 0.7, 0.5])

    assert n_selected(scores, _Bare().threshold(scores)) == 3


def test_every_document_identical_admits_all_of_them():
    scores = np.full(20, 0.42)

    assert n_selected(scores, _Bare().threshold(scores)) == 20
    assert n_selected(scores, upstream_threshold(scores)) == 0, "upstream drops the whole corpus"


def test_top_k_of_none_is_rejected():
    with pytest.raises(ValueError, match="top_k"):
        _Bare(top_k=None).threshold(np.linspace(0.1, 0.9, 10))
