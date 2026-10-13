"""The DP retrieval threshold must be drawable on a fully duplicated corpus.

Upstream normalises ``exp(eps * u / 2) * width`` directly. When every document is
repeated, the scores are atomic and most intervals have zero width; the weight that is
left sits far from the top-k boundary and underflows float64 to an exact 0. The sum is
then 0, ``pdf /= sum`` gives NaN and ``np.random.choice`` raises "probabilities contain
NaN". Three Gemma 4 cells (``duplication_rate=1.0``, ``n_copies=10``, ``eps=5``) died
that way.

The fix lives in ``baselines/grislain/pup_vector_store.py``, which is gitignored and
restored from ``patches/grislain.patch`` by ``setup_baselines.py``. If the patch is
regenerated without the fix, these tests fail on import rather than a sweep failing
mid-matrix.

No model is loaded: only the threshold arithmetic is exercised.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import dp_rag_eval.wrappers.grislain_imports  # noqa: F401  (imports the upstream module from baselines/grislain)

pup_vector_store = sys.modules["pup_vector_store"]

TOP_K = 5
EPS = 5.0
MIN_SCORE, MAX_SCORE = -0.5, 0.8


def padded(scores: np.ndarray) -> np.ndarray:
    """Upstream's preprocessing: sort, pad with -1 and 1, clip to the score range."""
    s = np.insert(np.sort(scores), 0, -1)
    s = np.insert(s, len(s), 1)
    return np.clip(s, MIN_SCORE, MAX_SCORE)


def utilities(sorted_scores: np.ndarray, top_k: int = TOP_K) -> np.ndarray:
    return -np.abs(len(sorted_scores) - top_k - np.arange(len(sorted_scores)))


def upstream_pdf(sorted_scores: np.ndarray, sorted_utilities: np.ndarray, epsilon: float) -> np.ndarray:
    """Verbatim from upstream ``_exp_mechanism_top_k_threshold`` before the fix."""
    with np.errstate(invalid="ignore"):
        pdf = np.exp(epsilon * sorted_utilities[:-1] / 2) * np.diff(sorted_scores)
        return pdf / np.sum(pdf)


def captured_pdf(monkeypatch, sorted_scores: np.ndarray, sorted_utilities: np.ndarray, epsilon: float) -> np.ndarray:
    """Run the fixed sampler and return the distribution it hands to ``np.random.choice``."""
    seen: dict[str, np.ndarray] = {}

    def fake_choice(values, p):
        seen["p"] = np.asarray(p)
        return values[int(np.argmax(p))]

    monkeypatch.setattr(pup_vector_store.np.random, "choice", fake_choice)
    pup_vector_store._sample_threshold_log_space(sorted_scores, sorted_utilities, epsilon)
    return seen["p"]


def saturated_corpus(n_saturated: int = 500, n_rest: int = 500) -> np.ndarray:
    """Many documents clipped to ``MAX_SCORE``: the top of the ranking is one atom."""
    return np.concatenate([np.full(n_saturated, 0.95), np.linspace(-0.4, 0.7, n_rest)])


def test_upstream_form_produces_nan_on_saturated_corpus():
    """Documents the defect. If this stops holding, the helper is no longer needed."""
    s = padded(saturated_corpus())
    assert np.isnan(upstream_pdf(s, utilities(s), EPS)).all()


def test_fixed_sampler_returns_valid_distribution_on_saturated_corpus(monkeypatch):
    s = padded(saturated_corpus())
    p = captured_pdf(monkeypatch, s, utilities(s), EPS)
    assert np.isfinite(p).all()
    assert p.sum() == pytest.approx(1.0)
    assert (p[np.diff(s) == 0] == 0).all(), "a zero-width interval must stay impossible"


@pytest.mark.parametrize("eps", [0.5, 1.0, 5.0])
def test_fixed_sampler_matches_upstream_where_upstream_is_finite(monkeypatch, eps: float):
    """Same mechanism, only the arithmetic differs: identical distribution when both work."""
    rng = np.random.default_rng(0)
    s = padded(rng.uniform(-0.3, 0.75, 1000))
    reference = upstream_pdf(s, utilities(s), eps)
    assert np.isfinite(reference).all()
    np.testing.assert_allclose(captured_pdf(monkeypatch, s, utilities(s), eps), reference, rtol=1e-9, atol=1e-300)


def test_ten_copies_of_every_document_draws_a_threshold():
    """The shape of the corpus that crashed: every document present ten times."""
    rng = np.random.default_rng(1)
    base = rng.uniform(0.2, 0.95, 1100)
    s = padded(np.repeat(base, 10))
    np.random.seed(0)
    threshold = pup_vector_store._sample_threshold_log_space(s, utilities(s), EPS)
    assert MIN_SCORE <= threshold <= MAX_SCORE


def test_fully_degenerate_corpus_raises_instead_of_guessing():
    s = np.full(50, MAX_SCORE)
    with pytest.raises(ValueError, match="no support"):
        pup_vector_store._sample_threshold_log_space(s, utilities(s), EPS)
