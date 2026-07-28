"""Tests for privacy budget calibration.

These are the tests that protect the framework's central claim — that a point on
the epsilon axis means the same thing for every pipeline. If calibration drifts,
every comparison in the thesis silently becomes apples-to-oranges, and nothing
else in the test suite would notice.

The Koga cases import the upstream accountant from ``baselines/koga``; they are
skipped when the baseline has not been cloned.
"""

from __future__ import annotations

import math

import pytest

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.privacy import calibrate_grislain, calibrate_non_private, ld_vote_threshold, min_voters_for


def _has_baseline(module: str) -> bool:
    try:
        __import__(module)
    except Exception:
        return False
    return True


requires_koga = pytest.mark.skipif(
    not _has_baseline("dp_rag_eval.wrappers.koga_imports"),
    reason="Koga baseline not cloned (run python setup_baselines.py)",
)
requires_grislain = pytest.mark.skipif(
    not _has_baseline("dp_rag_eval.wrappers.grislain_imports"),
    reason="Grislain baseline not cloned (run python setup_baselines.py)",
)

TOLERANCE = 0.02  # 2% — the bisection targets 0.1%, this leaves room for accountant noise


class TestNonPrivate:
    def test_infinite_budget_is_feasible_and_infinite(self) -> None:
        target = PrivacyTarget(epsilon=float("inf"), delta=1e-6, n_tokens=32)
        result = calibrate_non_private(target)
        assert result.feasible
        assert math.isinf(result.realized.epsilon)


class TestLdThreshold:
    def test_tighter_epsilon_needs_more_votes(self) -> None:
        assert ld_vote_threshold(0.1, 1e-8) > ld_vote_threshold(1.0, 1e-8)

    def test_smaller_delta_needs_more_votes(self) -> None:
        assert ld_vote_threshold(1.0, 1e-10) > ld_vote_threshold(1.0, 1e-4)


@requires_koga
class TestKoga:
    @pytest.mark.parametrize("epsilon", [1.0, 4.0, 16.0])
    def test_realized_epsilon_hits_the_target(self, epsilon: float) -> None:
        from dp_rag_eval.privacy import calibrate_koga

        target = PrivacyTarget(epsilon=epsilon, delta=1e-6, n_tokens=8)
        result = calibrate_koga(target, n_split=1000)
        assert result.realized.epsilon == pytest.approx(epsilon, rel=TOLERANCE)

    def test_realized_delta_is_exactly_the_target(self) -> None:
        """Koga's accountant derives delta; calibration must steer it onto the target."""
        from dp_rag_eval.privacy import calibrate_koga

        target = PrivacyTarget(epsilon=4.0, delta=1e-6, n_tokens=8)
        result = calibrate_koga(target, n_split=1000)
        assert result.realized.delta == 1e-6
        # The two components must sum to the total, which is what makes it exact.
        params = result.params
        assert params["delta1"] + params["dp_delta"] * target.n_tokens == pytest.approx(1e-6)

    def test_step_epsilon_increases_with_target(self) -> None:
        from dp_rag_eval.privacy import calibrate_koga

        loose = calibrate_koga(PrivacyTarget(8.0, 1e-6, 8), n_split=1000)
        tight = calibrate_koga(PrivacyTarget(1.0, 1e-6, 8), n_split=1000)
        assert loose.diagnostics["eps_step"] > tight.diagnostics["eps_step"]
        assert loose.diagnostics["ld_threshold"] < tight.diagnostics["ld_threshold"]

    def test_infeasible_when_voters_cannot_reach_the_threshold(self) -> None:
        """The case that would otherwise masquerade as a successful defence."""
        from dp_rag_eval.privacy import calibrate_koga

        result = calibrate_koga(PrivacyTarget(0.5, 1e-6, 32), n_split=8)
        assert not result.feasible
        assert "votes" in result.reason
        assert result.diagnostics["min_n_split"] > 8

    def test_min_voters_makes_it_feasible(self) -> None:
        from dp_rag_eval.privacy import calibrate_koga

        target = PrivacyTarget(2.0, 1e-6, 8)
        needed = min_voters_for(target)
        assert not calibrate_koga(target, n_split=needed - 2).feasible
        assert calibrate_koga(target, n_split=needed).feasible

    def test_longer_answers_cost_more(self) -> None:
        """Composition over more tokens must leave less budget per selection."""
        from dp_rag_eval.privacy import calibrate_koga

        short = calibrate_koga(PrivacyTarget(4.0, 1e-6, 4), n_split=1000)
        long = calibrate_koga(PrivacyTarget(4.0, 1e-6, 32), n_split=1000)
        assert long.diagnostics["eps_step"] < short.diagnostics["eps_step"]


@requires_grislain
class TestGrislain:
    @pytest.mark.parametrize("epsilon", [1.0, 4.0])
    def test_realized_epsilon_hits_the_target(self, epsilon: float) -> None:
        target = PrivacyTarget(epsilon=epsilon, delta=1e-6, n_tokens=8)
        result = calibrate_grislain(target)
        assert result.feasible
        assert result.realized.epsilon == pytest.approx(epsilon, rel=TOLERANCE)

    def test_calibration_is_not_a_no_op(self) -> None:
        """Handing the target straight to the stages lands off the requested budget.

        Retrieval and generation compose sub-additively, so the solved scale must
        exceed the target for the composition to land on it. The error the naive
        approach makes is not in a fixed direction — it depends on the answer
        length — so the test asserts that calibration is *closer*, not that the
        naive value sits on a particular side.
        """
        target = PrivacyTarget(epsilon=4.0, delta=1e-6, n_tokens=32)
        result = calibrate_grislain(target)
        assert result.diagnostics["budget_scale"] > target.epsilon

        naive_error = abs(result.diagnostics["naive_total_if_uncalibrated"] - target.epsilon)
        calibrated_error = abs(result.realized.epsilon - target.epsilon)
        assert naive_error > 0.01 * target.epsilon, "expected a material error to correct"
        assert calibrated_error < naive_error

    def test_budget_splits_in_the_requested_proportion(self) -> None:
        target = PrivacyTarget(epsilon=4.0, delta=1e-6, n_tokens=8)
        result = calibrate_grislain(target, retrieval_share=0.25)
        total = result.params["retrieval_epsilon"] + result.params["epsilon"]
        assert result.params["retrieval_epsilon"] / total == pytest.approx(0.25)
