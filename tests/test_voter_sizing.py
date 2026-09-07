"""Tests for Koga's voter-count sizing policies.

The sizing is the difference between a cell that measures privacy and one that
measures the mechanism refusing to operate, and it fails silently in both
directions: too few voters and every selection is refused while the AUC still
prints; too many and GPU hours are spent for nothing. Neither shows up as an error.
"""

from __future__ import annotations

import math

import pytest

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.privacy import ld_vote_threshold, min_voters_for
from dp_rag_eval.privacy.calibration import _koga_split_deltas, _solve_koga_step_epsilon
from dp_rag_eval.runner.sweep import PipelineParams, koga_voter_count


def target(epsilon: float, n_tokens: int = 32) -> PrivacyTarget:
    return PrivacyTarget(epsilon=epsilon, delta=1e-6, n_tokens=n_tokens)


def test_headroom_one_reproduces_the_bare_minimum() -> None:
    """The default must stay backwards compatible with the runs already reported."""
    params = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_headroom=1.0)
    assert koga_voter_count(target(5.0), params) == min_voters_for(target(5.0))


def test_headroom_scales_the_minimum() -> None:
    params = PipelineParams(n_split=20, auto_n_split_cap=1024, n_split_headroom=2.0)
    minimum = min_voters_for(target(5.0))
    assert koga_voter_count(target(5.0), params) == math.ceil(2.0 * minimum)


def test_headroom_is_still_capped() -> None:
    params = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_headroom=4.0)
    assert koga_voter_count(target(5.0), params) == 128


def test_fixed_count_overrides_sizing_and_cap() -> None:
    params = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_headroom=3.0, n_split_fixed=200)
    for epsilon in (0.5, 5.0, 32.0, float("inf")):
        assert koga_voter_count(target(epsilon), params) == 200


def test_fixed_count_applies_to_the_non_private_reference_too() -> None:
    """Otherwise the eps=inf point of an answer-rate curve sits at a different voter
    count from every other point, and the curve compares two systems rather than one."""
    params = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_fixed=200)
    assert koga_voter_count(target(float("inf")), params) == 200


def test_floor_wins_when_the_target_needs_fewer() -> None:
    params = PipelineParams(n_split=64, auto_n_split_cap=128, n_split_headroom=1.0)
    assert min_voters_for(target(100.0)) < 64
    assert koga_voter_count(target(100.0), params) == 64


@pytest.mark.parametrize("n_split", [105, 200, 512, 1024])
def test_vote_threshold_does_not_depend_on_the_voter_count(n_split: int) -> None:
    """The load-bearing fact behind raising the count: it buys agreement for free.

    If the cutoff moved with the voter count, adding voters would raise the bar it
    has to clear and the whole plan would be self-defeating.
    """
    eps_step, _ = _solve_koga_step_epsilon(5.0, 1e-6, 32, 0.5)
    delta_step, _ = _koga_split_deltas(target(5.0), 0.5)
    assert ld_vote_threshold(eps_step, delta_step) == pytest.approx(103.16, abs=0.01)


def test_required_agreement_falls_as_voters_are_added() -> None:
    """What the headroom actually buys, stated as the quantity that matters."""
    eps_step, _ = _solve_koga_step_epsilon(5.0, 1e-6, 32, 0.5)
    delta_step, _ = _koga_split_deltas(target(5.0), 0.5)
    threshold = ld_vote_threshold(eps_step, delta_step)
    assert threshold / 105 > 0.95  # unanimity
    assert threshold / 200 < 0.55  # a plurality
