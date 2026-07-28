"""Solve each mechanism's parameters for a common total privacy guarantee.

Why this exists: the two baselines expose different knobs and account for them
differently. Koga's ``dp_eps`` is a per-selection budget composed over generated
tokens with RDP; Grislain's ``epsilon`` is a total that it internally bisects into
per-token budgets under PLD, and it spends a separate budget on retrieval. Sweeping
either knob directly produces an axis that means something different for each
system, so the two cannot be compared.

Here every pipeline is instead *calibrated*: given a target
:class:`~dp_rag_eval.core.types.PrivacyTarget` — a total (epsilon, delta) for one
answer of ``n_tokens`` tokens — we solve that pipeline's parameters so that **its
own accountant** reports the target. Using the method's own accountant matters: the
comparison should be between the published mechanisms, not between our opinions of
how tightly they could be accounted for.

Calibration can fail, and that failure is informative rather than exceptional. See
:func:`calibrate_koga`.
"""

from __future__ import annotations

import logging
import math
from functools import lru_cache

from dp_rag_eval.core.types import CalibrationResult, PrivacyBudget, PrivacyTarget

logger = logging.getLogger(__name__)

# Bisection settings shared by both calibrators. The tolerance is relative: the
# realized epsilon is what gets plotted, so being within 0.1% of the request is
# ample, and both accountants are far less precise than that anyway.
_MAX_ITERATIONS = 80
_REL_TOLERANCE = 1e-3
_EPS_SEARCH_HI = 200.0
_EPS_SEARCH_LO = 1e-8


def _bisect_increasing(f, target: float, lo: float, hi: float) -> float:
    """Return x in [lo, hi] with f(x) ~= target, for monotonically increasing f.

    Both accountants are increasing in the per-step budget, so plain bisection is
    enough; no derivative is available in either case.
    """
    for _ in range(_MAX_ITERATIONS):
        mid = 0.5 * (lo + hi)
        value = f(mid)
        if abs(value - target) <= _REL_TOLERANCE * target:
            return mid
        if value < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate_non_private(target: PrivacyTarget) -> CalibrationResult:
    """The epsilon=infinity reference point: no mechanism, nothing to solve."""
    return CalibrationResult(
        feasible=True,
        realized=PrivacyBudget(epsilon=float("inf"), delta=0.0),
        params={"dp": False},
        diagnostics={"note": "no privacy mechanism (reference point)"},
    )


# ---------------------------------------------------------------------------
# Koga — LD-Gumbel token voting, RDP composition
# ---------------------------------------------------------------------------


def ld_vote_threshold(eps_step: float, delta_step: float) -> float:
    """Votes the Limited-Domain selection needs before it can return any token.

    Mirrors the cutoff computed inside ``LDGumbelMechanism.get_top1``:
    ``1 + log(min(sens, k_bar, dim - k_bar) / delta) / eps`` with the upstream
    default sensitivity of 2 (the ``min`` resolves to 2 for any realistic voter
    count and vocabulary). Compare against the number of voters: no vote histogram
    can exceed ``n_split``, so a threshold above it means the mechanism can never
    select anything.
    """
    return 1.0 + math.log(2.0 / delta_step) / eps_step


def _koga_split_deltas(target: PrivacyTarget, delta1_share: float) -> tuple[float, float]:
    """Split the target delta into the per-step and limited-domain residual parts.

    Koga's accountant does not accept a delta to evaluate at — it *derives* the
    total as ``delta1 + delta_step * subsampling_rate * n_steps``. So rather than
    asking it for epsilon at our delta, we choose the two inputs such that the
    derived total lands exactly on the target.
    """
    delta1 = target.delta * delta1_share
    delta_step = target.delta * (1.0 - delta1_share) / max(target.n_tokens, 1)
    return delta_step, delta1


@lru_cache(maxsize=256)
def _solve_koga_step_epsilon(epsilon: float, delta: float, n_tokens: int, delta1_share: float) -> tuple[float, float]:
    """Bisect the per-selection epsilon that composes to ``epsilon`` over ``n_tokens``.

    Cached because a sweep asks for the same handful of targets hundreds of times
    and each evaluation runs an RDP-to-approxDP conversion.
    """
    from dp_rag_eval.wrappers.koga_imports import compose_subsampled_LimitedDomain_to_approxDP

    target = PrivacyTarget(epsilon=epsilon, delta=delta, n_tokens=n_tokens)
    delta_step, delta1 = _koga_split_deltas(target, delta1_share)

    def composed(eps_step: float) -> float:
        total_eps, _ = compose_subsampled_LimitedDomain_to_approxDP(
            eps_step,
            delta_step,
            None,
            subsampling_rate=1.0,
            niter=n_tokens,
            delta1=delta1,
        )
        return float(total_eps)

    eps_step = _bisect_increasing(composed, epsilon, _EPS_SEARCH_LO, _EPS_SEARCH_HI)
    return eps_step, composed(eps_step)


def calibrate_koga(target: PrivacyTarget, n_split: int, *, delta1_share: float = 0.5) -> CalibrationResult:
    """Solve Koga's per-selection budget for a total guarantee, and check feasibility.

    The infeasible case is the reason this returns a result object rather than a
    number. LD-Gumbel only emits a token once some candidate leads by
    :func:`ld_vote_threshold` votes, and a histogram over ``n_split`` voters cannot
    exceed ``n_split``. Tightening epsilon raises that threshold, so below some
    epsilon the mechanism returns nothing no matter what the data looks like. An
    attack run against it then scores every target identically and AUC lands on
    0.5 — indistinguishable, in the plotted results, from a mechanism that
    genuinely defeated the attack. Detecting it here keeps the two apart.

    Args:
        target: The common guarantee to hit.
        n_split: Number of voters the pipeline will use.
        delta1_share: Fraction of the delta budget assigned to the limited-domain
            residual, the rest going to the per-step noise. The split is largely
            arbitrary; 0.5 keeps both terms comfortably below the total.
    """
    if not target.is_private:
        return calibrate_non_private(target)

    eps_step, realized_eps = _solve_koga_step_epsilon(target.epsilon, target.delta, target.n_tokens, delta1_share)
    delta_step, delta1 = _koga_split_deltas(target, delta1_share)
    threshold = ld_vote_threshold(eps_step, delta_step)
    feasible = n_split > threshold

    reason = ""
    if not feasible:
        reason = (
            f"LD-Gumbel needs a lead of ~{threshold:.0f} votes at eps_step={eps_step:.4f}, "
            f"but only {n_split} voters exist. Every selection would fail, so this cell "
            f"would measure mechanism infeasibility rather than privacy. "
            f"Needs n_split >= {math.ceil(threshold) + 1}."
        )

    return CalibrationResult(
        feasible=feasible,
        realized=PrivacyBudget(epsilon=realized_eps, delta=target.delta),
        params={
            "dp_eps": eps_step,
            "dp_delta": delta_step,
            "delta1": delta1,
            "target_eps": target.epsilon,
            "target_delta": target.delta,
            "n_split": n_split,
            "n_tokens": target.n_tokens,
        },
        reason=reason,
        diagnostics={
            "eps_step": eps_step,
            "ld_threshold": threshold,
            "n_split": n_split,
            "min_n_split": math.ceil(threshold) + 1,
        },
    )


def min_voters_for(target: PrivacyTarget, *, delta1_share: float = 0.5) -> int:
    """Smallest voter count at which Koga can reach ``target``.

    Used by the runner's automatic voter sizing: paying for 200 voters where 20
    suffice is pure cost, and running 20 where 200 are needed produces a cell that
    silently measures nothing.
    """
    if not target.is_private:
        return 1
    eps_step, _ = _solve_koga_step_epsilon(target.epsilon, target.delta, target.n_tokens, delta1_share)
    delta_step, _ = _koga_split_deltas(target, delta1_share)
    return math.ceil(ld_vote_threshold(eps_step, delta_step)) + 1


# ---------------------------------------------------------------------------
# Grislain — exponential-mechanism retrieval + DP logit aggregation, PLD composition
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _grislain_composed_epsilon(scale: float, delta: float, n_tokens: int, retrieval_share: float) -> float:
    """Total epsilon at ``delta`` when each stage gets its share of ``scale``.

    Grislain spends budget twice per query: once choosing a score threshold with
    the exponential mechanism, once generating. Both are pure-epsilon mechanisms
    composed under PLD, and PLD composition is sub-additive, so simply handing each
    stage a slice of the target under-spends it.
    """
    from dp_accounting.pld.common import DifferentialPrivacyParameters
    from dp_accounting.pld.privacy_loss_distribution import from_privacy_parameters

    from dp_rag_eval.wrappers.grislain_imports import DPGenerationConfig

    retrieval_eps = scale * retrieval_share
    generation_eps = scale * (1.0 - retrieval_share)
    # DPGenerationConfig bisects its own per-token budget so that composing
    # n_tokens of them yields `generation_eps` at `delta` — that internal step is
    # upstream logic and is deliberately left untouched.
    gen_config = DPGenerationConfig(
        max_new_tokens=n_tokens,
        epsilon=generation_eps,
        delta=delta,
    )
    retrieval_pld = from_privacy_parameters(DifferentialPrivacyParameters(epsilon=retrieval_eps))
    composed = retrieval_pld.compose(gen_config.privacy_loss_distribution)
    return float(composed.get_epsilon_for_delta(delta))


def calibrate_grislain(
    target: PrivacyTarget,
    *,
    retrieval_share: float = 0.1,
) -> CalibrationResult:
    """Solve Grislain's retrieval and generation budgets for a total guarantee.

    Unlike Koga, this mechanism has no feasibility floor: any epsilon is reachable,
    it just gets noisier. The calibration is still necessary, because the naive
    approach — handing the target epsilon straight to ``DPGenerationConfig`` and
    picking a retrieval epsilon separately — produces a system whose *composed*
    guarantee is not the number on the axis. The error is small and its sign
    depends on the answer length (PLD composition is sub-additive, but the
    generation config's own bisection over-shoots slightly at short lengths), which
    is exactly why it should be solved rather than reasoned about.

    Args:
        target: The common guarantee to hit.
        retrieval_share: Fraction of the budget spent on DP retrieval. Grislain's
            paper spends comparatively little there; 0.1 is the framework default.
    """
    if not target.is_private:
        return calibrate_non_private(target)

    scale, realized = _solve_grislain_scale_cached(target.epsilon, target.delta, target.n_tokens, retrieval_share)
    retrieval_eps = scale * retrieval_share
    generation_eps = scale * (1.0 - retrieval_share)

    return CalibrationResult(
        feasible=True,
        realized=PrivacyBudget(epsilon=realized, delta=target.delta),
        params={
            "epsilon": generation_eps,
            "retrieval_epsilon": retrieval_eps,
            "delta": target.delta,
            "max_new_tokens": target.n_tokens,
        },
        diagnostics={
            "budget_scale": scale,
            "retrieval_share": retrieval_share,
            "naive_total_if_uncalibrated": _grislain_composed_epsilon(
                target.epsilon, target.delta, target.n_tokens, retrieval_share
            ),
        },
    )


@lru_cache(maxsize=256)
def _solve_grislain_scale_cached(
    epsilon: float, delta: float, n_tokens: int, retrieval_share: float
) -> tuple[float, float]:
    """Cached bisection for :func:`calibrate_grislain`.

    Each evaluation constructs a ``DPGenerationConfig``, which itself bisects a
    per-token epsilon over ``n_tokens`` PLD compositions, so an uncached outer
    bisection would cost seconds per cell for a value that repeats across the whole
    sweep.
    """

    def composed(scale: float) -> float:
        return _grislain_composed_epsilon(scale, delta, n_tokens, retrieval_share)

    scale = _bisect_increasing(composed, epsilon, _EPS_SEARCH_LO, _EPS_SEARCH_HI)
    return scale, composed(scale)
