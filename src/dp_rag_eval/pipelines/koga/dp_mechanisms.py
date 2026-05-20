"""DP mechanisms for the Koga pipeline.

Ported from Koga et al. (2024) utils/dp.py:
- LDGumbelMechanism: Limited Domain Gumbel mechanism for token voting
- ExpMechanism: Exponential mechanism for DP selection

References:
    - Koga et al. (2024), "Privacy-Preserving RAG with Differential Privacy"
"""

from __future__ import annotations

import logging
import math

import numpy as np

from dp_rag_eval.pipelines.koga.dp_accountant import compose_limited_domain_to_approxdp

logger = logging.getLogger(__name__)


class NotFoundLDTop1(Exception):
    """Raised when the Limited Domain top-1 selection fails threshold check."""


class DPExpenseOverflow(Exception):
    """Raised when the accumulated DP expense exceeds the target budget."""


class LDGumbelMechanism:
    """Limited Domain Gumbel Mechanism for differentially private top-1 selection.

    Used in Koga's DPVoteRAG for selecting the winning token from the vote histogram.
    Adds Gumbel(0, 1/ε) noise to the vote counts and selects the argmax,
    with a threshold check to ensure a minimum gap.

    Args:
        eps: Per-invocation privacy parameter.
        delta: Per-invocation delta.
        k_bar: Upper bound on domain size (vocabulary cap).
        target_eps: Total privacy budget (epsilon).
        target_delta: Total privacy budget (delta).
        fail_mode: Behavior on threshold failure: "raise" or "zero".
    """

    def __init__(
        self,
        eps: float,
        delta: float,
        k_bar: int,
        target_eps: float,
        target_delta: float,
        fail_mode: str = "zero",
    ) -> None:
        self.eps = eps
        self.delta = delta
        self.k_bar = k_bar
        self.target_eps = target_eps
        self.target_delta = target_delta
        self.fail_mode = fail_mode
        self._n_invocations = 0

    def get_top1(self, counts: np.ndarray, dim: int) -> int:
        """Select the top-1 element from counts using the LD Gumbel mechanism.

        Args:
            counts: Vote histogram of shape (dim,).
            dim: Domain size (should match counts length).

        Returns:
            Index of the selected element.

        Raises:
            NotFoundLDTop1: If threshold check fails and fail_mode="raise".
            DPExpenseOverflow: If privacy budget is exhausted.
        """
        self._n_invocations += 1
        self.check_dp_budget()

        # Add Gumbel noise scaled by 1/eps
        noise_scale = 1.0 / self.eps if self.eps > 0 else 1e10
        noise = np.random.gumbel(loc=0, scale=noise_scale, size=dim)
        noisy_counts = counts[:dim].astype(np.float64) + noise

        # Threshold check: the gap between top-1 and top-2 should be meaningful
        sorted_indices = np.argsort(noisy_counts)[::-1]
        top1_idx = sorted_indices[0]
        top1_val = noisy_counts[top1_idx]
        top2_val = noisy_counts[sorted_indices[1]] if dim > 1 else -np.inf

        # Log-based threshold from the LD mechanism
        threshold = (1.0 / self.eps) * math.log(self.k_bar / self.delta) if self.eps > 0 else np.inf

        if top1_val - top2_val < threshold:
            if self.fail_mode == "raise":
                raise NotFoundLDTop1(f"Gap {top1_val - top2_val:.4f} < threshold {threshold:.4f}")
            return 0  # Return padding/zero token on failure

        return int(top1_idx)

    def get_dp_expense(self) -> float:
        """Compute current (ε,δ)-DP expense from all invocations."""
        if self._n_invocations == 0:
            return 0.0
        return compose_limited_domain_to_approxdp(
            eps_per_token=self.eps,
            k_bar=self.k_bar,
            n_tokens=self._n_invocations,
            delta=self.target_delta,
        )

    def check_dp_budget(self) -> None:
        """Check if accumulated DP expense exceeds the target budget.

        Raises:
            DPExpenseOverflow: If total ε exceeds target_eps.
        """
        current_eps = self.get_dp_expense()
        if current_eps > self.target_eps:
            raise DPExpenseOverflow(
                f"DP budget exhausted: current ε={current_eps:.4f} > target ε={self.target_eps:.4f}"
            )

    @property
    def n_invocations(self) -> int:
        """Number of times the mechanism has been invoked."""
        return self._n_invocations

    def reset(self) -> None:
        """Reset invocation counter."""
        self._n_invocations = 0


class ExpMechanism:
    """Exponential Mechanism for differentially private selection.

    Selects an item from a set with probability proportional to
    exp(ε * utility / (2 * sensitivity)).

    Args:
        eps: Privacy parameter.
        sensitivity: Sensitivity of the utility function (default 1.0).
    """

    def __init__(self, eps: float, sensitivity: float = 1.0) -> None:
        self.eps = eps
        self.sensitivity = sensitivity

    def select(self, utilities: np.ndarray) -> int:
        """Select an item using the exponential mechanism.

        Args:
            utilities: Array of utility scores for each candidate.

        Returns:
            Index of the selected item.
        """
        if len(utilities) == 0:
            raise ValueError("Cannot select from empty utilities array")

        scaled = self.eps * utilities / (2.0 * self.sensitivity)
        # Numerical stability: subtract max before exp
        scaled -= np.max(scaled)
        probs = np.exp(scaled)
        probs /= probs.sum()
        return int(np.random.choice(len(utilities), p=probs))
