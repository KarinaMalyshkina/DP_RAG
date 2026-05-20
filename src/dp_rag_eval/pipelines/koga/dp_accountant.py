"""RDP-based privacy accounting for the Koga pipeline.

Ported from Koga et al. (2024) utils/dp_accountant.py and DP-OPT utilities.
Uses autodp for RDP composition and conversion to (ε,δ)-DP.
"""

from __future__ import annotations

import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def rdp_to_approxdp(rdp_values: np.ndarray, alphas: np.ndarray, delta: float) -> float:
    """Convert RDP guarantees to (ε,δ)-DP.

    For each alpha, compute ε = rdp - log(delta) / (alpha - 1) and return the minimum.

    Args:
        rdp_values: RDP values at each alpha.
        alphas: RDP orders.
        delta: Target delta.

    Returns:
        The tightest ε achievable at the given delta.
    """
    eps_candidates = rdp_values - (np.log(delta) + np.log(alphas - 1)) / (alphas - 1) + np.log(1 - 1.0 / alphas)
    return float(np.min(eps_candidates))


def compute_rdp_limited_domain(eps_per_token: float, k_bar: int, alpha: float) -> float:
    """Compute RDP for the Limited Domain mechanism at a single order.

    Based on Theorem 3 from the Limited Domain Gumbel mechanism paper.

    Args:
        eps_per_token: Per-token privacy parameter.
        k_bar: Upper bound on domain size (vocabulary size cap).
        alpha: RDP order.

    Returns:
        RDP value at the given order.
    """
    if alpha == 1:
        return eps_per_token * (math.exp(eps_per_token) - 1) / (math.exp(eps_per_token) + k_bar - 1)
    log_terms = []
    for j in range(int(alpha) + 1):
        binom_coeff = math.lgamma(alpha + 1) - math.lgamma(j + 1) - math.lgamma(alpha - j + 1)
        term = binom_coeff + j * eps_per_token + math.log(max(k_bar - 1, 1)) * (alpha - j)
        log_terms.append(term)
    log_terms_arr = np.array(log_terms, dtype=np.float64)
    max_log = np.max(log_terms_arr)
    log_sum = max_log + np.log(np.sum(np.exp(log_terms_arr - max_log)))
    denominator = alpha * math.log(math.exp(eps_per_token) + k_bar - 1)
    return float((log_sum - denominator) / (alpha - 1))


def compose_limited_domain_to_approxdp(
    eps_per_token: float,
    k_bar: int,
    n_tokens: int,
    delta: float,
    n_alphas: int = 100,
) -> float:
    """Compose n_tokens applications of the Limited Domain mechanism and convert to (ε,δ)-DP.

    Args:
        eps_per_token: Per-token privacy parameter.
        k_bar: Vocabulary size cap for the Limited Domain mechanism.
        n_tokens: Number of tokens generated (composition steps).
        delta: Target delta for approximate DP.
        n_alphas: Number of RDP orders to evaluate.

    Returns:
        Total ε under (ε,δ)-DP after composition.
    """
    alphas = np.arange(2, n_alphas + 2, dtype=np.float64)
    rdp_values = np.array([n_tokens * compute_rdp_limited_domain(eps_per_token, k_bar, a) for a in alphas])
    return rdp_to_approxdp(rdp_values, alphas, delta)


def exponential_mechanism_rdp(eps: float, sensitivity: float, alpha: float) -> float:
    """Compute RDP for the exponential mechanism.

    Args:
        eps: Privacy parameter of the exponential mechanism.
        sensitivity: Sensitivity of the utility function.
        alpha: RDP order.

    Returns:
        RDP value at the given order.
    """
    effective_eps = eps * sensitivity
    if alpha == 1:
        return effective_eps * (math.exp(effective_eps) - 1)
    return min(effective_eps, alpha * effective_eps * effective_eps / 2)


def compose_em_peeling_to_approxdp(
    eps: float,
    sensitivity: float,
    n_selections: int,
    delta: float,
    n_alphas: int = 100,
) -> float:
    """Compose n_selections applications of the Exponential Mechanism to (ε,δ)-DP.

    Args:
        eps: Per-selection privacy parameter.
        sensitivity: Sensitivity of the utility function.
        n_selections: Number of selections.
        delta: Target delta.
        n_alphas: Number of RDP orders.

    Returns:
        Total ε under (ε,δ)-DP.
    """
    alphas = np.arange(2, n_alphas + 2, dtype=np.float64)
    rdp_values = np.array([n_selections * exponential_mechanism_rdp(eps, sensitivity, a) for a in alphas])
    return rdp_to_approxdp(rdp_values, alphas, delta)
