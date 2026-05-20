"""Token-level ensemble voting for the Koga pipeline.

Ported from DP-OPT utils/ensemble.py. Implements:
- majority_vote: histogram + DP top-1 selection
- ensemble_generate_token_ids: token-by-token generation with DP voting
- SVT-based threshold comparison for the sparse variant

References:
    - Koga et al. (2024), "Privacy-Preserving RAG with Differential Privacy"
    - Tang et al. (2024), "DP-OPT"
"""

from __future__ import annotations

import logging

import numpy as np

from dp_rag_eval.core.protocols import LLMBackend
from dp_rag_eval.pipelines.koga.dp_mechanisms import (
    DPExpenseOverflow,
    LDGumbelMechanism,
    NotFoundLDTop1,
)

logger = logging.getLogger(__name__)


def majority_vote(
    tokens: list[int],
    dim: int,
    dp_engine: LDGumbelMechanism,
    k_bar: int,
) -> int:
    """Compute DP majority vote over token predictions.

    Builds a histogram of voted tokens and uses the LD Gumbel mechanism
    to privately select the winner.

    Args:
        tokens: List of token IDs from each voter.
        dim: Vocabulary size (histogram dimension).
        dp_engine: The LD Gumbel mechanism instance.
        k_bar: Upper bound on number of distinct tokens that can appear.

    Returns:
        The selected token ID.
    """
    counts = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        if 0 <= t < dim:
            counts[t] += 1
    effective_dim = min(dim, k_bar)
    return dp_engine.get_top1(counts, effective_dim)


def get_majority_token_vote(
    tokens: list[int],
    dim: int,
    dp_engine: LDGumbelMechanism,
    k_bar: int,
    eos_token_id: int,
) -> tuple[int, str]:
    """DP voting with error handling for budget exhaustion and LD failures.

    Args:
        tokens: Token IDs from each voter.
        dim: Vocabulary size.
        dp_engine: LD Gumbel mechanism.
        k_bar: Domain size cap.
        eos_token_id: EOS token ID for fallback.

    Returns:
        Tuple of (selected_token_id, exit_status).
        exit_status is "success", "budget_exhausted", or "ld_failed".
    """
    try:
        token_id = majority_vote(tokens, dim, dp_engine, k_bar)
        return token_id, "success"
    except DPExpenseOverflow:
        logger.warning("DP budget exhausted during voting")
        return eos_token_id, "budget_exhausted"
    except NotFoundLDTop1:
        logger.debug("LD top-1 not found, returning EOS")
        return eos_token_id, "ld_failed"


def get_majority_token_below_threshold(
    tokens: list[int],
    public_token: int,
    dim: int,
    dp_engine: LDGumbelMechanism,
    k_bar: int,
    threshold: float,
    sparse_eps: float,
    eos_token_id: int,
) -> tuple[int, str, bool]:
    """SVT-based voting: skip DP vote if voters agree with the public LLM.

    Uses the Sparse Vector Technique: if the majority of voters agree with the
    public (no-context) LLM's prediction, use the public token without spending
    privacy budget. Otherwise, fall back to standard DP voting.

    Args:
        tokens: Token IDs from each voter.
        public_token: Token predicted by the public (no-context) LLM.
        dim: Vocabulary size.
        dp_engine: LD Gumbel mechanism.
        k_bar: Domain size cap.
        threshold: Agreement threshold for SVT.
        sparse_eps: Privacy parameter for the SVT query.
        eos_token_id: EOS token ID for fallback.

    Returns:
        Tuple of (token_id, exit_status, used_public).
    """
    counts = np.zeros(dim, dtype=np.float64)
    for t in tokens:
        if 0 <= t < dim:
            counts[t] += 1

    n_voters = len(tokens)
    agreement = counts[public_token] / n_voters if n_voters > 0 else 0.0

    # Add Laplace noise for SVT
    noisy_agreement = agreement + np.random.laplace(0, 1.0 / sparse_eps)
    noisy_threshold = threshold + np.random.laplace(0, 1.0 / sparse_eps)

    if noisy_agreement >= noisy_threshold:
        # Voters agree with public LLM — use public token (no DP cost)
        return public_token, "success", True

    # Fall back to standard DP voting
    token_id, status = get_majority_token_vote(tokens, dim, dp_engine, k_bar, eos_token_id)
    return token_id, status, False


def ensemble_generate_token_ids(
    llm_backend: LLMBackend,
    prompt_list: list[str],
    max_new_tokens: int,
    dp_engine: LDGumbelMechanism,
    k_bar: int,
    below_threshold: bool = False,
    threshold: float = 0.5,
    sparse_eps: float = 1.0,
    public_prompt: str | None = None,
) -> tuple[list[int], str]:
    """Generate tokens one at a time with DP ensemble voting.

    Each step: all voters generate their next token, votes are aggregated
    with the LD Gumbel mechanism, and the winning token is appended.

    Args:
        llm_backend: LLM backend for token generation.
        prompt_list: List of prompts, one per voter.
        max_new_tokens: Maximum tokens to generate.
        dp_engine: LD Gumbel mechanism for private voting.
        k_bar: Vocabulary size cap.
        below_threshold: Whether to use SVT (sparse variant).
        threshold: SVT agreement threshold.
        sparse_eps: SVT privacy parameter.
        public_prompt: Prompt for the public (no-context) LLM (required if below_threshold=True).

    Returns:
        Tuple of (list of generated token IDs, exit_status).
    """
    eos_token_id = llm_backend.tokenizer.eos_token_id or 0
    dim = llm_backend.vocab_size
    generated_tokens: list[int] = []
    exit_status = "success"

    for _step in range(max_new_tokens):
        # Each voter generates their next token
        voter_tokens = llm_backend.generate_single_token(prompt_list)

        if below_threshold and public_prompt is not None:
            # SVT variant: compare with public LLM
            public_tokens = llm_backend.generate_single_token([public_prompt])
            public_token = public_tokens[0]
            token_id, status, used_public = get_majority_token_below_threshold(
                voter_tokens,
                public_token,
                dim,
                dp_engine,
                k_bar,
                threshold,
                sparse_eps,
                eos_token_id,
            )
            if not used_public:
                # Update public prompt too
                pass
        else:
            token_id, status = get_majority_token_vote(
                voter_tokens,
                dim,
                dp_engine,
                k_bar,
                eos_token_id,
            )

        if status != "success":
            exit_status = status
            break

        if token_id == eos_token_id:
            break

        generated_tokens.append(token_id)

        # Append the selected token to each voter's prompt for the next step
        token_str = llm_backend.tokenizer.decode([token_id])
        prompt_list = [p + token_str for p in prompt_list]
        if below_threshold and public_prompt is not None:
            public_prompt = public_prompt + token_str

    return generated_tokens, exit_status
