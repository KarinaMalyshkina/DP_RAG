"""DP generation components for the Grislain pipeline.

Ported from Grislain (2025) dp_model.py:
- DPGenerationConfig: generation config with DP parameters and per-token epsilon calculation
- DPLogitsAggregator: HuggingFace LogitsProcessor for DP logit aggregation

References:
    Grislain (2025), "RAG with Differential Privacy"
"""

from __future__ import annotations

import logging
from typing import Any

import torch
from transformers import GenerationConfig, LogitsProcessor

logger = logging.getLogger(__name__)


def _compute_token_epsilon(
    target_epsilon: float,
    delta: float,
    max_new_tokens: int,
) -> float:
    """Compute per-token epsilon to achieve target total epsilon under PLD composition.

    Uses bisection search. Under simple composition, token_eps = target_eps / max_new_tokens,
    but PLD composition is tighter so we can use a larger per-token epsilon.

    Args:
        target_epsilon: Total privacy budget (epsilon).
        delta: Total delta.
        max_new_tokens: Number of tokens to generate.

    Returns:
        Per-token epsilon that composes to target_epsilon under PLD.
    """
    try:
        from dp_accounting.pld import privacy_loss_distribution

        def _compose_pld(token_eps: float) -> float:
            """Compose token_eps across max_new_tokens steps and return total epsilon."""
            single = privacy_loss_distribution.from_laplace_mechanism(
                parameter=1.0 / token_eps if token_eps > 0 else 1e10,
                sensitivity=1.0,
            )
            composed = single.self_compose(max_new_tokens)
            return composed.get_epsilon_for_delta(delta)

        # Bisection search
        lo, hi = 0.0, target_epsilon
        for _ in range(64):
            mid = (lo + hi) / 2.0
            if mid <= 0:
                lo = mid
                continue
            composed_eps = _compose_pld(mid)
            if composed_eps <= target_epsilon:
                lo = mid
            else:
                hi = mid
        return lo

    except (ImportError, Exception) as e:
        logger.warning("PLD composition not available (%s), using simple composition", e)
        return target_epsilon / max_new_tokens


class DPGenerationConfig(GenerationConfig):
    """Generation config with differential privacy parameters.

    Extends HuggingFace GenerationConfig with DP-specific parameters for
    Grislain's logit aggregation approach.

    Args:
        alpha: Exponentiation parameter for logits (sharpening).
        omega: Weight for the public prior in aggregation.
        temperature: Sampling temperature (also affects DP sensitivity).
        epsilon: Total privacy budget.
        delta: Privacy delta.
        max_new_tokens: Maximum tokens to generate.
    """

    alpha: float = 4.0
    omega: float = 1.0
    epsilon: float = 1.0
    dp_delta: float = 1e-5

    def __init__(self, **kwargs: Any) -> None:
        # Extract DP-specific params before passing to parent
        self.alpha = kwargs.pop("alpha", 4.0)
        self.omega = kwargs.pop("omega", 1.0)
        self.epsilon = kwargs.pop("epsilon", 1.0)
        self.dp_delta = kwargs.pop("dp_delta", 1e-5)
        # Ensure do_sample=True for the exponential mechanism
        kwargs.setdefault("do_sample", True)
        kwargs.setdefault("temperature", 1.0)
        super().__init__(**kwargs)

    @property
    def token_epsilon(self) -> float:
        """Per-token epsilon computed via PLD composition."""
        return _compute_token_epsilon(
            target_epsilon=self.epsilon,
            delta=self.dp_delta,
            max_new_tokens=self.max_new_tokens or 50,
        )


class DPLogitsAggregator(LogitsProcessor):
    """LogitsProcessor that performs DP aggregation of per-document logits.

    During generation, the model processes k+1 prompts simultaneously:
    - prompts[0]: public prior (no document context)
    - prompts[1:k+1]: one prompt per retrieved document

    This processor:
    1. Separates public logits from document logits
    2. Exponentiates document logits with alpha (sharpening)
    3. Centers and clips the per-document contributions (bounding sensitivity)
    4. Sums clipped contributions + weighted public prior
    5. Returns aggregated logits (sampling with temperature implements the EM)

    Args:
        n_docs: Number of retrieved documents (excluding public prior).
        config: DPGenerationConfig with DP parameters.
    """

    def __init__(self, n_docs: int, config: DPGenerationConfig) -> None:
        self.n_docs = n_docs
        self.alpha = config.alpha
        self.omega = config.omega
        self.temperature = config.temperature if config.temperature else 1.0
        self.token_epsilon = config.token_epsilon
        self._n_prompts = n_docs + 1  # +1 for public prior

    @property
    def clip_bound(self) -> float:
        """Clipping bound for per-document contributions (sensitivity control)."""
        return self.token_epsilon * self.temperature / 2.0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Aggregate logits from public prior and per-document prompts.

        Args:
            input_ids: Input token IDs, shape (n_prompts, seq_len).
            scores: Logits from the model, shape (n_prompts, vocab_size).

        Returns:
            Aggregated logits, shape (n_prompts, vocab_size).
            All rows are set to the same aggregated values.
        """
        if scores.shape[0] < self._n_prompts:
            # Not enough prompts — return as-is (shouldn't happen in normal flow)
            return scores

        # Separate public prior and document logits
        public_logits = scores[0]  # (vocab_size,)
        doc_logits = scores[1 : self._n_prompts]  # (n_docs, vocab_size)

        # Exponentiate with alpha for sharpening
        doc_probs = torch.pow(torch.softmax(doc_logits, dim=-1).clamp(min=1e-10), self.alpha)
        doc_probs = doc_probs / doc_probs.sum(dim=-1, keepdim=True)

        # Center: subtract mean across vocabulary for each document
        centered = doc_probs - doc_probs.mean(dim=-1, keepdim=True)

        # Clip per-document contributions
        clip = self.clip_bound
        if clip > 0:
            centered = centered.clamp(-clip, clip)

        # Sum contributions from all documents
        aggregated = centered.sum(dim=0)  # (vocab_size,)

        # Add weighted public prior
        public_probs = torch.softmax(public_logits, dim=-1)
        public_probs_alpha = torch.pow(public_probs.clamp(min=1e-10), self.alpha)
        public_probs_alpha = public_probs_alpha / public_probs_alpha.sum()
        aggregated = aggregated + self.omega * public_probs_alpha

        # Ensure non-negative and normalize
        aggregated = aggregated.clamp(min=1e-10)
        aggregated = aggregated / aggregated.sum()

        # Convert back to log-space for sampling
        aggregated_logits = torch.log(aggregated)

        # Broadcast to all prompts (they all get the same aggregated result)
        result = aggregated_logits.unsqueeze(0).expand_as(scores).clone()
        return result
