"""Tests for DP mechanisms and privacy accounting."""

from __future__ import annotations

import numpy as np
import torch

from dp_rag_eval.pipelines.koga.dp_accountant import (
    compose_limited_domain_to_approxdp,
    rdp_to_approxdp,
)
from dp_rag_eval.pipelines.koga.dp_mechanisms import LDGumbelMechanism


class TestDPAccountant:
    """Tests for the RDP-based privacy accountant."""

    def test_rdp_to_approxdp_basic(self) -> None:
        """Basic sanity: converting RDP should give finite epsilon."""
        alphas = np.arange(2.0, 10.0)
        rdp_values = np.ones_like(alphas) * 0.5  # Constant RDP
        eps = rdp_to_approxdp(rdp_values, alphas, delta=1e-5)
        assert eps > 0
        assert np.isfinite(eps)

    def test_composition_increases_epsilon(self) -> None:
        """More tokens should mean more privacy spent."""
        eps1 = compose_limited_domain_to_approxdp(
            eps_per_token=0.1,
            k_bar=100,
            n_tokens=10,
            delta=1e-5,
        )
        eps2 = compose_limited_domain_to_approxdp(
            eps_per_token=0.1,
            k_bar=100,
            n_tokens=100,
            delta=1e-5,
        )
        assert eps2 > eps1

    def test_result_is_positive_and_finite(self) -> None:
        """Composed epsilon should always be positive and finite."""
        eps = compose_limited_domain_to_approxdp(
            eps_per_token=0.5,
            k_bar=50,
            n_tokens=20,
            delta=1e-5,
        )
        assert eps > 0
        assert np.isfinite(eps)


class TestLDGumbelBudget:
    """Tests for the LD Gumbel mechanism budget tracking."""

    def test_budget_tracking(self) -> None:
        """get_dp_expense should increase with invocations."""
        engine = LDGumbelMechanism(
            eps=0.1,
            delta=1e-5,
            k_bar=10,
            target_eps=100.0,
            target_delta=1e-5,
        )
        assert engine.get_dp_expense() == 0.0

        np.random.seed(42)
        counts = np.ones(10)
        engine.get_top1(counts, dim=10)
        eps_after_1 = engine.get_dp_expense()
        assert eps_after_1 > 0

        engine.get_top1(counts, dim=10)
        eps_after_2 = engine.get_dp_expense()
        assert eps_after_2 > eps_after_1


class TestDPLogitsAggregator:
    """Tests for the Grislain DPLogitsAggregator."""

    def test_output_shape(self) -> None:
        """Output shape should match input shape."""
        from dp_rag_eval.pipelines.grislain.dp_model import DPGenerationConfig, DPLogitsAggregator

        config = DPGenerationConfig(
            alpha=4.0,
            omega=1.0,
            temperature=1.0,
            epsilon=1.0,
            dp_delta=1e-5,
            max_new_tokens=10,
        )
        agg = DPLogitsAggregator(n_docs=3, config=config)

        input_ids = torch.zeros(4, 10, dtype=torch.long)  # 4 prompts, 10 tokens
        scores = torch.randn(4, 1000)  # 4 prompts, vocab_size=1000

        result = agg(input_ids, scores)
        assert result.shape == scores.shape

    def test_aggregation_is_uniform_across_prompts(self) -> None:
        """All prompts should receive the same aggregated logits."""
        from dp_rag_eval.pipelines.grislain.dp_model import DPGenerationConfig, DPLogitsAggregator

        config = DPGenerationConfig(
            alpha=4.0,
            omega=1.0,
            temperature=1.0,
            epsilon=1.0,
            dp_delta=1e-5,
            max_new_tokens=10,
        )
        agg = DPLogitsAggregator(n_docs=2, config=config)

        input_ids = torch.zeros(3, 5, dtype=torch.long)
        scores = torch.randn(3, 500)

        result = agg(input_ids, scores)
        # All rows should be identical
        assert torch.allclose(result[0], result[1], atol=1e-6)
        assert torch.allclose(result[0], result[2], atol=1e-6)

    def test_clipping_bounds_sensitivity(self) -> None:
        """Clip bound should be token_epsilon * temperature / 2."""
        from dp_rag_eval.pipelines.grislain.dp_model import DPGenerationConfig, DPLogitsAggregator

        config = DPGenerationConfig(
            alpha=4.0,
            omega=1.0,
            temperature=2.0,
            epsilon=1.0,
            dp_delta=1e-5,
            max_new_tokens=10,
        )
        agg = DPLogitsAggregator(n_docs=1, config=config)
        expected_clip = config.token_epsilon * 2.0 / 2.0
        assert abs(agg.clip_bound - expected_clip) < 1e-6


class TestPUPVectorStore:
    """Tests for the PUP Vector Store."""

    def test_add_and_retrieve(self, sample_documents) -> None:
        """Should be able to add docs and retrieve some."""
        from dp_rag_eval.pipelines.grislain.pup_store import PUPVectorStore, PUPVectorStoreConfig

        config = PUPVectorStoreConfig(top_k=3, epsilon=1.0, device="cpu")
        store = PUPVectorStore(config)
        store.add(sample_documents)

        docs, scores, privacy = store.retrieve("capital of France")
        assert len(docs) <= 3
        assert len(scores) == len(docs)
        assert privacy.epsilon == 1.0

    def test_reset(self, sample_documents) -> None:
        from dp_rag_eval.pipelines.grislain.pup_store import PUPVectorStore, PUPVectorStoreConfig

        config = PUPVectorStoreConfig(top_k=3, device="cpu")
        store = PUPVectorStore(config)
        store.add(sample_documents)
        store.reset()
        docs, scores, privacy = store.retrieve("test")
        assert docs == []
