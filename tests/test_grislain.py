"""CPU tests for Grislain DP-RAG components (from original dp-rag repo)."""

import torch
import pytest


@pytest.fixture
def grislain(grislain_available):
    """Import Grislain wrapper (skips if baseline not cloned)."""
    from dp_rag_eval.wrappers import grislain_imports
    return grislain_imports


class TestDPGenerationConfig:
    """Tests for DPGenerationConfig and token_epsilon computation."""

    def test_token_epsilon_positive_and_less_than_total(self, grislain):
        """token_epsilon should be > 0 and < total epsilon."""
        config = grislain.DPGenerationConfig(
            epsilon=2.0, delta=1e-3, max_new_tokens=50, temperature=1.0
        )
        te = config.token_epsilon()
        assert te > 0
        assert te < config.epsilon

    def test_composed_epsilon_close_to_target(self, grislain):
        """Composing token_epsilon over max_new_tokens should approximate the target epsilon."""
        config = grislain.DPGenerationConfig(
            epsilon=2.0, delta=1e-3, max_new_tokens=50, temperature=1.0
        )
        te = config.token_epsilon()
        composed = config.composed_epsilon(te)
        # Should be within tolerance of bisection (default tol=1e-3)
        assert abs(composed - config.epsilon) < 0.1

    def test_higher_epsilon_gives_higher_token_epsilon(self, grislain):
        """A larger total epsilon budget should yield a larger per-token epsilon."""
        config_low = grislain.DPGenerationConfig(epsilon=1.0, delta=1e-3, max_new_tokens=50)
        config_high = grislain.DPGenerationConfig(epsilon=4.0, delta=1e-3, max_new_tokens=50)
        assert config_high.token_epsilon() > config_low.token_epsilon()


class TestDPLogitsAggregator:
    """Tests for the DPLogitsAggregator LogitsProcessor."""

    def test_output_shape(self, grislain):
        """Aggregator should reduce batch dimension to 1."""
        config = grislain.DPGenerationConfig(
            epsilon=2.0, delta=1e-3, max_new_tokens=10, temperature=1.0
        )
        aggregator = grislain.DPLogitsAggregator(config)
        # batch_size=4: row 0 is public prior, rows 1-3 are document-conditioned
        scores = torch.randn(4, 1000)
        input_ids = torch.zeros(4, 5, dtype=torch.long)
        result = aggregator(input_ids, scores)
        assert result.shape == (1, 1000)

    def test_all_rows_get_same_aggregated_output(self, grislain):
        """The aggregator returns a single row — all prompts collapse into one output."""
        config = grislain.DPGenerationConfig(
            epsilon=2.0, delta=1e-3, max_new_tokens=10, temperature=1.0
        )
        aggregator = grislain.DPLogitsAggregator(config)
        scores = torch.randn(5, 500)
        input_ids = torch.zeros(5, 5, dtype=torch.long)
        result = aggregator(input_ids, scores)
        # Output should have exactly 1 row (aggregated)
        assert result.shape[0] == 1

    def test_non_dp_mode(self, grislain):
        """With differential_pivacy=False, aggregator should still produce valid output."""
        config = grislain.DPGenerationConfig(
            epsilon=2.0, delta=1e-3, max_new_tokens=10,
            temperature=1.0, differential_pivacy=False,
        )
        aggregator = grislain.DPLogitsAggregator(config)
        scores = torch.randn(4, 500)
        input_ids = torch.zeros(4, 5, dtype=torch.long)
        result = aggregator(input_ids, scores)
        assert result.shape == (1, 500)
        assert torch.isfinite(result).all()


class TestPUPVectorStoreConfig:
    """Tests for PUPVectorStoreConfig."""

    def test_default_config(self, grislain):
        """Default config should have sensible defaults."""
        config = grislain.PUPVectorStoreConfig()
        assert config.epsilon == 0.1
        assert config.differential_pivacy is True
        assert config.model_id == "Snowflake/snowflake-arctic-embed-m-v1.5"
