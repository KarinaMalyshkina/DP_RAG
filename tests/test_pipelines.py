"""Tests for pipeline implementations."""

from __future__ import annotations

import numpy as np
import pytest
from tests.conftest import MockPipeline

from dp_rag_eval.core.types import Document
from dp_rag_eval.pipelines.koga.dp_mechanisms import (
    ExpMechanism,
    LDGumbelMechanism,
)
from dp_rag_eval.pipelines.koga.ensemble import majority_vote
from dp_rag_eval.pipelines.koga.retriever import KogaRetriever


class TestMockPipeline:
    """Tests for the mock pipeline used in other tests."""

    def test_index_and_answer(self, sample_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(sample_documents)
        result = pipeline.answer("What is the capital of France?", eps=1.0, delta=1e-5)
        assert result.exit_status == "success"
        assert result.text != ""
        assert len(result.retrieved_doc_ids) > 0

    def test_privacy_tracking(self, sample_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(sample_documents)

        # Initial budget should be zero
        budget = pipeline.get_privacy_spent()
        assert budget.epsilon == 0.0
        assert budget.delta == 0.0

        # After one query
        pipeline.answer("test", eps=1.0, delta=1e-5)
        budget = pipeline.get_privacy_spent()
        assert budget.epsilon == 1.0
        assert budget.delta == 1e-5

        # After two queries (naive composition)
        pipeline.answer("test2", eps=2.0, delta=1e-5)
        budget = pipeline.get_privacy_spent()
        assert budget.epsilon == 3.0

    def test_reset(self, sample_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(sample_documents)
        pipeline.answer("test", eps=1.0, delta=1e-5)
        assert pipeline.get_privacy_spent().epsilon > 0

        pipeline.reset()
        assert pipeline.get_privacy_spent().epsilon == 0.0


class TestKogaRetriever:
    """Tests for the Koga simple retriever."""

    def test_index_and_retrieve(self, sample_documents: list[Document]) -> None:
        retriever = KogaRetriever(use_dpr=False)
        retriever.index(sample_documents)
        docs, scores = retriever.retrieve("capital of France", top_k=3)
        assert len(docs) == 3
        assert len(scores) == 3
        # The doc about France should have highest overlap
        assert any("France" in d.text for d in docs)

    def test_retrieve_empty(self) -> None:
        retriever = KogaRetriever(use_dpr=False)
        docs, scores = retriever.retrieve("test query", top_k=5)
        assert docs == []
        assert scores == []


class TestKogaDPMechanisms:
    """Tests for the Koga DP mechanisms."""

    def test_ld_gumbel_basic(self) -> None:
        """High epsilon should approximate argmax."""
        np.random.seed(42)
        engine = LDGumbelMechanism(
            eps=100.0,
            delta=1e-5,
            k_bar=10,
            target_eps=10000.0,
            target_delta=1e-5,
            fail_mode="zero",
        )
        counts = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        result = engine.get_top1(counts, dim=10)
        assert result == 3  # Should pick the clear majority

    def test_ld_gumbel_reset(self) -> None:
        engine = LDGumbelMechanism(
            eps=1.0,
            delta=1e-5,
            k_bar=10,
            target_eps=100.0,
            target_delta=1e-5,
        )
        np.random.seed(42)
        counts = np.ones(10)
        engine.get_top1(counts, dim=10)
        assert engine.n_invocations == 1
        engine.reset()
        assert engine.n_invocations == 0

    def test_exp_mechanism_basic(self) -> None:
        """High epsilon should approximately select the highest utility."""
        np.random.seed(42)
        mech = ExpMechanism(eps=100.0, sensitivity=1.0)
        utilities = np.array([0.0, 0.0, 0.0, 10.0, 0.0])
        # With very high epsilon, should almost always pick index 3
        selections = [mech.select(utilities) for _ in range(100)]
        assert sum(s == 3 for s in selections) > 90

    def test_exp_mechanism_empty(self) -> None:
        mech = ExpMechanism(eps=1.0)
        with pytest.raises(ValueError, match="empty"):
            mech.select(np.array([]))


class TestMajorityVote:
    """Tests for the majority vote function."""

    def test_clear_majority(self) -> None:
        """Token that appears most should win with high epsilon."""
        np.random.seed(42)
        engine = LDGumbelMechanism(
            eps=100.0,
            delta=1e-5,
            k_bar=100,
            target_eps=10000.0,
            target_delta=1e-5,
            fail_mode="zero",
        )
        # 8 voters pick token 5, 2 voters pick token 3
        tokens = [5, 5, 5, 5, 5, 5, 5, 5, 3, 3]
        result = majority_vote(tokens, dim=100, dp_engine=engine, k_bar=100)
        assert result == 5
