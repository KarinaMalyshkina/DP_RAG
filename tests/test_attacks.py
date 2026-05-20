"""Tests for privacy attacks."""

from __future__ import annotations

import pytest
from tests.conftest import MockPipeline

from dp_rag_eval.attacks.canary import CanaryExtractionAttack
from dp_rag_eval.attacks.membership import MembershipInferenceAttack
from dp_rag_eval.attacks.property import PropertyInferenceAttack
from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.variants import CanaryInjectionVariant


class TestCanaryExtractionAttack:
    """Tests for the canary extraction attack."""

    def test_setup_finds_canaries(self, sample_documents: list[Document]) -> None:
        variant = CanaryInjectionVariant(n_canaries=5, seed=42)
        docs_with_canaries = variant.apply(sample_documents)

        pipeline = MockPipeline()
        pipeline.index(docs_with_canaries)

        attack = CanaryExtractionAttack()
        attack.setup(pipeline, docs_with_canaries)
        assert len(attack._canary_secrets) == 5

    def test_run_returns_valid_result(self, sample_documents: list[Document]) -> None:
        variant = CanaryInjectionVariant(n_canaries=3, seed=42)
        docs_with_canaries = variant.apply(sample_documents)

        pipeline = MockPipeline()
        pipeline.index(docs_with_canaries)

        attack = CanaryExtractionAttack()
        attack.setup(pipeline, docs_with_canaries)
        result = attack.run(n_queries=3, eps=1.0, delta=1e-5)

        assert result.attack_type == "canary_extraction"
        assert result.n_queries == 3
        assert 0.0 <= result.success_rate <= 1.0
        assert len(result.predictions) == 3
        assert len(result.scores) == 3

    def test_run_without_setup_raises(self) -> None:
        attack = CanaryExtractionAttack()
        with pytest.raises(RuntimeError, match="not set up"):
            attack.run(n_queries=1, eps=1.0, delta=1e-5)


class TestMembershipInferenceAttack:
    """Tests for the membership inference attack."""

    def test_setup_creates_members_and_non_members(self, sample_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(sample_documents)

        attack = MembershipInferenceAttack(n_members=3, n_non_members=3, seed=42)
        attack.setup(pipeline, sample_documents)
        assert len(attack._member_docs) == 3
        assert len(attack._non_member_docs) == 3

    def test_run_returns_valid_result(self, medical_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(medical_documents)

        attack = MembershipInferenceAttack(n_members=2, n_non_members=2, seed=42)
        attack.setup(pipeline, medical_documents)
        result = attack.run(n_queries=4, eps=1.0, delta=1e-5)

        assert result.attack_type == "membership_inference"
        assert result.n_queries > 0
        assert 0.0 <= result.success_rate <= 1.0
        assert "ground_truth" in result.metadata

    def test_run_without_setup_raises(self) -> None:
        attack = MembershipInferenceAttack()
        with pytest.raises(RuntimeError, match="not set up"):
            attack.run(n_queries=1, eps=1.0, delta=1e-5)


class TestPropertyInferenceAttack:
    """Tests for the property inference attack."""

    def test_setup_extracts_distribution(self, medical_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(medical_documents)

        attack = PropertyInferenceAttack(property_key="disease")
        attack.setup(pipeline, medical_documents)

        # Hypertension appears 3 times, most common
        assert attack._true_distribution["hypertension"] == 3
        assert attack._property_values[0] == "hypertension"

    def test_run_returns_valid_result(self, medical_documents: list[Document]) -> None:
        pipeline = MockPipeline()
        pipeline.index(medical_documents)

        attack = PropertyInferenceAttack(property_key="disease")
        attack.setup(pipeline, medical_documents)
        result = attack.run(n_queries=5, eps=1.0, delta=1e-5)

        assert result.attack_type == "property_inference"
        assert result.n_queries > 0
        assert "true_top_property" in result.metadata

    def test_run_without_setup_raises(self) -> None:
        attack = PropertyInferenceAttack()
        with pytest.raises(RuntimeError, match="not set up"):
            attack.run(n_queries=1, eps=1.0, delta=1e-5)
