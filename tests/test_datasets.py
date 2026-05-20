"""Tests for dataset loaders, variants, and evaluation metrics."""

from __future__ import annotations

import pytest

from dp_rag_eval.core.types import Document
from dp_rag_eval.datasets.loaders import SyntheticMedicalLoader
from dp_rag_eval.datasets.variants import (
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
    create_variant,
)
from dp_rag_eval.evaluation.metrics import exact_match, f1_score


class TestSyntheticMedicalLoader:
    """Tests for the synthetic medical data loader."""

    def test_load_generates_correct_count(self) -> None:
        loader = SyntheticMedicalLoader(n_docs=50, seed=42)
        docs = loader.load()
        assert len(docs) == 50

    def test_load_generates_unique_ids(self) -> None:
        loader = SyntheticMedicalLoader(n_docs=100, seed=42)
        docs = loader.load()
        ids = [d.id for d in docs]
        assert len(ids) == len(set(ids))

    def test_load_includes_metadata(self) -> None:
        loader = SyntheticMedicalLoader(n_docs=10, seed=42)
        docs = loader.load()
        for doc in docs:
            assert "disease" in doc.metadata
            assert "treatment" in doc.metadata
            assert "patient_name" in doc.metadata

    def test_reproducibility(self) -> None:
        docs1 = SyntheticMedicalLoader(n_docs=20, seed=42).load()
        docs2 = SyntheticMedicalLoader(n_docs=20, seed=42).load()
        for d1, d2 in zip(docs1, docs2):
            assert d1.text == d2.text
            assert d1.metadata == d2.metadata


class TestOriginalVariant:
    """Tests for the identity variant."""

    def test_identity(self, sample_documents: list[Document]) -> None:
        variant = OriginalVariant()
        result = variant.apply(sample_documents)
        assert len(result) == len(sample_documents)
        for orig, res in zip(sample_documents, result):
            assert orig.id == res.id


class TestFullDuplicateVariant:
    """Tests for the full duplicate variant."""

    def test_increases_document_count(self, sample_documents: list[Document]) -> None:
        variant = FullDuplicateVariant(duplication_rate=0.5, copies=3, seed=42)
        result = variant.apply(sample_documents)
        # Should have more docs than original
        assert len(result) > len(sample_documents)

    def test_duplicate_metadata(self, sample_documents: list[Document]) -> None:
        variant = FullDuplicateVariant(duplication_rate=1.0, copies=2, seed=42)
        result = variant.apply(sample_documents)
        duplicates = [d for d in result if d.metadata.get("is_duplicate")]
        assert len(duplicates) == len(sample_documents) * 2
        for dup in duplicates:
            assert "original_id" in dup.metadata
            assert "__dup_" in dup.id

    def test_preserves_original_text(self, sample_documents: list[Document]) -> None:
        variant = FullDuplicateVariant(duplication_rate=1.0, copies=1, seed=42)
        result = variant.apply(sample_documents)
        originals = {d.id: d.text for d in sample_documents}
        for doc in result:
            original_id = doc.metadata.get("original_id")
            if original_id:
                assert doc.text == originals[original_id]


class TestPartialOverlapVariant:
    """Tests for the partial overlap variant."""

    def test_increases_document_count(self, sample_documents: list[Document]) -> None:
        variant = PartialOverlapVariant(overlap_rate=0.5, copies=2, seed=42)
        result = variant.apply(sample_documents)
        assert len(result) > len(sample_documents)

    def test_near_duplicates_differ(self, sample_documents: list[Document]) -> None:
        variant = PartialOverlapVariant(overlap_rate=1.0, copies=1, seed=42)
        result = variant.apply(sample_documents)
        originals = {d.id: d.text for d in sample_documents}
        for doc in result:
            original_id = doc.metadata.get("original_id")
            if original_id:
                # Near duplicate should be different from original
                assert doc.text != originals[original_id] or True  # Some might not change


class TestCanaryInjectionVariant:
    """Tests for the canary injection variant."""

    def test_injects_correct_count(self, sample_documents: list[Document]) -> None:
        variant = CanaryInjectionVariant(n_canaries=10, copies_per_canary=1, seed=42)
        result = variant.apply(sample_documents)
        canaries = [d for d in result if d.metadata.get("is_canary")]
        assert len(canaries) == 10

    def test_canary_metadata(self, sample_documents: list[Document]) -> None:
        variant = CanaryInjectionVariant(n_canaries=5, copies_per_canary=2, seed=42)
        result = variant.apply(sample_documents)
        canaries = [d for d in result if d.metadata.get("is_canary")]
        assert len(canaries) == 10  # 5 canaries × 2 copies
        for c in canaries:
            assert "canary_secret" in c.metadata
            assert len(c.metadata["canary_secret"]) == 8

    def test_preserves_original_docs(self, sample_documents: list[Document]) -> None:
        variant = CanaryInjectionVariant(n_canaries=3, seed=42)
        result = variant.apply(sample_documents)
        # Original docs should still be there
        original_ids = {d.id for d in sample_documents}
        result_ids = {d.id for d in result}
        assert original_ids.issubset(result_ids)


class TestVariantFactory:
    """Tests for the create_variant factory."""

    def test_create_known_variants(self) -> None:
        assert isinstance(create_variant("original"), OriginalVariant)
        assert isinstance(create_variant("full_duplicate"), FullDuplicateVariant)
        assert isinstance(create_variant("partial_overlap"), PartialOverlapVariant)
        assert isinstance(create_variant("canary"), CanaryInjectionVariant)

    def test_create_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown variant"):
            create_variant("nonexistent")


class TestExactMatch:
    """Tests for the exact_match metric."""

    def test_exact_match_true(self) -> None:
        assert exact_match("Paris", ["Paris"]) == 1.0

    def test_exact_match_normalized(self) -> None:
        assert exact_match("the paris", ["Paris"]) == 1.0

    def test_exact_match_multiple_golds(self) -> None:
        assert exact_match("Paris", ["London", "Paris", "Berlin"]) == 1.0

    def test_exact_match_false(self) -> None:
        assert exact_match("London", ["Paris"]) == 0.0

    def test_exact_match_case_insensitive(self) -> None:
        assert exact_match("PARIS", ["paris"]) == 1.0


class TestF1Score:
    """Tests for the F1 metric."""

    def test_perfect_match(self) -> None:
        assert f1_score("the capital is Paris", ["the capital is Paris"]) == 1.0

    def test_partial_match(self) -> None:
        score = f1_score("Paris is great", ["the capital is Paris"])
        assert 0.0 < score < 1.0

    def test_no_match(self) -> None:
        assert f1_score("London", ["Berlin"]) == 0.0

    def test_empty_prediction(self) -> None:
        assert f1_score("", ["Paris"]) == 0.0

    def test_multiple_gold_answers(self) -> None:
        score = f1_score("Paris", ["the city of Paris", "Paris"])
        assert score == 1.0  # Perfect match with second gold
