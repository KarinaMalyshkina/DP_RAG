"""Tests for the utility metrics."""

from __future__ import annotations

import math

from dp_rag_eval.evaluation import (
    aggregate_scores,
    exact_match,
    normalize_answer,
    rouge_l,
    score_answer,
    token_f1,
)


class TestNormalize:
    def test_strips_case_punctuation_and_articles(self) -> None:
        assert normalize_answer("The Patient's fever, gone!") == "patients fever gone"

    def test_collapses_whitespace(self) -> None:
        assert normalize_answer("  rest   and\n fluids ") == "rest and fluids"


class TestExactMatch:
    def test_identical_after_normalisation(self) -> None:
        assert exact_match("The rest.", "rest") == 1.0

    def test_different(self) -> None:
        assert exact_match("rest", "surgery") == 0.0

    def test_both_empty(self) -> None:
        assert exact_match("", "") == 1.0


class TestTokenF1:
    def test_perfect_overlap(self) -> None:
        assert token_f1("rest and fluids", "rest and fluids") == 1.0

    def test_no_overlap(self) -> None:
        assert token_f1("rest", "surgery") == 0.0

    def test_partial_overlap(self) -> None:
        # prediction has 1 of 2 reference tokens and 1 spurious token:
        # precision 1/2, recall 1/2 -> F1 1/2
        assert token_f1("rest surgery", "rest fluids") == 0.5

    def test_both_empty_is_a_match(self) -> None:
        assert token_f1("", "") == 1.0

    def test_one_empty_is_zero(self) -> None:
        assert token_f1("", "rest") == 0.0
        assert token_f1("rest", "") == 0.0

    def test_repetition_does_not_inflate(self) -> None:
        # Repeating a correct token adds no true positives but costs precision.
        assert token_f1("rest rest rest", "rest") < 1.0


class TestRougeL:
    def test_order_matters(self) -> None:
        """The discriminating property versus token F1: word order."""
        scrambled = "fluids and rest"
        reference = "rest and fluids"
        assert token_f1(scrambled, reference) == 1.0
        assert rouge_l(scrambled, reference) < 1.0

    def test_identical(self) -> None:
        assert rouge_l("rest and fluids", "rest and fluids") == 1.0

    def test_disjoint(self) -> None:
        assert rouge_l("aspirin", "surgery") == 0.0


class TestAggregate:
    def test_means_across_answers(self) -> None:
        rows = [score_answer("rest", "rest"), score_answer("surgery", "rest")]
        agg = aggregate_scores(rows)
        assert agg["n_answers"] == 2.0
        assert agg["exact_match_mean"] == 0.5
        assert agg["token_f1_mean"] == 0.5

    def test_empty_input_reports_nan_not_error(self) -> None:
        """A cell where no answer was produced is a result, not a crash."""
        agg = aggregate_scores([])
        assert agg["n_answers"] == 0.0
        assert math.isnan(agg["token_f1_mean"])
