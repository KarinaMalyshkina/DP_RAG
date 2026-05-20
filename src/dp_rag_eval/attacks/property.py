"""Property inference attack.

Infers aggregate properties of the corpus (e.g., most common diagnosis)
from pipeline responses.

Based on Ateniese et al. (2015), "Hacking Smart Machines with Smarter Ones."
"""

from __future__ import annotations

import logging
from collections import Counter

from dp_rag_eval.attacks.base import BaseAttack
from dp_rag_eval.core.types import AttackResult, Document

logger = logging.getLogger(__name__)


class PropertyInferenceAttack(BaseAttack):
    """Property inference attack against DP-RAG pipelines.

    Attempts to infer aggregate properties of the corpus (e.g., the most
    frequent disease, the distribution of treatments) by querying the pipeline.

    Args:
        property_key: Metadata key to infer (e.g., "disease", "treatment").
        query_templates: List of query templates for probing the property.
        seed: Random seed.
    """

    DEFAULT_QUERY_TEMPLATES = [
        "What is the most common diagnosis in the records?",
        "What diseases are most frequently seen?",
        "What is the primary condition treated?",
        "Summarize the most common medical conditions.",
        "What health conditions appear most often?",
    ]

    def __init__(
        self,
        property_key: str = "disease",
        query_templates: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._property_key = property_key
        self._query_templates = query_templates or self.DEFAULT_QUERY_TEMPLATES
        self._seed = seed
        self._pipeline: object | None = None
        self._true_distribution: Counter[str] = Counter()
        self._property_values: list[str] = []

    def setup(self, pipeline: object, dataset: list[Document]) -> None:
        """Analyze the true property distribution from the corpus metadata.

        Args:
            pipeline: The DP-RAG pipeline to attack.
            dataset: The corpus.
        """
        self._pipeline = pipeline

        # Extract true distribution
        for doc in dataset:
            value = doc.metadata.get(self._property_key)
            if value is not None:
                self._true_distribution[str(value)] += 1

        self._property_values = [v for v, _ in self._true_distribution.most_common()]

        logger.info(
            "PropertyInferenceAttack: property '%s' has %d unique values, most common: %s",
            self._property_key,
            len(self._property_values),
            self._true_distribution.most_common(3),
        )

    def run(self, n_queries: int, eps: float, delta: float) -> AttackResult:
        """Execute property inference queries.

        Queries the pipeline with property-probing questions and checks
        if responses reveal the true property distribution.

        Args:
            n_queries: Number of queries to issue.
            eps: Privacy budget per query.
            delta: Privacy delta per query.

        Returns:
            AttackResult with property inference metrics.
        """
        if self._pipeline is None:
            raise RuntimeError("Attack not set up. Call setup() first.")
        if not self._property_values:
            return AttackResult(
                attack_type="property_inference",
                n_queries=0,
                n_successes=0,
                success_rate=0.0,
                metadata={"error": "No property values found in corpus metadata."},
            )

        # The ground truth: most common property value
        true_top = self._true_distribution.most_common(1)[0][0]

        predictions: list[bool] = []
        scores: list[float] = []
        inferred_counts: Counter[str] = Counter()

        queries_to_run = min(n_queries, len(self._query_templates) * 3)
        for i in range(queries_to_run):
            query = self._query_templates[i % len(self._query_templates)]
            result = self._pipeline.answer(query, eps, delta)  # type: ignore[union-attr]

            # Check which property values appear in the response
            response_lower = result.text.lower()
            for value in self._property_values:
                value_normalized = value.replace("_", " ").lower()
                if value_normalized in response_lower:
                    inferred_counts[value] += 1

            # Score: did the response mention the true top value?
            true_top_normalized = true_top.replace("_", " ").lower()
            mentioned = true_top_normalized in response_lower
            predictions.append(mentioned)
            scores.append(1.0 if mentioned else 0.0)

        # Overall success: was the most frequently inferred value the true top?
        if inferred_counts:
            inferred_top = inferred_counts.most_common(1)[0][0]
            overall_success = inferred_top == true_top
        else:
            overall_success = False

        n_successes = sum(predictions)
        return AttackResult(
            attack_type="property_inference",
            n_queries=queries_to_run,
            n_successes=n_successes,
            success_rate=n_successes / max(queries_to_run, 1),
            predictions=predictions,
            scores=scores,
            metadata={
                "true_top_property": true_top,
                "true_distribution": dict(self._true_distribution.most_common(10)),
                "inferred_counts": dict(inferred_counts.most_common(10)),
                "overall_correct": overall_success,
            },
        )
