"""Canary extraction attack.

Inserts canary documents with known secrets into the corpus, then queries
the pipeline to determine if the secrets can be extracted from responses.

Based on Carlini et al. (2019), "The Secret Sharer."
"""

from __future__ import annotations

import logging

from dp_rag_eval.attacks.base import BaseAttack
from dp_rag_eval.core.types import AttackResult, Document

logger = logging.getLogger(__name__)


class CanaryExtractionAttack(BaseAttack):
    """Canary extraction attack against DP-RAG pipelines.

    Finds canary documents in the corpus (by metadata), crafts queries designed
    to elicit the canary secrets, and scores responses by substring match.

    Args:
        query_template: Template for extraction queries. Must contain {hint}.
        match_mode: How to check for secret extraction: "substring" or "exact".
    """

    def __init__(
        self,
        query_template: str = "What is the secret code?",
        match_mode: str = "substring",
    ) -> None:
        self._query_template = query_template
        self._match_mode = match_mode
        self._pipeline: object | None = None
        self._canary_secrets: list[str] = []
        self._canary_docs: list[Document] = []

    def setup(self, pipeline: object, dataset: list[Document]) -> None:
        """Find canary documents in the dataset.

        Args:
            pipeline: The DP-RAG pipeline to attack.
            dataset: The corpus (must contain documents with is_canary=True metadata).
        """
        self._pipeline = pipeline
        self._canary_docs = [d for d in dataset if d.metadata.get("is_canary", False)]
        self._canary_secrets = list({d.metadata["canary_secret"] for d in self._canary_docs})
        logger.info(
            "CanaryExtractionAttack: found %d canary docs (%d unique secrets)",
            len(self._canary_docs),
            len(self._canary_secrets),
        )

    def run(self, n_queries: int, eps: float, delta: float) -> AttackResult:
        """Execute canary extraction queries.

        For each unique canary secret, query the pipeline and check if the
        secret appears in the response.

        Args:
            n_queries: Maximum number of queries to issue.
            eps: Privacy budget per query.
            delta: Privacy delta per query.

        Returns:
            AttackResult with extraction success rate.
        """
        if self._pipeline is None:
            raise RuntimeError("Attack not set up. Call setup() first.")

        queries_to_run = min(n_queries, len(self._canary_secrets))
        predictions: list[bool] = []
        scores: list[float] = []

        for secret in self._canary_secrets[:queries_to_run]:
            query = self._query_template
            result = self._pipeline.answer(query, eps, delta)  # type: ignore[union-attr]

            extracted = self._check_extraction(result.text, secret)
            predictions.append(extracted)
            # Score: fraction of the secret that appears in the response
            score = self._compute_extraction_score(result.text, secret)
            scores.append(score)

        n_successes = sum(predictions)
        return AttackResult(
            attack_type="canary_extraction",
            n_queries=queries_to_run,
            n_successes=n_successes,
            success_rate=n_successes / max(queries_to_run, 1),
            predictions=predictions,
            scores=scores,
            metadata={
                "n_canary_secrets": len(self._canary_secrets),
                "match_mode": self._match_mode,
            },
        )

    def _check_extraction(self, response: str, secret: str) -> bool:
        """Check if the secret was extracted in the response."""
        response_lower = response.lower()
        secret_lower = secret.lower()
        if self._match_mode == "exact":
            return secret_lower == response_lower.strip()
        return secret_lower in response_lower

    @staticmethod
    def _compute_extraction_score(response: str, secret: str) -> float:
        """Compute a soft extraction score based on character overlap."""
        if not secret:
            return 0.0
        response_lower = response.lower()
        secret_lower = secret.lower()
        if secret_lower in response_lower:
            return 1.0
        # Partial match: longest common substring ratio
        max_match = 0
        for i in range(len(secret_lower)):
            for j in range(i + 1, len(secret_lower) + 1):
                if secret_lower[i:j] in response_lower:
                    max_match = max(max_match, j - i)
        return max_match / len(secret_lower)
