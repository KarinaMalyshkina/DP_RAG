"""Membership inference attack.

Determines whether specific documents were in the training/retrieval corpus
by analyzing the pipeline's response specificity.

Based on Shokri et al. (2017), "Membership Inference Attacks Against ML Models."
"""

from __future__ import annotations

import logging
import random

from dp_rag_eval.attacks.base import BaseAttack
from dp_rag_eval.core.types import AttackResult, Document

logger = logging.getLogger(__name__)


class MembershipInferenceAttack(BaseAttack):
    """Membership inference attack against DP-RAG pipelines.

    Tests whether the pipeline reveals if a specific document is in its corpus
    by measuring how specifically the pipeline responds to queries about
    member vs. non-member documents.

    Args:
        n_members: Number of member documents to test.
        n_non_members: Number of non-member documents to test.
        seed: Random seed for selection.
        query_template: Template for membership queries. Must contain {text_snippet}.
    """

    def __init__(
        self,
        n_members: int = 200,
        n_non_members: int = 200,
        seed: int = 42,
        query_template: str = "Tell me about: {text_snippet}",
    ) -> None:
        self._n_members = n_members
        self._n_non_members = n_non_members
        self._seed = seed
        self._query_template = query_template
        self._pipeline: object | None = None
        self._member_docs: list[Document] = []
        self._non_member_docs: list[Document] = []

    def setup(self, pipeline: object, dataset: list[Document]) -> None:
        """Select member and non-member documents.

        Members are randomly selected from the dataset.
        Non-members are synthetic documents NOT in the corpus.

        Args:
            pipeline: The DP-RAG pipeline to attack.
            dataset: The corpus used by the pipeline.
        """
        self._pipeline = pipeline
        rng = random.Random(self._seed)

        # Select members from the corpus
        n_members = min(self._n_members, len(dataset))
        self._member_docs = rng.sample(dataset, n_members)

        # Generate non-members: documents not in the corpus
        self._non_member_docs = self._generate_non_members(dataset, rng)

        logger.info(
            "MembershipInferenceAttack: %d members, %d non-members",
            len(self._member_docs),
            len(self._non_member_docs),
        )

    def _generate_non_members(self, dataset: list[Document], rng: random.Random) -> list[Document]:
        """Generate synthetic non-member documents."""
        corpus_texts = {d.text for d in dataset}
        non_members: list[Document] = []

        for i in range(self._n_non_members):
            # Create a plausible but non-existent document
            if dataset:
                base = rng.choice(dataset)
                words = base.text.split()
                rng.shuffle(words)
                fake_text = " ".join(words[: max(len(words) // 2, 5)])
                # Ensure it's not in the corpus
                if fake_text in corpus_texts:
                    fake_text = f"Non-existent record #{i}: " + fake_text
            else:
                fake_text = f"Non-existent document number {i}."

            non_members.append(
                Document(
                    id=f"non_member_{i:05d}",
                    text=fake_text,
                    metadata={"is_member": False},
                )
            )

        return non_members

    def run(self, n_queries: int, eps: float, delta: float) -> AttackResult:
        """Execute membership inference queries.

        Queries the pipeline about member and non-member documents,
        measures response specificity, and classifies membership.

        Args:
            n_queries: Total number of queries (split equally between members/non-members).
            eps: Privacy budget per query.
            delta: Privacy delta per query.

        Returns:
            AttackResult with membership inference metrics.
        """
        if self._pipeline is None:
            raise RuntimeError("Attack not set up. Call setup() first.")

        n_per_class = min(n_queries // 2, len(self._member_docs), len(self._non_member_docs))
        predictions: list[bool] = []
        scores: list[float] = []
        ground_truth: list[bool] = []

        # Query about members
        for doc in self._member_docs[:n_per_class]:
            snippet = doc.text[:100]
            query = self._query_template.format(text_snippet=snippet)
            result = self._pipeline.answer(query, eps, delta)  # type: ignore[union-attr]
            score = self._compute_specificity(result.text, doc.text)
            scores.append(score)
            ground_truth.append(True)

        # Query about non-members
        for doc in self._non_member_docs[:n_per_class]:
            snippet = doc.text[:100]
            query = self._query_template.format(text_snippet=snippet)
            result = self._pipeline.answer(query, eps, delta)  # type: ignore[union-attr]
            score = self._compute_specificity(result.text, doc.text)
            scores.append(score)
            ground_truth.append(False)

        # Classify: high specificity → member
        if scores:
            threshold = sum(scores) / len(scores)  # Simple threshold at mean
            predictions = [s > threshold for s in scores]
        else:
            predictions = []

        n_correct = sum(p == g for p, g in zip(predictions, ground_truth))
        total = len(predictions)

        return AttackResult(
            attack_type="membership_inference",
            n_queries=total,
            n_successes=n_correct,
            success_rate=n_correct / max(total, 1),
            predictions=predictions,
            scores=scores,
            metadata={
                "n_members_tested": n_per_class,
                "n_non_members_tested": n_per_class,
                "ground_truth": ground_truth,
                "threshold": threshold if scores else 0.0,
            },
        )

    @staticmethod
    def _compute_specificity(response: str, reference: str) -> float:
        """Compute how specifically the response relates to the reference document.

        Uses word overlap as a simple proxy for specificity.
        """
        if not response or not reference:
            return 0.0
        response_words = set(response.lower().split())
        reference_words = set(reference.lower().split())
        if not reference_words:
            return 0.0
        overlap = len(response_words & reference_words)
        return overlap / len(reference_words)
