"""Base class for privacy attacks."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from dp_rag_eval.core.types import AttackResult, Document

logger = logging.getLogger(__name__)


class BaseAttack(ABC):
    """Abstract base class for privacy attacks against DP-RAG pipelines.

    Subclasses must implement setup() and run().
    """

    @abstractmethod
    def setup(self, pipeline: object, dataset: list[Document]) -> None:
        """Prepare the attack.

        Args:
            pipeline: The DP-RAG pipeline to attack (satisfies DPRagPipeline protocol).
            dataset: The document corpus used by the pipeline.
        """
        ...

    @abstractmethod
    def run(self, n_queries: int, eps: float, delta: float) -> AttackResult:
        """Execute the attack.

        Args:
            n_queries: Number of attack queries to issue.
            eps: Privacy budget per query.
            delta: Privacy delta per query.

        Returns:
            AttackResult with success metrics.
        """
        ...
