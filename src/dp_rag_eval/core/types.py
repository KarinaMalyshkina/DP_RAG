"""Core data types used throughout the dp-rag-eval framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PrivacyBudget:
    """Represents a (ε, δ)-differential privacy budget."""

    epsilon: float
    delta: float

    def __add__(self, other: PrivacyBudget) -> PrivacyBudget:
        """Naive sequential composition: epsilons add, deltas add."""
        return PrivacyBudget(
            epsilon=self.epsilon + other.epsilon,
            delta=self.delta + other.delta,
        )

    def __repr__(self) -> str:
        return f"PrivacyBudget(ε={self.epsilon:.4f}, δ={self.delta:.2e})"


@dataclass
class Document:
    """A document in the retrieval corpus."""

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Query:
    """A query with optional gold answers for evaluation."""

    id: str
    text: str
    gold_answers: list[str] = field(default_factory=list)


@dataclass
class RetrievalResult:
    """Result from DP-private retrieval."""

    doc_ids: list[str]
    scores: list[float]
    privacy_spent: PrivacyBudget


@dataclass
class AnswerResult:
    """Result from a full DP-RAG pipeline (retrieve + generate)."""

    text: str
    retrieved_doc_ids: list[str]
    privacy_spent: PrivacyBudget
    exit_status: str  # "success" | "budget_exhausted" | "ld_failed"


@dataclass
class AttackResult:
    """Result from a privacy attack."""

    attack_type: str
    n_queries: int
    n_successes: int
    success_rate: float
    predictions: list[bool] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
