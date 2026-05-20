"""Protocol definitions for the dp-rag-eval framework."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from dp_rag_eval.core.types import (
    AnswerResult,
    AttackResult,
    Document,
    PrivacyBudget,
    RetrievalResult,
)


@runtime_checkable
class DPRagPipeline(Protocol):
    """Common interface for all DP-RAG implementations."""

    def index(self, documents: list[Document]) -> None:
        """Index documents into the retrieval store."""
        ...

    def retrieve(self, query: str, eps: float, delta: float) -> RetrievalResult:
        """DP-private retrieval only (no generation). Used by attacks."""
        ...

    def answer(self, query: str, eps: float, delta: float) -> AnswerResult:
        """Full DP-RAG pipeline: retrieve + generate."""
        ...

    def get_privacy_spent(self) -> PrivacyBudget:
        """Cumulative privacy budget spent so far."""
        ...

    def reset(self) -> None:
        """Reset privacy accountant for a new experiment run."""
        ...


@runtime_checkable
class Attack(Protocol):
    """Common interface for privacy attacks."""

    def setup(self, pipeline: DPRagPipeline, dataset: list[Document]) -> None:
        """Prepare the attack (e.g., generate canaries, select targets)."""
        ...

    def run(self, n_queries: int, eps: float, delta: float) -> AttackResult:
        """Execute the attack and return metrics."""
        ...


@runtime_checkable
class DatasetLoader(Protocol):
    """Loads a base corpus."""

    def load(self) -> list[Document]:
        """Load and return the document corpus."""
        ...


@runtime_checkable
class Metric(Protocol):
    """Interface for evaluation metrics."""

    def compute(self, predictions: list[str], references: list[list[str]]) -> float:
        """Compute metric value over a batch of predictions and references."""
        ...


@runtime_checkable
class LLMBackend(Protocol):
    """Interface for LLM inference backends."""

    @property
    def tokenizer(self) -> Any:
        """Return the tokenizer associated with this backend."""
        ...

    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        ...

    def generate_single_token(self, prompts: list[str]) -> list[int]:
        """Generate a single next token for each prompt. Returns token IDs."""
        ...

    def generate(self, prompts: list[str], max_new_tokens: int) -> list[str]:
        """Generate text completions for the given prompts."""
        ...
