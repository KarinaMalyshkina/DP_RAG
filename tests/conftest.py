"""Shared test fixtures for dp-rag-eval."""

from __future__ import annotations

import pytest

from dp_rag_eval.core.types import AnswerResult, Document, PrivacyBudget, RetrievalResult

# --- Sample Documents ---


@pytest.fixture
def sample_documents() -> list[Document]:
    """A small set of generic test documents."""
    return [
        Document(id="doc_001", text="The capital of France is Paris.", metadata={"topic": "geography"}),
        Document(id="doc_002", text="Python is a programming language.", metadata={"topic": "technology"}),
        Document(id="doc_003", text="The Earth orbits the Sun.", metadata={"topic": "science"}),
        Document(id="doc_004", text="Water boils at 100 degrees Celsius.", metadata={"topic": "science"}),
        Document(id="doc_005", text="Shakespeare wrote Hamlet.", metadata={"topic": "literature"}),
    ]


@pytest.fixture
def medical_documents() -> list[Document]:
    """A small set of medical test documents with structured metadata."""
    return [
        Document(
            id="med_001",
            text="Patient: John Smith, Age: 45. Diagnosis: hypertension. Treatment plan: ACE inhibitors.",
            metadata={
                "patient_name": "John Smith",
                "age": 45,
                "disease": "hypertension",
                "treatment": "ACE inhibitors",
            },
        ),
        Document(
            id="med_002",
            text="Patient: Jane Doe, Age: 32. Diagnosis: diabetes type 2. Treatment plan: metformin.",
            metadata={"patient_name": "Jane Doe", "age": 32, "disease": "diabetes_type2", "treatment": "metformin"},
        ),
        Document(
            id="med_003",
            text="Patient: Bob Wilson, Age: 55. Diagnosis: hypertension. Treatment plan: beta-blockers.",
            metadata={"patient_name": "Bob Wilson", "age": 55, "disease": "hypertension", "treatment": "beta-blockers"},
        ),
        Document(
            id="med_004",
            text="Patient: Alice Brown, Age: 28. Diagnosis: asthma. Treatment plan: inhaled corticosteroids.",
            metadata={
                "patient_name": "Alice Brown",
                "age": 28,
                "disease": "asthma",
                "treatment": "inhaled corticosteroids",
            },
        ),
        Document(
            id="med_005",
            text="Patient: Charlie Davis, Age: 60. Diagnosis: hypertension. Treatment plan: lifestyle changes.",
            metadata={
                "patient_name": "Charlie Davis",
                "age": 60,
                "disease": "hypertension",
                "treatment": "lifestyle changes",
            },
        ),
    ]


# --- Mock Pipeline ---


class MockPipeline:
    """A mock DP-RAG pipeline that returns predictable answers for testing.

    Returns the first document's text as the answer, or a default string
    if no documents are indexed.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)
        self._call_count = 0

    def index(self, documents: list[Document]) -> None:
        self._documents = list(documents)

    def retrieve(self, query: str, eps: float, delta: float) -> RetrievalResult:
        n = min(3, len(self._documents))
        docs = self._documents[:n]
        return RetrievalResult(
            doc_ids=[d.id for d in docs],
            scores=[1.0 / (i + 1) for i in range(n)],
            privacy_spent=PrivacyBudget(epsilon=eps * 0.5, delta=delta),
        )

    def answer(self, query: str, eps: float, delta: float) -> AnswerResult:
        self._call_count += 1
        retrieval = self.retrieve(query, eps, delta)
        # Return text from the first document, or a generic answer
        if self._documents:
            text = self._documents[0].text
        else:
            text = "I don't know."

        budget = PrivacyBudget(epsilon=eps, delta=delta)
        self._total_privacy = self._total_privacy + budget

        return AnswerResult(
            text=text,
            retrieved_doc_ids=retrieval.doc_ids,
            privacy_spent=budget,
            exit_status="success",
        )

    def get_privacy_spent(self) -> PrivacyBudget:
        return self._total_privacy

    def reset(self) -> None:
        self._total_privacy = PrivacyBudget(epsilon=0.0, delta=0.0)
        self._call_count = 0


@pytest.fixture
def mock_pipeline() -> MockPipeline:
    """A mock pipeline for testing attacks and evaluation."""
    return MockPipeline()
