"""Dataset loaders for dp-rag-eval.

Provides:
- NQLoader: Loads Natural Questions from DPR-format JSON files
- SyntheticMedicalLoader: Generates synthetic medical records using Faker
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)


class NQLoader:
    """Loads Natural Questions from DPR-format biencoder JSON.

    Expects the format from facebook/DPR: a JSON array of entries, each with
    "positive_ctxs" containing passages with "title" and "text" fields.

    Args:
        data_path: Path to the DPR-format JSON file (e.g., biencoder-nq-dev.json).
        max_docs: Maximum number of unique documents to load (None = all).
    """

    def __init__(
        self,
        data_path: str = "data/biencoder-nq-dev.json",
        max_docs: int | None = None,
    ) -> None:
        self._data_path = Path(data_path)
        self._max_docs = max_docs

    def load(self) -> list[Document]:
        """Load unique passages from the NQ dataset.

        Returns:
            List of Document objects with unique IDs based on content hash.
        """
        if not self._data_path.exists():
            logger.warning("NQ data file not found: %s", self._data_path)
            return []

        with open(self._data_path, encoding="utf-8") as f:
            data = json.load(f)

        seen_texts: set[str] = set()
        documents: list[Document] = []

        for entry in data:
            for ctx in entry.get("positive_ctxs", []):
                text = ctx.get("text", "").strip()
                if not text or text in seen_texts:
                    continue
                seen_texts.add(text)

                doc_id = hashlib.md5(text.encode()).hexdigest()[:12]
                title = ctx.get("title", "")
                documents.append(
                    Document(
                        id=f"nq_{doc_id}",
                        text=text,
                        metadata={"title": title, "source": "nq"},
                    )
                )

                if self._max_docs and len(documents) >= self._max_docs:
                    break
            if self._max_docs and len(documents) >= self._max_docs:
                break

        logger.info("Loaded %d unique NQ passages from %s", len(documents), self._data_path)
        return documents


class SyntheticMedicalLoader:
    """Generates synthetic medical records using Faker.

    Creates documents representing patient medical records with structured
    metadata (patient name, disease, treatment). Useful for testing property
    inference attacks and controlled duplication experiments.

    Args:
        n_docs: Number of documents to generate.
        seed: Random seed for reproducibility.
        diseases: List of diseases to sample from.
        treatments: Map from disease to possible treatments.
    """

    DEFAULT_DISEASES = [
        "hypertension",
        "diabetes_type2",
        "asthma",
        "migraine",
        "arthritis",
        "depression",
        "anxiety",
        "pneumonia",
        "bronchitis",
        "influenza",
    ]

    DEFAULT_TREATMENTS: dict[str, list[str]] = {
        "hypertension": ["ACE inhibitors", "beta-blockers", "lifestyle changes"],
        "diabetes_type2": ["metformin", "insulin therapy", "dietary management"],
        "asthma": ["inhaled corticosteroids", "bronchodilators", "allergy management"],
        "migraine": ["triptans", "preventive medications", "lifestyle modifications"],
        "arthritis": ["NSAIDs", "physical therapy", "disease-modifying agents"],
        "depression": ["SSRIs", "cognitive behavioral therapy", "counseling"],
        "anxiety": ["benzodiazepines", "SSRIs", "mindfulness therapy"],
        "pneumonia": ["antibiotics", "antiviral medication", "supportive care"],
        "bronchitis": ["bronchodilators", "cough suppressants", "rest and fluids"],
        "influenza": ["oseltamivir", "supportive care", "rest and hydration"],
    }

    def __init__(
        self,
        n_docs: int = 500,
        seed: int = 42,
        diseases: list[str] | None = None,
        treatments: dict[str, list[str]] | None = None,
    ) -> None:
        self._n_docs = n_docs
        self._seed = seed
        self._diseases = diseases or self.DEFAULT_DISEASES
        self._treatments = treatments or self.DEFAULT_TREATMENTS

    def load(self) -> list[Document]:
        """Generate synthetic medical documents.

        Returns:
            List of Document objects with medical record text and metadata.
        """
        from faker import Faker

        fake = Faker()
        Faker.seed(self._seed)

        import random

        random.seed(self._seed)

        documents: list[Document] = []
        for i in range(self._n_docs):
            name = fake.name()
            age = random.randint(18, 90)
            disease = random.choice(self._diseases)
            treatment_options = self._treatments.get(disease, ["general treatment"])
            treatment = random.choice(treatment_options)

            text = (
                f"Patient: {name}, Age: {age}. "
                f"Diagnosis: {disease.replace('_', ' ')}. "
                f"Treatment plan: {treatment}. "
                f"Follow-up scheduled in {random.randint(1, 12)} weeks."
            )

            documents.append(
                Document(
                    id=f"med_{i:05d}",
                    text=text,
                    metadata={
                        "patient_name": name,
                        "age": age,
                        "disease": disease,
                        "treatment": treatment,
                        "source": "synthetic_medical",
                    },
                )
            )

        logger.info("Generated %d synthetic medical documents", len(documents))
        return documents


def create_loader(
    name: str,
    max_docs: int | None = None,
    **kwargs: Any,
) -> NQLoader | SyntheticMedicalLoader:
    """Factory for creating dataset loaders.

    Args:
        name: Loader name ("nq" or "synthetic_medical").
        max_docs: Maximum documents to load.
        **kwargs: Additional loader-specific arguments.

    Returns:
        An initialized loader instance.

    Raises:
        ValueError: If the loader name is not recognized.
    """
    if name == "nq":
        return NQLoader(max_docs=max_docs, **kwargs)
    elif name == "synthetic_medical":
        return SyntheticMedicalLoader(n_docs=max_docs or 500, **kwargs)
    else:
        raise ValueError(f"Unknown dataset loader: {name!r}. Expected 'nq' or 'synthetic_medical'.")
