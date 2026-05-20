"""Dataset variants for controlled duplication experiments.

Provides transformations on a base corpus to create different experimental conditions:
- OriginalVariant: identity (no changes)
- FullDuplicateVariant: exact copies of a fraction of documents
- PartialOverlapVariant: near-duplicates with minor text changes
- CanaryInjectionVariant: inject canary documents with known secrets
"""

from __future__ import annotations

import logging
import random
import re
from typing import Any

from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)


class OriginalVariant:
    """Identity variant — returns the corpus unchanged."""

    def apply(self, documents: list[Document]) -> list[Document]:
        """Return the original documents unchanged."""
        return list(documents)


class FullDuplicateVariant:
    """Creates exact copies of a fraction of documents.

    Simulates the scenario where the same information appears in multiple
    records (e.g., the same diagnosis in multiple hospital records).

    Args:
        duplication_rate: Fraction of documents to duplicate (0.0 to 1.0).
        copies: Number of copies to create per selected document.
        seed: Random seed for selecting which documents to duplicate.
    """

    def __init__(
        self,
        duplication_rate: float = 0.3,
        copies: int = 5,
        seed: int = 42,
    ) -> None:
        self.duplication_rate = duplication_rate
        self.copies = copies
        self.seed = seed

    def apply(self, documents: list[Document]) -> list[Document]:
        """Create exact duplicates of selected documents.

        New documents get IDs like "{original_id}__dup_{n}".
        Metadata includes "is_duplicate": True and "original_id".
        """
        rng = random.Random(self.seed)
        n_to_dup = max(1, int(len(documents) * self.duplication_rate))
        selected = rng.sample(documents, min(n_to_dup, len(documents)))

        result = list(documents)
        for doc in selected:
            for i in range(self.copies):
                dup = Document(
                    id=f"{doc.id}__dup_{i + 1}",
                    text=doc.text,
                    metadata={
                        **doc.metadata,
                        "is_duplicate": True,
                        "original_id": doc.id,
                        "copy_number": i + 1,
                    },
                )
                result.append(dup)

        logger.info(
            "FullDuplicateVariant: %d originals → %d total (added %d copies of %d docs)",
            len(documents),
            len(result),
            len(result) - len(documents),
            len(selected),
        )
        return result


class PartialOverlapVariant:
    """Creates near-duplicates with minor text modifications.

    Simulates partial information overlap: same key facts but different
    phrasing, sentence order, or minor details.

    Args:
        overlap_rate: Fraction of documents to create near-duplicates for.
        copies: Number of near-duplicate copies per document.
        seed: Random seed.
    """

    def __init__(
        self,
        overlap_rate: float = 0.3,
        copies: int = 3,
        seed: int = 42,
    ) -> None:
        self.overlap_rate = overlap_rate
        self.copies = copies
        self.seed = seed

    def apply(self, documents: list[Document]) -> list[Document]:
        """Create near-duplicates with minor text modifications."""
        rng = random.Random(self.seed)
        n_to_modify = max(1, int(len(documents) * self.overlap_rate))
        selected = rng.sample(documents, min(n_to_modify, len(documents)))

        result = list(documents)
        for doc in selected:
            for i in range(self.copies):
                modified_text = self._modify_text(doc.text, rng)
                near_dup = Document(
                    id=f"{doc.id}__overlap_{i + 1}",
                    text=modified_text,
                    metadata={
                        **doc.metadata,
                        "is_near_duplicate": True,
                        "original_id": doc.id,
                        "copy_number": i + 1,
                    },
                )
                result.append(near_dup)

        logger.info(
            "PartialOverlapVariant: %d originals → %d total",
            len(documents),
            len(result),
        )
        return result

    @staticmethod
    def _modify_text(text: str, rng: random.Random) -> str:
        """Apply minor modifications to text while preserving key facts."""
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            rng.shuffle(sentences)

        # Randomly capitalize or add filler words
        modifications = [
            lambda s: s.replace(". ", ". Additionally, "),
            lambda s: s.replace("Patient:", "Record for patient:"),
            lambda s: s.replace("Diagnosis:", "Diagnosed with:"),
            lambda s: s.replace("Treatment plan:", "Prescribed treatment:"),
        ]
        if modifications:
            mod = rng.choice(modifications)
            return mod(" ".join(sentences))
        return " ".join(sentences)


class CanaryInjectionVariant:
    """Injects canary documents with known secrets into the corpus.

    Used for canary extraction attacks: insert documents containing known
    secrets, then query the pipeline to see if the secrets leak.

    Args:
        n_canaries: Number of canary documents to inject.
        canary_format: Template for canary text. Must contain {secret}.
        copies_per_canary: Number of copies of each canary to inject.
        seed: Random seed for generating secrets.
        secret_length: Length of the generated secret strings.
    """

    def __init__(
        self,
        n_canaries: int = 100,
        canary_format: str = "The secret code is {secret}.",
        copies_per_canary: int = 1,
        seed: int = 42,
        secret_length: int = 8,
    ) -> None:
        self.n_canaries = n_canaries
        self.canary_format = canary_format
        self.copies_per_canary = copies_per_canary
        self.seed = seed
        self.secret_length = secret_length

    def apply(self, documents: list[Document]) -> list[Document]:
        """Inject canary documents into the corpus."""
        rng = random.Random(self.seed)
        result = list(documents)
        canaries_added = 0

        for i in range(self.n_canaries):
            secret = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=self.secret_length))
            canary_text = self.canary_format.format(secret=secret)

            for c in range(self.copies_per_canary):
                canary_doc = Document(
                    id=f"canary_{i:04d}_copy_{c}",
                    text=canary_text,
                    metadata={
                        "is_canary": True,
                        "canary_secret": secret,
                        "canary_index": i,
                        "copy_number": c,
                        "source": "canary_injection",
                    },
                )
                result.append(canary_doc)
                canaries_added += 1

        logger.info(
            "CanaryInjectionVariant: injected %d canary documents (%d unique secrets)",
            canaries_added,
            self.n_canaries,
        )
        return result


def create_variant(
    name: str,
    params: dict[str, Any] | None = None,
) -> OriginalVariant | FullDuplicateVariant | PartialOverlapVariant | CanaryInjectionVariant:
    """Factory for creating dataset variants.

    Args:
        name: Variant name.
        params: Variant-specific parameters.

    Returns:
        An initialized variant instance.

    Raises:
        ValueError: If the variant name is not recognized.
    """
    params = params or {}
    if name == "original":
        return OriginalVariant()
    elif name == "full_duplicate":
        return FullDuplicateVariant(**params)
    elif name == "partial_overlap":
        return PartialOverlapVariant(**params)
    elif name == "canary":
        return CanaryInjectionVariant(**params)
    else:
        raise ValueError(
            f"Unknown variant: {name!r}. Expected 'original', 'full_duplicate', 'partial_overlap', or 'canary'."
        )
