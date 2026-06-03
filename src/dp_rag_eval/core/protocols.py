"""Structural protocols for pluggable framework components.

Only the dataset-side protocols are defined here for now. Pipeline / attack
protocols are added as those subsystems land.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from dp_rag_eval.core.types import Document


@runtime_checkable
class DatasetLoader(Protocol):
    """Loads a base corpus. Dataset variants are applied on top of the result."""

    def load(self) -> list[Document]:
        """Return the base corpus as a list of documents with unique ids."""
        ...


@runtime_checkable
class DatasetVariant(Protocol):
    """Transforms a base corpus into one controlled evaluation setting.

    A variant takes a base ``list[Document]`` and returns a new list, without
    mutating the input. Implementations must be deterministic given their seed so
    that experiments are reproducible.
    """

    name: str

    def apply(self, documents: list[Document]) -> list[Document]:
        """Return a new corpus for this variant. Must not mutate ``documents``."""
        ...
