"""Core data types shared across the dp-rag-eval framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    """A single retrievable unit in a RAG corpus.

    Attributes:
        id: Unique, stable identifier within a corpus.
        text: The document text exposed to the retriever / generator.
        metadata: Free-form provenance and bookkeeping. Conventional keys used by
            this framework include:
              - ``source``: dataset name the document came from (e.g. ``"healthcaremagic"``).
              - ``query``: an associated question, when the source is a QA corpus.
              - ``variant``: which dataset variant produced/owns this document.
              - ``is_duplicate`` / ``duplicate_of``: set by the full-duplicate variant.
              - ``shared_fragment_from`` / ``shared_fragment``: set by the partial-overlap variant.
              - ``is_canary_host`` / ``canary_id`` / ``canary_secret``: set by the canary variant.
    """

    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        return {"id": self.id, "text": self.text, "metadata": self.metadata}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Document:
        """Reconstruct a ``Document`` from :meth:`to_dict` output."""
        return cls(id=data["id"], text=data["text"], metadata=dict(data.get("metadata", {})))

    def copy(self, **overrides: Any) -> Document:
        """Return a deep-ish copy of this document.

        The ``metadata`` dict is shallow-copied so callers can mutate the copy's
        metadata without affecting the original. ``overrides`` may set ``id``,
        ``text``, or ``metadata`` on the new instance.

        Args:
            **overrides: Field values to override on the copy.

        Returns:
            A new ``Document`` independent of this one at the top level.
        """
        return Document(
            id=overrides.get("id", self.id),
            text=overrides.get("text", self.text),
            metadata=overrides.get("metadata", dict(self.metadata)),
        )
