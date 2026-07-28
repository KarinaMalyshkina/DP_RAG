"""Core data types shared across the dp-rag-eval framework."""

from __future__ import annotations

import math
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


@dataclass(frozen=True)
class PrivacyBudget:
    """An (epsilon, delta) differential privacy guarantee.

    ``epsilon = inf`` denotes no privacy mechanism at all — the reference point the
    non-private baseline occupies. It is deliberately not represented as a very
    large finite number, so that "no DP" and "weak DP" never get confused.
    """

    epsilon: float
    delta: float

    @property
    def is_private(self) -> bool:
        return math.isfinite(self.epsilon)

    def __str__(self) -> str:
        eps = "inf" if not math.isfinite(self.epsilon) else f"{self.epsilon:g}"
        return f"(eps={eps}, delta={self.delta:g})"


@dataclass(frozen=True)
class PrivacyTarget:
    """The privacy guarantee every pipeline is calibrated to before a comparison.

    This is the common x-axis of the whole evaluation. It fixes three things that
    must not differ between systems being compared:

    * ``epsilon`` / ``delta`` — the **total** guarantee for answering one query end
      to end, including every stage the pipeline spends budget on (retrieval as
      well as generation). Not a per-token quantity: mechanisms differ in how many
      internal steps they take, so per-step budgets are not comparable.
    * ``n_tokens`` — the answer length the guarantee covers. Composition cost grows
      with it, so an epsilon quoted without the token count is meaningless.

    The privacy unit is one document of the corpus: the guarantee bounds what the
    system reveals about whether any single document is present.
    """

    epsilon: float
    delta: float
    n_tokens: int

    @property
    def is_private(self) -> bool:
        return math.isfinite(self.epsilon)

    def label(self) -> str:
        """Short, filesystem-safe identifier used in result keys."""
        eps = "inf" if not math.isfinite(self.epsilon) else f"{self.epsilon:g}"
        return f"eps={eps}"


@dataclass
class CalibrationResult:
    """Outcome of solving a pipeline's knobs for a :class:`PrivacyTarget`.

    ``feasible=False`` is a first-class result, not an error. A mechanism can be
    unable to deliver a target at any setting of its parameters — Koga's
    LD-Gumbel selection, for instance, needs more agreeing voters than exist once
    epsilon is small enough. Recording that explicitly keeps it separate from
    "the attack failed", which is what an infeasible cell would otherwise look
    like once the metrics are plotted.

    Attributes:
        feasible: Whether the pipeline can be configured to meet the target.
        realized: What the pipeline's own accountant reports for the chosen
            parameters. Should match the target closely; plot against this, not
            against the requested value.
        params: The solved mechanism parameters, recorded for reproducibility.
        reason: Human-readable explanation when ``feasible`` is False.
        diagnostics: Mechanism-specific numbers worth keeping (e.g. the LD-Gumbel
            vote threshold and the voter count it was compared against).
    """

    feasible: bool
    realized: PrivacyBudget
    params: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "realized_epsilon": self.realized.epsilon,
            "realized_delta": self.realized.delta,
            "params": self.params,
            "reason": self.reason,
            "diagnostics": self.diagnostics,
        }
