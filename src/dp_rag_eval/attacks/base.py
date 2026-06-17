"""Shared types for privacy attacks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AttackResult:
    """Outcome of running an attack.

    Attributes:
        name: Attack identifier.
        metrics: Aggregate scores (e.g. ``auc``, ``accuracy``, ``tpr``, ``fpr``).
        predictions: Per-target records (target id, ground-truth label, score, raw response).
    """

    name: str
    metrics: dict[str, float]
    predictions: list[dict[str, Any]] = field(default_factory=list)
