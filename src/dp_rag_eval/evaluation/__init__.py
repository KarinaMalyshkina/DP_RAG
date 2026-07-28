"""Utility measurement: how good the answers are, at a given privacy guarantee.

The privacy side of this framework is meaningless on its own — any system can be
made perfectly private by refusing to answer. Utility is the other half of the
trade-off, and it must be measured on the *same* calibrated configuration the
attacks run against.
"""

from dp_rag_eval.evaluation.metrics import (
    METRICS,
    aggregate_scores,
    exact_match,
    normalize_answer,
    rouge_l,
    score_answer,
    token_f1,
)

__all__ = [
    "METRICS",
    "aggregate_scores",
    "exact_match",
    "normalize_answer",
    "rouge_l",
    "score_answer",
    "token_f1",
]
