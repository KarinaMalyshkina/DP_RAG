"""Utility measurement: how good the answers are, at a given privacy guarantee.

The privacy side of this framework is meaningless on its own — any system can be
made perfectly private by refusing to answer. Utility is the other half of the
trade-off, and it must be measured on the *same* calibrated configuration the
attacks run against.

:mod:`~dp_rag_eval.evaluation.metrics` scores a single cell.
:mod:`~dp_rag_eval.evaluation.reporter` reads finished cells back and compares them
— duplicated corpora against the zero-duplication control, pooled over seeds.
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
from dp_rag_eval.evaluation.reporter import (
    ANCHOR_VARIANT,
    GROUP_KEYS,
    PAIR_FIELDS,
    POOLED_FIELDS,
    PairedCell,
    PooledStat,
    detect_metrics,
    load_rows,
    pair_against_anchor,
    pool_over_seeds,
    render_markdown,
    student_t_95,
    write_csv,
)

__all__ = [
    "ANCHOR_VARIANT",
    "GROUP_KEYS",
    "METRICS",
    "PAIR_FIELDS",
    "POOLED_FIELDS",
    "PairedCell",
    "PooledStat",
    "aggregate_scores",
    "detect_metrics",
    "exact_match",
    "load_rows",
    "normalize_answer",
    "pair_against_anchor",
    "pool_over_seeds",
    "render_markdown",
    "rouge_l",
    "score_answer",
    "student_t_95",
    "token_f1",
    "write_csv",
]
