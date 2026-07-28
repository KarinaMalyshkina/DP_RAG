"""Privacy budget calibration — the machinery that makes epsilon comparable.

Two DP-RAG methods quote epsilon in different units unless something forces them
onto a common one. This package solves each mechanism's parameters so that its own
accountant reports a specified total guarantee for one answered query, and reports
honestly when a mechanism cannot reach the target at all.
"""

from dp_rag_eval.privacy.calibration import (
    calibrate_grislain,
    calibrate_koga,
    calibrate_non_private,
    ld_vote_threshold,
    min_voters_for,
)

__all__ = [
    "calibrate_grislain",
    "calibrate_koga",
    "calibrate_non_private",
    "ld_vote_threshold",
    "min_voters_for",
]
