"""Thin import wrapper around the original Koga et al. DPRAG codebase.

Adds baselines/koga/ to sys.path and re-exports the key DP mechanisms,
voting functions, and privacy accounting utilities.

Requires: git clone https://github.com/tacchan7412/DPRAG.git baselines/koga/
"""

import sys
from pathlib import Path

import numpy as np

# The original Koga code uses np.infty, removed in NumPy 2.0.
# Restore it as an alias so the unmodified baseline code works.
if not hasattr(np, "infty"):
    np.infty = np.inf  # type: ignore[attr-defined]

_KOGA_ROOT = Path(__file__).resolve().parents[3] / "baselines" / "koga"

if not _KOGA_ROOT.exists():
    raise ImportError(f"Koga baseline not found at {_KOGA_ROOT}. Run `python setup_baselines.py` to clone it.")

# Add baselines/koga/ to sys.path so that `from utils.dp import ...` works
# (the original repo uses `from utils.X import Y` in its scripts).
if str(_KOGA_ROOT) not in sys.path:
    sys.path.insert(0, str(_KOGA_ROOT))

from utils.dp import (  # noqa: E402
    LDGumbelMechanism,
    ExpMechanism,
    NotFoundLDTop1,
    DPExpenseOverflow,
)
from utils.ensemble import (  # noqa: E402
    majority_vote,
    get_majority_token_vote,
)
from utils.dp_accountant import (  # noqa: E402
    compose_subsampled_LimitedDomain_to_approxDP,
    compose_subsampled_LimitedDomain_and_EM_to_approxDP,
    EMpeeling_to_approxDP,
)

__all__ = [
    "LDGumbelMechanism",
    "ExpMechanism",
    "NotFoundLDTop1",
    "DPExpenseOverflow",
    "majority_vote",
    "get_majority_token_vote",
    "compose_subsampled_LimitedDomain_to_approxDP",
    "compose_subsampled_LimitedDomain_and_EM_to_approxDP",
    "EMpeeling_to_approxDP",
]
