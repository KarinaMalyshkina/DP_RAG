"""Thin import wrapper around the original Grislain DP-RAG codebase.

Adds baselines/grislain/ to sys.path and re-exports the key DP generation,
logit aggregation, and vector store classes.

Requires: git clone https://github.com/sarus-tech/dp-rag.git baselines/grislain/
"""

import sys
from pathlib import Path

_GRISLAIN_ROOT = Path(__file__).resolve().parents[3] / "baselines" / "grislain"

if not _GRISLAIN_ROOT.exists():
    raise ImportError(f"Grislain baseline not found at {_GRISLAIN_ROOT}. Run `python setup_baselines.py` to clone it.")

# Add baselines/grislain/ to sys.path so that `from dp_model import ...` works.
if str(_GRISLAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_GRISLAIN_ROOT))

from dp_model import (  # noqa: E402
    DPGenerationConfig,
    DPLogitsAggregator,
    DPModel,
)
from pup_vector_store import (  # noqa: E402
    PUPVectorStore,
    PUPVectorStoreConfig,
)

__all__ = [
    "DPGenerationConfig",
    "DPLogitsAggregator",
    "DPModel",
    "PUPVectorStore",
    "PUPVectorStoreConfig",
]
