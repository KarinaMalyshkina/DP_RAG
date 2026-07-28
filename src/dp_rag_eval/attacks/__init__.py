"""Privacy attacks against DP-RAG pipelines.

* :class:`RAGMembershipInferenceAttack` — black-box corpus membership inference
  (Anderson et al., 2024, https://arxiv.org/abs/2405.20446).
* :class:`CanaryExtractionAttack` — Secret-Sharer canary extraction / exposure
  (Carlini et al., 2019, https://arxiv.org/abs/1802.08232).
"""

from dp_rag_eval.attacks.base import AttackResult
from dp_rag_eval.attacks.canary import CanaryExtractionAttack, exposure
from dp_rag_eval.attacks.membership import RAGMembershipInferenceAttack, roc_auc

__all__ = [
    "AttackResult",
    "RAGMembershipInferenceAttack",
    "CanaryExtractionAttack",
    "exposure",
    "roc_auc",
]
