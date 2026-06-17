"""Privacy attacks against DP-RAG pipelines.

* :class:`RAGMembershipInferenceAttack` — black-box corpus membership inference
  (Anderson et al., 2024, https://arxiv.org/abs/2405.20446).
"""

from dp_rag_eval.attacks.base import AttackResult
from dp_rag_eval.attacks.membership import RAGMembershipInferenceAttack, roc_auc

__all__ = ["AttackResult", "RAGMembershipInferenceAttack", "roc_auc"]
