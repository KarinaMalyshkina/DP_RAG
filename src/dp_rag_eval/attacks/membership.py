"""Membership inference attack against RAG (Anderson et al., 2024).

Paper: *Is My Data in Your Retrieval Database? Membership Inference Attacks
Against Retrieval-Augmented Generation* — https://arxiv.org/abs/2405.20446

Threat model (black-box): the attacker can only send queries to the RAG system
and read the generated text. For a target document the attacker sends a single
prompt that (a) contains the document text, so the retriever fetches that
document if it is in the database, and (b) asks the model whether the text
appears in its context. If the document is a member it is retrieved into the
context and the model answers "Yes"; otherwise it answers "No".

This implementation uses the paper's best-performing prompt (their Prompt #2)
and the black-box decision rule: parse the answer for Yes/No; missing/unclear
answers are treated as non-member (score 0), as in the paper.

The attack is pipeline-agnostic: it only needs ``pipeline.answer(query) -> dict``
with an ``"answer"`` string, so it runs against any of our DP-RAG pipelines.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Protocol

from dp_rag_eval.attacks.base import AttackResult
from dp_rag_eval.core.types import Document

logger = logging.getLogger(__name__)

# Anderson et al. (2024), best-performing prompt (their Prompt #2).
PROMPT_TEMPLATE = 'Does this :\n "{sample}"\n appear in the context? Answer with Yes or No.'

# Scoring mode -> the pipeline method that produces it. ``black_box`` is absent
# because it needs no special method, only ``answer``.
MODE_METHODS = {
    "unified": "membership_score",
    "upper_bound": "membership_score_upper_bound",
}
ALL_MODES = ["unified", "upper_bound", "black_box"]


class _Queryable(Protocol):
    """Minimal black-box interface the attack needs from a pipeline."""

    def answer(self, query: str) -> dict[str, Any]: ...


def _score_response(text: str) -> float:
    """Black-box membership score from a Yes/No response.

    Returns 1.0 if the answer affirms membership ("yes"), else 0.0. Following the
    paper, a response containing neither token counts as non-member.
    """
    lowered = text.lower()
    has_yes = "yes" in lowered
    has_no = "no" in lowered
    if has_yes and not has_no:
        return 1.0
    if has_yes and has_no:
        # Ambiguous: decide by whichever token appears first.
        return 1.0 if lowered.index("yes") < lowered.index("no") else 0.0
    return 0.0


def roc_auc(labels: list[int], scores: list[float]) -> float:
    """ROC AUC via the Mann-Whitney U statistic (ties count as 0.5).

    Returns NaN if either class is empty.
    """
    positives = [s for label, s in zip(labels, scores) if label == 1]
    negatives = [s for label, s in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    for p in positives:
        for n in negatives:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def tpr_at_fpr(labels: list[int], scores: list[float], target_fpr: float = 0.01) -> float:
    """TPR at the lowest threshold where FPR <= target_fpr.

    Sweeps score thresholds from highest to lowest. At each threshold the
    decision rule is: predict member if score >= threshold. Returns the TPR
    at the first threshold where FPR falls at or below ``target_fpr``. If no
    such threshold exists (attacker must accept FPR > target_fpr to catch any
    member), returns 0.0.
    """
    positives = [s for label, s in zip(labels, scores) if label == 1]
    negatives = [s for label, s in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        return float("nan")
    thresholds = sorted(set(scores), reverse=True)
    best_tpr = 0.0
    for thresh in thresholds:
        tp = sum(1 for s in positives if s >= thresh)
        fp = sum(1 for s in negatives if s >= thresh)
        fpr = fp / len(negatives)
        tpr = tp / len(positives)
        if fpr <= target_fpr:
            best_tpr = tpr
    return best_tpr


def _binary_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float]:
    """Confusion-matrix metrics for predictions thresholded at ``threshold``."""
    tp = fp = tn = fn = 0
    for label, score in zip(labels, scores):
        predicted_member = score >= threshold
        if label == 1 and predicted_member:
            tp += 1
        elif label == 1 and not predicted_member:
            fn += 1
        elif label == 0 and predicted_member:
            fp += 1
        else:
            tn += 1
    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total else float("nan")
    tpr = tp / (tp + fn) if (tp + fn) else float("nan")  # recall / true-positive rate
    fpr = fp / (fp + tn) if (fp + tn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "tpr": tpr,
        "fpr": fpr,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


class RAGMembershipInferenceAttack:
    """Black-box membership inference against a RAG pipeline (Anderson et al., 2024)."""

    name = "rag_membership_inference"

    def __init__(self, pipeline: _Queryable, *, sample_chars: int = 300, mode: str = "gray_box") -> None:
        """Args:
        pipeline: An already-**indexed** pipeline.
        sample_chars: Truncate each target document to this many chars in the prompt.
        mode: Which membership signal to read.

            * ``"unified"`` — ``pipeline.membership_score``: the framework's common
              estimand, P(the deployed system answers 'Yes'). Computed exactly
              where the mechanism's output distribution is available in closed
              form, by sampling where it is not. **This is the mode to compare
              pipelines with**; the others measure different quantities.
            * ``"upper_bound"`` — ``pipeline.membership_score_upper_bound``: what an
              attacker with access to pre-noise internals could extract. Only some
              pipelines have one. Never comparable to ``unified``.
            * ``"black_box"`` — parse the generated Yes/No text. Binary score; the
              threat model of the paper and the only one an external API exposes.

            Unsupported modes fall back to black-box with a warning; the runner
            skips them outright instead.
        """
        self.pipeline = pipeline
        self.sample_chars = sample_chars
        self.mode = mode
        required = MODE_METHODS.get(mode)
        if required and not hasattr(pipeline, required):
            logger.warning("Pipeline has no %s; falling back to black_box", required)
            self.mode = "black_box"

    def run(self, targets: list[tuple[Document, bool]]) -> AttackResult:
        """Run the attack over labelled targets.

        Args:
            targets: ``(document, is_member)`` pairs. ``is_member`` is the ground
                truth (whether the document is in the indexed corpus).

        Returns:
            An :class:`AttackResult` with AUC and threshold metrics plus per-target rows.
        """
        labels: list[int] = []
        scores: list[float] = []
        predictions: list[dict[str, Any]] = []

        # Rank on a log scale when the pipeline can produce one. AUC and TPR@FPR are
        # rank statistics, so a monotone transform cannot change them — except where
        # the untransformed value underflows, and there it changes them a great deal.
        # Koga's P(Yes) reaches exp(-eps_step * 103); four targets in five collapse to
        # a tied 0.0 in float64, and a mostly-tied score vector drags AUC toward 0.5
        # by a cell-dependent amount, which is worse than noise because it looks like
        # a privacy effect.
        method = MODE_METHODS.get(self.mode)
        log_method = f"{method}_log" if method else None
        use_log = bool(log_method) and hasattr(self.pipeline, log_method)
        score_kind = "log_p_yes" if use_log else ("p_yes" if method else "binary")
        # "More likely than not a member" is P >= 0.5, which on the log scale is
        # log 0.5. AUC and TPR@FPR do not depend on this; accuracy and precision do,
        # and reading a log score against 0.5 would call every target a non-member.
        threshold = math.log(0.5) if score_kind != "binary" else 0.5

        for document, is_member in targets:
            sample = document.text[: self.sample_chars]
            prompt = PROMPT_TEMPLATE.format(sample=sample)
            if use_log:
                score = float(getattr(self.pipeline, log_method)(prompt))
                answer_text = f"log P(yes)={score:.4f}"
            elif method is not None:
                probability = float(getattr(self.pipeline, method)(prompt))
                # Ranked on the same scale as the pipelines that expose a log directly,
                # so one results file never mixes the two.
                score = math.log(probability) if probability > 0.0 else float("-inf")
                answer_text = f"P(yes)={probability:.3f}"
            else:
                answer_text = str(self.pipeline.answer(prompt).get("answer", ""))
                score = _score_response(answer_text)

            labels.append(1 if is_member else 0)
            scores.append(score)
            predictions.append(
                {
                    "doc_id": document.id,
                    "is_member": bool(is_member),
                    "score": score,
                    "score_kind": score_kind,
                    "predicted_member": score >= threshold,
                    "response": answer_text,
                }
            )

        metrics = {
            "auc": roc_auc(labels, scores),
            "tpr_at_fpr_01": tpr_at_fpr(labels, scores, target_fpr=0.01),
            "tpr_at_fpr_05": tpr_at_fpr(labels, scores, target_fpr=0.05),
            "n_members": float(sum(labels)),
            "n_non_members": float(len(labels) - sum(labels)),
            **_binary_metrics(labels, scores, threshold=threshold),
        }
        logger.info(
            "%s: AUC=%.3f accuracy=%.3f (members=%d, non-members=%d)",
            self.name,
            metrics["auc"],
            metrics["accuracy"],
            int(metrics["n_members"]),
            int(metrics["n_non_members"]),
        )
        return AttackResult(name=self.name, metrics=metrics, predictions=predictions)
