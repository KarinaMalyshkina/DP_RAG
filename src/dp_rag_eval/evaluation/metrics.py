"""Evaluation metrics for utility and privacy.

Utility metrics:
- exact_match: Binary match after normalization
- f1_score: Token-level F1 between prediction and reference

Privacy metrics:
- attack_success_rate: Fraction of successful attack queries
- tpr_at_fpr: True Positive Rate at a given False Positive Rate threshold
"""

from __future__ import annotations

import re
import string
from collections import Counter

from dp_rag_eval.core.types import AttackResult

# --- Text Normalization ---


def _normalize_text(text: str) -> str:
    """Normalize text for evaluation: lowercase, strip articles and punctuation."""
    text = text.lower()
    # Remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Collapse whitespace
    text = " ".join(text.split())
    return text.strip()


# --- Utility Metrics ---


def exact_match(prediction: str, gold_answers: list[str]) -> float:
    """Check if the normalized prediction exactly matches any gold answer.

    Args:
        prediction: Model's predicted answer.
        gold_answers: List of acceptable gold answers.

    Returns:
        1.0 if any match, 0.0 otherwise.
    """
    pred_normalized = _normalize_text(prediction)
    for gold in gold_answers:
        if _normalize_text(gold) == pred_normalized:
            return 1.0
    return 0.0


def f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Compute token-level F1 between prediction and best-matching gold answer.

    Args:
        prediction: Model's predicted answer.
        gold_answers: List of acceptable gold answers.

    Returns:
        Maximum F1 score across all gold answers.
    """
    pred_tokens = _normalize_text(prediction).split()
    if not pred_tokens:
        return 0.0

    best_f1 = 0.0
    for gold in gold_answers:
        gold_tokens = _normalize_text(gold).split()
        if not gold_tokens:
            continue

        common = Counter(pred_tokens) & Counter(gold_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            continue

        precision = num_common / len(pred_tokens)
        recall = num_common / len(gold_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        best_f1 = max(best_f1, f1)

    return best_f1


def compute_utility(
    predictions: list[str],
    references: list[list[str]],
    metric_name: str = "f1",
) -> float:
    """Compute a utility metric over a batch.

    Args:
        predictions: List of predicted answers.
        references: List of gold answer lists.
        metric_name: "exact_match" or "f1".

    Returns:
        Average metric value.
    """
    if not predictions:
        return 0.0

    metric_fn = exact_match if metric_name == "exact_match" else f1_score
    scores = [metric_fn(pred, refs) for pred, refs in zip(predictions, references)]
    return sum(scores) / len(scores)


# --- Privacy Metrics ---


def attack_success_rate(result: AttackResult) -> float:
    """Compute attack success rate from an AttackResult.

    Args:
        result: AttackResult from a privacy attack.

    Returns:
        Success rate (fraction of successful queries).
    """
    return result.success_rate


def tpr_at_fpr(result: AttackResult, fpr_threshold: float = 0.01) -> float:
    """Compute True Positive Rate at a given False Positive Rate threshold.

    For membership inference attacks where ground_truth is available.

    Args:
        result: AttackResult with scores and ground_truth in metadata.
        fpr_threshold: FPR threshold (default 0.01 = 1%).

    Returns:
        TPR at the given FPR. Returns 0.0 if not enough data.
    """
    ground_truth = result.metadata.get("ground_truth", [])
    scores = result.scores

    if not ground_truth or not scores or len(ground_truth) != len(scores):
        return result.success_rate  # Fallback to overall success rate

    # Separate positive (member) and negative (non-member) scores
    pos_scores = [s for s, g in zip(scores, ground_truth) if g]
    neg_scores = [s for s, g in zip(scores, ground_truth) if not g]

    if not pos_scores or not neg_scores:
        return 0.0

    # Find threshold at the desired FPR
    neg_sorted = sorted(neg_scores, reverse=True)
    fpr_index = max(0, int(len(neg_sorted) * fpr_threshold) - 1)
    score_threshold = neg_sorted[fpr_index] if fpr_index < len(neg_sorted) else neg_sorted[-1]

    # Compute TPR at this threshold
    tp = sum(1 for s in pos_scores if s > score_threshold)
    tpr = tp / len(pos_scores)
    return tpr
