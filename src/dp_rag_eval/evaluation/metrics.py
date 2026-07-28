"""Answer-quality metrics for the utility half of the privacy-utility trade-off.

Normalisation and the exact-match / token-F1 pair follow the SQuAD convention
(Rajpurkar et al., 2016), which is what the RAG literature reports.

**Read exact match with care on this corpus.** The reference answers here are
free-form doctor replies of tens of words; no generated answer will reproduce one
verbatim, so EM is ~0 for every system including the non-private baseline. That is
a property of the task, not a finding about DP. It is computed and reported for
completeness because a zero EM across the board is itself evidence that EM is the
wrong instrument here — but **token F1 is the primary metric**, with ROUGE-L as a
sequence-sensitive cross-check.
"""

from __future__ import annotations

import re
import statistics
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace (SQuAD-style)."""
    text = text.lower()
    text = text.translate(_PUNCT_TABLE)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    """1.0 if the normalised strings are identical, else 0.0."""
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    """Bag-of-tokens F1 between prediction and reference after normalisation.

    Multiplicity is respected (``Counter`` intersection), so repeating a correct
    word does not inflate the score. Two empty strings count as a perfect match;
    one empty and one not counts as zero.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    n_same = sum(overlap.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pred_tokens)
    recall = n_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def _lcs_length(a: list[str], b: list[str]) -> int:
    """Length of the longest common subsequence, via the usual DP table.

    Rolling one row keeps this O(len(b)) in memory, which matters because answers
    here can run to a few hundred tokens and this is called once per query per cell.
    """
    if not a or not b:
        return 0
    previous = [0] * (len(b) + 1)
    for token_a in a:
        current = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current
    return previous[-1]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1: longest-common-subsequence overlap, order-sensitive.

    Complements :func:`token_f1`, which ignores word order entirely — a DP
    mechanism that emits the right vocabulary in scrambled order scores well on F1
    and poorly here, and that difference is worth seeing.
    """
    pred_tokens = normalize_answer(prediction).split()
    ref_tokens = normalize_answer(reference).split()
    if not pred_tokens and not ref_tokens:
        return 1.0
    if not pred_tokens or not ref_tokens:
        return 0.0
    lcs = _lcs_length(pred_tokens, ref_tokens)
    if lcs == 0:
        return 0.0
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


METRICS = {
    "exact_match": exact_match,
    "token_f1": token_f1,
    "rouge_l": rouge_l,
}


def score_answer(prediction: str, reference: str) -> dict[str, float]:
    """All metrics for one (prediction, reference) pair."""
    return {name: fn(prediction, reference) for name, fn in METRICS.items()}


def aggregate_scores(per_answer: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across answers, plus the answer count.

    Returns NaN means for an empty list rather than raising: a cell where the DP
    retrieval admitted no documents for any query is a real outcome that should be
    recorded, not an error.
    """
    if not per_answer:
        return {"n_answers": 0.0, **{f"{name}_mean": float("nan") for name in METRICS}}
    result: dict[str, float] = {"n_answers": float(len(per_answer))}
    for name in METRICS:
        values = [row[name] for row in per_answer if name in row]
        result[f"{name}_mean"] = statistics.fmean(values) if values else float("nan")
    return result
