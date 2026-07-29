"""Read finished sweep results back and compute the comparisons between cells.

Every other module here produces one number for one configuration. That is not the
quantity RQ1 asks about. RQ1 asks how much *worse* the protection gets when the
corpus contains duplicates — a difference between two cells that are identical
except for the corpus variant. This module is where that difference is computed
rather than left to a reader with a spreadsheet.

Two operations, both of which the framework previously described in prose and
performed nowhere:

* **Pairing against the control.** Within one (pipeline, mode, epsilon, seed), the
  ``original`` cell is the zero-duplication anchor. Each duplicated cell is reported
  as ``delta = value - anchor`` and ``ratio = value / anchor``.
* **Pooling over seeds.** Seeds are independent replicates of the whole experiment
  (the member/non-member split is redrawn per seed), so a single seed's delta is a
  draw, not a result. Pooling gives mean, standard deviation and a 95% confidence
  interval across the seed axis.

What this module deliberately does **not** do is choose the presentation: it emits
every duplication axis as rows and takes no view on which one to plot first, or
whether to plot at all.

Cells with a non-``OK`` status are never silently dropped. ``INFEASIBLE`` means the
mechanism could not deliver that budget at any setting, and an anchor that is
missing or infeasible makes every comparison in its group undefined — that is
reported as a warning, because a quietly absent row and a computed zero look
identical in a results table.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import statistics
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

logger = logging.getLogger(__name__)

#: Variant name treated as the zero-duplication control.
ANCHOR_VARIANT = "original"

#: A comparison is only meaningful between cells identical in all of these.
GROUP_KEYS: tuple[str, ...] = ("pipeline", "mode", "epsilon", "seed")

#: Held fixed *in addition* to :data:`GROUP_KEYS` when the canary sweep produced the
#: rows — see :func:`_conditioning_keys`.
CANARY_AXIS = "canary_n_repeats"

#: Columns identifying the corpus condition. Written by
#: :meth:`dp_rag_eval.runner.sweep.VariantCell.columns`, so pairing never has to
#: parse the human-readable ``params`` string.
VARIANT_KEYS: tuple[str, ...] = (
    "variant",
    "params",
    "dup_axis",
    "dup_strength",
    "duplication_rate",
    "n_copies",
    "overlap_rate",
    "n_hosts",
    "n_canaries",
    "canary_n_repeats",
)

#: Metrics the reporter knows how to compare, in the order it auto-detects them.
#: Attack metrics first (higher = more leakage), utility last (higher = better).
KNOWN_METRICS: tuple[str, ...] = (
    "auc",
    "tpr_at_fpr_01",
    "tpr_at_fpr_05",
    "mean_exposure",
    "max_exposure",
    "extraction_rate",
    "token_f1_mean",
    "rouge_l_mean",
)

OK_STATUS = "OK"

# Two-sided 95% Student-t critical values by degrees of freedom. scipy is not a
# dependency, and at the 3 seeds this framework runs the normal approximation is
# badly wrong (1.96 vs 4.303), which would understate every interval by more than
# a factor of two.
_T_95: dict[int, float] = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def student_t_95(df: int) -> float:
    """Two-sided 95% critical value for ``df`` degrees of freedom.

    Exact from a table up to df=20, then the nearest tabulated value, falling back
    to the normal quantile for large samples where the difference is negligible.
    """
    if df < 1:
        return float("nan")
    if df in _T_95:
        return _T_95[df]
    if df > 30:
        return NormalDist().inv_cdf(0.975)
    return _T_95[min((k for k in _T_95 if k >= df), default=30)]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _coerce(value: Any) -> Any:
    """Turn a CSV string back into a number where it plainly is one."""
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return ""
    lowered = text.lower()
    if lowered in {"inf", "infinity", "+inf"}:
        return math.inf
    if lowered == "-inf":
        return -math.inf
    if lowered == "nan":
        return math.nan
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return value


def _read_one(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("results", payload) if isinstance(payload, dict) else payload
        return [dict(r) for r in rows]
    with path.open(newline="", encoding="utf-8") as fh:
        return [{k: _coerce(v) for k, v in row.items()} for row in csv.DictReader(fh)]


def load_rows(sources: Iterable[str | Path]) -> list[dict[str, Any]]:
    """Load result rows from run directories or explicit result files.

    ``results.json`` is preferred over ``results.csv`` in a run directory: it keeps
    numeric types and an infinite epsilon intact, where the CSV has stringified
    everything. Rows are deduplicated on ``cell``, so several output directories can
    be merged — which is what makes an epsilon sweep and a duplication sweep
    comparable even when they were run as separate commands into separate
    directories.

    Args:
        sources: Run directories (containing ``results.json`` / ``results.csv``) or
            paths to those files directly.

    Returns:
        One row per distinct ``cell``, in first-seen order.
    """
    merged: dict[str, dict[str, Any]] = {}
    unkeyed: list[dict[str, Any]] = []
    for source in sources:
        path = Path(source)
        if path.is_dir():
            candidates = [path / "results.json", path / "results.csv"]
            found = next((c for c in candidates if c.exists()), None)
            if found is None:
                logger.warning("No results.json or results.csv in %s — skipped", path)
                continue
            path = found
        elif not path.exists():
            logger.warning("No such results file: %s — skipped", path)
            continue

        rows = _read_one(path)
        logger.info("Loaded %d rows from %s", len(rows), path)
        for row in rows:
            cell = row.get("cell")
            if not cell:
                unkeyed.append(row)
            elif cell not in merged:
                merged[cell] = row

    if unkeyed:
        logger.warning("%d rows had no 'cell' key (pre-calibration-era run?) and were kept unmerged", len(unkeyed))
    return [*merged.values(), *unkeyed]


def detect_metrics(rows: Sequence[dict[str, Any]]) -> list[str]:
    """Metrics from :data:`KNOWN_METRICS` that these rows actually carry."""
    present = {k for row in rows for k, v in row.items() if isinstance(v, (int, float))}
    return [m for m in KNOWN_METRICS if m in present]


# ---------------------------------------------------------------------------
# Pairing against the zero-duplication control
# ---------------------------------------------------------------------------


@dataclass
class PairedCell:
    """One duplicated cell compared against its matched control."""

    metric: str
    pipeline: str
    mode: str
    epsilon: str
    seed: Any
    variant: str
    params: str
    dup_axis: str
    dup_strength: float
    canary_n_repeats: float
    value: float
    anchor_value: float
    delta: float
    ratio: float

    def group_label(self) -> str:
        return f"{self.pipeline}|{self.mode}|eps={self.epsilon}"


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _conditioning_keys(rows: Sequence[dict[str, Any]], anchor_variant: str) -> tuple[str, ...]:
    """Extra columns to hold fixed within a comparison group, beyond :data:`GROUP_KEYS`.

    ``canary_n_repeats`` means two different things depending on which sweep wrote
    the row, and getting it wrong pairs cells that are not comparable:

    * In the **membership** sweep it is part of the ``canary`` variant's identity —
      the control has no canaries at all, so its count is 0.
    * In the **canary** sweep every cell has an injected secret and the repeat count
      is an independent axis crossed with the corpus variant. Pairing a duplicated
      corpus at 5 repeats against a control at 1 repeat would attribute the secret
      duplication to the corpus.

    Which case applies is readable from the control rows: a control carrying a
    non-zero repeat count means injection is not what distinguishes the variants, so
    the repeat count must be held fixed instead.
    """
    anchors = [r for r in rows if r.get("variant") == anchor_variant]
    if any(_as_float(r.get(CANARY_AXIS)) != 0.0 for r in anchors):
        return (CANARY_AXIS,)
    return ()


def _metric_value(row: dict[str, Any], metric: str) -> float:
    value = row.get(metric)
    if isinstance(value, (int, float)):
        return float(value)
    return float("nan")


def pair_against_anchor(
    rows: Sequence[dict[str, Any]],
    metric: str,
    *,
    anchor_variant: str = ANCHOR_VARIANT,
) -> tuple[list[PairedCell], list[str]]:
    """Compare every duplicated cell to the control cell that matches it.

    Cells match when they agree on :data:`GROUP_KEYS` — pipeline, scoring mode,
    epsilon target and seed — and differ only in the corpus variant. Anything else
    would be comparing two different systems. For canary-sweep rows the secret
    repeat count is held fixed as well; see :func:`_conditioning_keys`.

    Args:
        rows: Result rows, e.g. from :func:`load_rows`.
        metric: Column to compare, e.g. ``"auc"`` or ``"mean_exposure"``.
        anchor_variant: Variant treated as the zero-duplication control.

    Returns:
        ``(pairs, warnings)``. ``warnings`` names every group that could not be
        paired and why; an empty list there is what "the grid is complete" looks
        like. ``ratio`` is NaN when the anchor is zero — the relative change is
        genuinely undefined there, not zero.
    """
    group_keys = GROUP_KEYS + _conditioning_keys(rows, anchor_variant)
    groups: dict[tuple, list[dict[str, Any]]] = {}
    skipped_status: dict[str, int] = {}
    for row in rows:
        status = str(row.get("status", OK_STATUS))
        if status != OK_STATUS:
            skipped_status[status] = skipped_status.get(status, 0) + 1
            continue
        if math.isnan(_metric_value(row, metric)):
            continue
        groups.setdefault(tuple(row.get(k) for k in group_keys), []).append(row)

    warnings: list[str] = []
    for status, count in sorted(skipped_status.items()):
        warnings.append(f"{count} rows with status={status} excluded from the comparison")

    pairs: list[PairedCell] = []
    for key, members in sorted(groups.items(), key=lambda kv: [str(p) for p in kv[0]]):
        label = "|".join(f"{name}={value}" for name, value in zip(group_keys, key))
        anchors = [r for r in members if r.get("variant") == anchor_variant]
        treatments = [r for r in members if r.get("variant") != anchor_variant]
        if not anchors:
            if treatments:
                warnings.append(f"no '{anchor_variant}' control for {label} — {len(treatments)} cells unpaired")
            continue
        if len(anchors) > 1:
            warnings.append(f"{len(anchors)} '{anchor_variant}' rows for {label}; using the first")
        anchor_value = _metric_value(anchors[0], metric)

        for row in treatments:
            value = _metric_value(row, metric)
            ratio = value / anchor_value if anchor_value not in (0.0, -0.0) else float("nan")
            pairs.append(
                PairedCell(
                    metric=metric,
                    pipeline=str(row.get("pipeline", "")),
                    mode=str(row.get("mode", "")),
                    epsilon=str(row.get("epsilon", "")),
                    seed=row.get("seed"),
                    variant=str(row.get("variant", "")),
                    params=str(row.get("params", "-")),
                    dup_axis=str(row.get("dup_axis", "none")),
                    dup_strength=_as_float(row.get("dup_strength")),
                    canary_n_repeats=_as_float(row.get(CANARY_AXIS)),
                    value=value,
                    anchor_value=anchor_value,
                    delta=value - anchor_value,
                    ratio=ratio,
                )
            )

    zero_anchors = sum(1 for p in pairs if math.isnan(p.ratio))
    if zero_anchors:
        warnings.append(f"{zero_anchors} pairs have a zero control value; their ratio is undefined (NaN)")
    return pairs, warnings


# ---------------------------------------------------------------------------
# Pooling over the seed axis
# ---------------------------------------------------------------------------


@dataclass
class PooledStat:
    """A quantity summarised across the seed axis for one configuration."""

    metric: str
    quantity: str  # value | delta | ratio
    pipeline: str
    mode: str
    epsilon: str
    variant: str
    params: str
    dup_axis: str
    dup_strength: float
    canary_n_repeats: float
    n_seeds: int
    mean: float
    std: float
    sem: float
    ci_low: float
    ci_high: float
    seeds: list[Any] = field(default_factory=list)


def _summarise(values: Sequence[float]) -> tuple[float, float, float, float, float]:
    """``(mean, std, sem, ci_low, ci_high)`` — sample std, Student-t 95% interval.

    A single seed has a mean but no spread: std/sem/CI are NaN rather than 0, so a
    one-seed cell cannot be mistaken for a perfectly reproducible one.
    """
    n = len(values)
    if n == 0:
        nan = float("nan")
        return nan, nan, nan, nan, nan
    mean = statistics.fmean(values)
    if n == 1:
        nan = float("nan")
        return mean, nan, nan, nan, nan
    std = statistics.stdev(values)
    sem = std / math.sqrt(n)
    half = student_t_95(n - 1) * sem
    return mean, std, sem, mean - half, mean + half


def pool_over_seeds(pairs: Sequence[PairedCell]) -> list[PooledStat]:
    """Pool ``value``, ``delta`` and ``ratio`` across seeds for each configuration.

    Grouped by everything except the seed. Seeds redraw the member/non-member split
    as well as the mechanism's noise, so this is a spread over independent
    replicates of the experiment, not over noise draws alone.
    """
    grouped: dict[tuple, list[PairedCell]] = {}
    for pair in pairs:
        key = (
            pair.metric,
            pair.pipeline,
            pair.mode,
            pair.epsilon,
            pair.variant,
            pair.params,
            pair.canary_n_repeats,
        )
        grouped.setdefault(key, []).append(pair)

    pooled: list[PooledStat] = []
    for key, members in sorted(grouped.items(), key=lambda kv: [str(p) for p in kv[0]]):
        metric, pipeline, mode, epsilon, variant, params, canary_n_repeats = key
        first = members[0]
        for quantity in ("value", "delta", "ratio"):
            values = [getattr(m, quantity) for m in members]
            usable = [v for v in values if not math.isnan(v)]
            mean, std, sem, ci_low, ci_high = _summarise(usable)
            pooled.append(
                PooledStat(
                    metric=metric,
                    quantity=quantity,
                    pipeline=pipeline,
                    mode=mode,
                    epsilon=epsilon,
                    variant=variant,
                    params=params,
                    dup_axis=first.dup_axis,
                    dup_strength=first.dup_strength,
                    canary_n_repeats=canary_n_repeats,
                    n_seeds=len(usable),
                    mean=mean,
                    std=std,
                    sem=sem,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    seeds=[m.seed for m in members],
                )
            )
    return pooled


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

PAIR_FIELDS = [
    "metric",
    "pipeline",
    "mode",
    "epsilon",
    "seed",
    "variant",
    "params",
    "dup_axis",
    "dup_strength",
    "canary_n_repeats",
    "value",
    "anchor_value",
    "delta",
    "ratio",
]

POOLED_FIELDS = [
    "metric",
    "quantity",
    "pipeline",
    "mode",
    "epsilon",
    "variant",
    "params",
    "dup_axis",
    "dup_strength",
    "canary_n_repeats",
    "n_seeds",
    "mean",
    "std",
    "sem",
    "ci_low",
    "ci_high",
]


def write_csv(path: Path, rows: Sequence[Any], fields: Sequence[str]) -> None:
    """Write dataclass rows to CSV, keeping only ``fields``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row) if not isinstance(row, dict) else row)


def _fmt(value: float, places: int = 4) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "nan"
    return f"{value:.{places}f}"


def render_markdown(
    pairs: Sequence[PairedCell],
    pooled: Sequence[PooledStat],
    warnings: Sequence[str],
    meta: dict[str, Any],
) -> str:
    """Render the degradation tables, following the runners' report style."""
    lines = ["# Duplication degradation vs. the zero-duplication control", ""]
    lines.append(f"- sources: {', '.join(str(s) for s in meta.get('sources', []))}")
    lines.append(f"- metrics: {', '.join(meta.get('metrics', []))}  |  control variant: `{meta.get('anchor')}`")
    lines.append(f"- rows read: {meta.get('n_rows', 0)}  |  paired cells: {len(pairs)}")
    lines.append("")
    lines.append(
        "Each duplicated cell is compared with the `original` cell at the **same** "
        "pipeline, scoring mode, epsilon target and seed. `delta = value - control`; "
        "`ratio = value / control`. For attack metrics a positive delta means "
        "duplication increased leakage — the gap between the formal per-document "
        "guarantee and the protection actually delivered."
    )
    lines.append("")
    lines.append(
        "**Compare pipelines only within the `unified` mode**, and only at epsilon "
        "values both mechanisms can reach. Rows the mechanism could not deliver were "
        "recorded `INFEASIBLE` by the runner and are excluded here rather than being "
        "counted as zero leakage."
    )
    lines.append("")

    if warnings:
        lines.append("## Warnings")
        lines.append("")
        lines += [f"- {w}" for w in warnings]
        lines.append("")

    lines.append("## Per-seed comparisons")
    lines.append("")
    lines.append(
        "| metric | pipeline | mode | eps | seed | variant | params | axis | strength | "
        "canary repeats | value | control | delta | ratio |"
    )
    lines.append("|" + "---|" * 14)
    for p in pairs:
        lines.append(
            f"| {p.metric} | {p.pipeline} | {p.mode} | {p.epsilon} | {p.seed} | {p.variant} | "
            f"{p.params} | {p.dup_axis} | {_fmt(p.dup_strength, 2)} | {_fmt(p.canary_n_repeats, 0)} | "
            f"{_fmt(p.value)} | {_fmt(p.anchor_value)} | {_fmt(p.delta)} | {_fmt(p.ratio, 3)} |"
        )
    lines.append("")

    lines.append("## Pooled across seeds")
    lines.append("")
    lines.append(
        "Mean, sample standard deviation and a 95% Student-t confidence interval over "
        "the seed axis. Seeds redraw the corpus split as well as the mechanism's noise, "
        "so this is the spread over independent replicates of the whole experiment. "
        "A `n_seeds` of 1 leaves the spread columns `nan` — one replicate has a value "
        "but no interval."
    )
    lines.append("")
    lines.append(
        "| metric | quantity | pipeline | mode | eps | variant | params | axis | strength | "
        "canary repeats | n | mean | std | sem | CI low | CI high |"
    )
    lines.append("|" + "---|" * 16)
    for s in pooled:
        lines.append(
            f"| {s.metric} | {s.quantity} | {s.pipeline} | {s.mode} | {s.epsilon} | {s.variant} | "
            f"{s.params} | {s.dup_axis} | {_fmt(s.dup_strength, 2)} | {_fmt(s.canary_n_repeats, 0)} | "
            f"{s.n_seeds} | {_fmt(s.mean)} | {_fmt(s.std)} | {_fmt(s.sem)} | "
            f"{_fmt(s.ci_low)} | {_fmt(s.ci_high)} |"
        )
    return "\n".join(lines) + "\n"
