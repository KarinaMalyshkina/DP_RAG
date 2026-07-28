"""Grid expansion, pipeline construction and resumable result writing for sweeps.

The experiment matrix is large (hundreds of cells across seeds, dataset variants,
epsilon values and pipelines) and runs for days on rented hardware, so two
properties matter more than elegance here:

* **Resumability.** Every finished cell is written immediately and keyed by its
  full configuration, so an interrupted run restarts where it stopped instead of
  redoing GPU-hours.
* **One definition of the grid.** Membership inference and canary extraction sweep
  the same axes; defining them twice is how the two sets of results silently stop
  being comparable.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyTarget
from dp_rag_eval.datasets.variants import (
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
)
from dp_rag_eval.pipelines import Embedder, GrislainPipeline, KogaPipeline, NaivePipeline
from dp_rag_eval.privacy import min_voters_for

logger = logging.getLogger(__name__)

ALL_PIPELINES = ["naive", "koga", "grislain"]
# Pipelines with no privacy mechanism: running them once per epsilon would repeat
# identical work, so they are pinned to the epsilon=infinity cell.
NON_PRIVATE = {"naive"}


# ---------------------------------------------------------------------------
# CLI value parsing
# ---------------------------------------------------------------------------


def parse_floats(text: str) -> list[float]:
    """Parse a comma-separated numeric list; ``inf`` is accepted."""
    values = [v.strip() for v in text.split(",") if v.strip()]
    return [float("inf") if v.lower() in {"inf", "infinity"} else float(v) for v in values]


def parse_ints(text: str) -> list[int]:
    return [int(v) for v in text.split(",") if v.strip()]


def parse_strs(text: str) -> list[str]:
    return [v.strip() for v in text.split(",") if v.strip()]


def param_str(params: dict[str, Any]) -> str:
    """Stable, human-readable rendering of variant params for keys and reports."""
    return ",".join(f"{k}={params[k]}" for k in sorted(params)) or "-"


def eps_label(epsilon: float) -> str:
    return "inf" if not math.isfinite(epsilon) else f"{epsilon:g}"


# ---------------------------------------------------------------------------
# Dataset variants
# ---------------------------------------------------------------------------


@dataclass
class VariantGrid:
    """Strength axes for the duplication sweep, plus the 'base' default strengths."""

    dup_rates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    dup_copies: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])
    overlap_rates: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    canary_repeats: list[int] = field(default_factory=lambda: [1, 2, 5, 10])
    base_dup_rate: float = 0.3
    base_dup_copies: int = 1
    base_overlap_rate: float = 0.3
    overlap_hosts: int = 3
    base_canary_repeats: int = 1
    n_canaries: int = 10


def variant_configs(
    grid: str,
    spec: VariantGrid,
    restrict: list[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Expand a grid name into ``(variant_name, params)`` pairs.

    ``"base"`` is the four variants at a single default strength — the corpus axis
    of the epsilon sweep. ``"duplication"`` instead varies the *strength* of each
    duplication phenomenon (how many verbatim copies, how much of the corpus
    overlaps, how often a secret repeats) while epsilon is held fixed; this is the
    axis RQ1 is about. Duplicates between the two full_duplicate sub-sweeps are
    removed.
    """
    configs: list[tuple[str, dict[str, Any]]] = []

    if grid in {"base", "all"}:
        configs += [
            ("original", {}),
            ("full_duplicate", {"duplication_rate": spec.base_dup_rate, "n_copies": spec.base_dup_copies}),
            ("partial_overlap", {"overlap_rate": spec.base_overlap_rate, "n_hosts": spec.overlap_hosts}),
            ("canary", {"n_canaries": spec.n_canaries, "n_repeats": spec.base_canary_repeats}),
        ]
    if grid in {"duplication", "all"}:
        configs += [
            ("full_duplicate", {"duplication_rate": r, "n_copies": spec.base_dup_copies}) for r in spec.dup_rates
        ]
        configs += [
            ("full_duplicate", {"duplication_rate": spec.base_dup_rate, "n_copies": c}) for c in spec.dup_copies
        ]
        configs += [("partial_overlap", {"overlap_rate": r, "n_hosts": spec.overlap_hosts}) for r in spec.overlap_rates]
        configs += [("canary", {"n_canaries": spec.n_canaries, "n_repeats": n}) for n in spec.canary_repeats]

    if restrict:
        wanted = set(restrict)
        configs = [c for c in configs if c[0] in wanted]

    deduped: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for name, params in configs:
        key = f"{name}:{param_str(params)}"
        if key not in seen:
            seen.add(key)
            deduped.append((name, params))
    return deduped


def make_variant(name: str, params: dict[str, Any], seed: int) -> Any:
    if name == "original":
        return OriginalVariant()
    if name == "full_duplicate":
        return FullDuplicateVariant(seed=seed, **params)
    if name == "partial_overlap":
        return PartialOverlapVariant(seed=seed, **params)
    if name == "canary":
        return CanaryInjectionVariant(seed=seed, **params)
    raise ValueError(f"Unknown variant: {name}")


def truncate_doc(doc: Document, max_chars: int) -> Document:
    if max_chars <= 0 or len(doc.text) <= max_chars:
        return doc.copy()
    return doc.copy(text=doc.text[:max_chars])


def split_members(
    base: list[Document],
    n_members: int,
    n_non_members: int,
    max_doc_chars: int,
) -> tuple[list[Document], list[Document]]:
    """Split a loaded corpus into indexed members and held-out non-members."""
    if len(base) < n_members + n_non_members:
        raise SystemExit(f"Loaded only {len(base)} docs; need {n_members + n_non_members}")
    members = [truncate_doc(d, max_doc_chars) for d in base[:n_members]]
    non_members = [truncate_doc(d, max_doc_chars) for d in base[n_members : n_members + n_non_members]]
    return members, non_members


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


@dataclass
class PipelineParams:
    """Everything the three pipelines need beyond the shared weights.

    Note there is no epsilon here. Budgets are not passed in as parameters — they
    are solved for by :meth:`calibrate` against a
    :class:`~dp_rag_eval.core.types.PrivacyTarget`, so that every pipeline lands on
    the same total guarantee rather than on the same value of some internal knob.
    """

    delta: float = 1e-6
    n_tokens: int = 32
    top_k: int = 5
    n_split: int = 8
    mc_samples: int = 200
    # Grislain splits its budget between retrieval and generation; this is
    # retrieval's share, applied before the outer calibration solves the total.
    retrieval_share: float = 0.1
    # When set, Koga's voter count is raised per cell to the minimum that makes the
    # target feasible, capped here. Paying for 200 voters where 20 suffice is pure
    # cost; running 20 where 200 are needed produces a cell that measures nothing.
    auto_n_split_cap: int | None = None


def build_pipeline(
    name: str,
    target: PrivacyTarget,
    params: PipelineParams,
    *,
    generator: Any,
    tokenizer: Any,
    embedder: Embedder,
    device: str,
) -> tuple[Any, CalibrationResult]:
    """Instantiate one pipeline and calibrate it to ``target``.

    Returns the pipeline together with its calibration. **Check
    ``calibration.feasible`` before running the cell**: an infeasible mechanism
    still produces numbers, but they describe it failing to operate rather than
    protecting anything, and at a glance they are indistinguishable from a
    successful defence.

    ``target.epsilon = inf`` selects each baseline's own non-private code path
    rather than a very large finite budget — Koga falls back to a plain plurality
    vote, Grislain to the upstream ``_non_dp_call`` and a non-DP retrieval
    threshold. That keeps the reference point the published non-private algorithm
    rather than DP with the noise turned down.
    """
    n_split = params.n_split
    if name == "koga" and params.auto_n_split_cap and target.is_private:
        n_split = min(max(params.n_split, min_voters_for(target)), params.auto_n_split_cap)

    if name == "naive":
        pipeline: Any = NaivePipeline(
            generator, tokenizer, embedder, top_k=params.top_k, max_new_tokens=params.n_tokens, device=device
        )
    elif name == "koga":
        pipeline = KogaPipeline(
            generator,
            tokenizer,
            embedder,
            n_split=n_split,
            n_docs=1,
            max_new_tokens=params.n_tokens,
            mc_samples=params.mc_samples,
            device=device,
        )
    elif name == "grislain":
        pipeline = GrislainPipeline(
            generator,
            tokenizer,
            embedder,
            retrieval_share=params.retrieval_share,
            max_new_tokens=params.n_tokens,
            top_k=params.top_k,
        )
    else:
        raise ValueError(f"Unknown pipeline: {name}")

    # Calibration must precede index(): Grislain bakes the retrieval budget into
    # the vector store at build time.
    calibration = pipeline.calibrate(target)
    return pipeline, calibration


# ---------------------------------------------------------------------------
# Resumable results
# ---------------------------------------------------------------------------


class CheckpointedRun:
    """Append-only result store that a killed run can resume from.

    ``results.json`` holds metrics plus the set of finished cells and is rewritten
    atomically after every cell. Per-target rows go to ``predictions.jsonl``
    instead, because at sweep scale (hundreds of cells x hundreds of targets)
    keeping them in the metrics file makes it too large to re-read on resume.
    """

    def __init__(self, output_dir: Path, meta: dict[str, Any], csv_fields: list[str]) -> None:
        self.output_dir = output_dir
        self.meta = meta
        self.csv_fields = csv_fields
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_path = output_dir / "results.json"
        self.predictions_path = output_dir / "predictions.jsonl"
        self.rows: list[dict[str, Any]] = []
        self.done: set[str] = set()
        if self.results_path.exists():
            try:
                saved = json.loads(self.results_path.read_text(encoding="utf-8"))
                self.rows = saved.get("results", [])
                self.done = {r["cell"] for r in self.rows}
                logger.info("Checkpoint: %d cells already done — resuming", len(self.done))
            except Exception:
                logger.warning("Could not read checkpoint at %s; starting fresh", self.results_path)

    def is_done(self, cell: str) -> bool:
        return cell in self.done

    def record(self, row: dict[str, Any], predictions: list[dict[str, Any]] | None = None) -> None:
        """Persist one finished cell. Safe to kill the process right after this returns."""
        self.rows.append(row)
        self.done.add(row["cell"])
        if predictions:
            with self.predictions_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"cell": row["cell"], "predictions": predictions}, ensure_ascii=False) + "\n")
        tmp = self.results_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"meta": self.meta, "results": self.rows}, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        tmp.replace(self.results_path)

    def finalise(self, report: str) -> None:
        with (self.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.csv_fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.rows)
        (self.output_dir / "report.md").write_text(report, encoding="utf-8")


def seeded_sample(items: list[Document], k: int, seed: int) -> list[Document]:
    """Deterministic subsample used to pick attack targets."""
    return random.Random(seed).sample(items, min(k, len(items)))
