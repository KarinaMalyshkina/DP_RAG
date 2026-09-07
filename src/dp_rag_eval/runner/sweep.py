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
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dp_rag_eval.core.types import CalibrationResult, Document, PrivacyTarget
from dp_rag_eval.datasets.variants import VARIANTS
from dp_rag_eval.pipelines import Embedder, GrislainPipeline, KogaPipeline, NaivePipeline
from dp_rag_eval.privacy import min_voters_for

logger = logging.getLogger(__name__)

ALL_PIPELINES = ["naive", "koga", "grislain"]

# Offset separating the corpus-construction random stream from the target-sampling
# one; see :func:`variant_seed`. Recorded in every run's ``meta`` because it changes
# the corpus, so results produced with and without it must not be pooled.
VARIANT_SEED_OFFSET = 7919
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


# Numeric variant parameters promoted to their own result columns. Duplication
# strength used to live only inside the ``params`` string, which meant pairing a
# duplicated cell with its control required parsing that string back apart. These
# columns are always present and always numeric, so the control row (all zeros) and
# a treatment row are directly comparable.
#
# ``canary_n_repeats`` rather than ``n_repeats``: the canary runner has its own
# repeats axis applied *on top of* a corpus variant, and the two must not collide.
PARAM_COLUMNS: dict[str, tuple[str, float]] = {
    "duplication_rate": ("duplication_rate", 0.0),
    "n_copies": ("n_copies", 0.0),
    "overlap_rate": ("overlap_rate", 0.0),
    "n_hosts": ("n_hosts", 0.0),
    "n_canaries": ("n_canaries", 0.0),
    "n_repeats": ("canary_n_repeats", 0.0),
}

# Column names contributed by :meth:`VariantCell.columns`, in report order.
VARIANT_COLUMNS: list[str] = ["variant", "params", "dup_axis", "dup_strength"] + [
    column for column, _ in PARAM_COLUMNS.values()
]


@dataclass(frozen=True)
class VariantCell:
    """One corpus condition in a sweep: a variant name plus its strength parameters.

    ``axis`` / ``strength`` record *which* duplication ladder this cell sits on and
    where. They are traceability hints for plotting, not something analysis should
    depend on: identical corpus configurations are deduplicated, so a config that
    appears on two ladders keeps only the first label. Pair cells on
    :meth:`columns` instead, which is complete.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    axis: str = "none"
    strength: float = 0.0

    def key(self) -> str:
        """Identity of the corpus this cell produces — the deduplication key."""
        return f"{self.name}:{param_str(self.params)}"

    def columns(self) -> dict[str, Any]:
        """Result-row columns describing this cell, joinable without parsing ``params``."""
        row: dict[str, Any] = {
            "variant": self.name,
            "params": param_str(self.params),
            "dup_axis": self.axis,
            "dup_strength": float(self.strength),
        }
        for param, (column, default) in PARAM_COLUMNS.items():
            row[column] = float(self.params.get(param, default))
        return row


def variant_configs(
    grid: str,
    spec: VariantGrid,
    restrict: list[str] | None = None,
) -> list[VariantCell]:
    """Expand a grid name into :class:`VariantCell` values.

    ``"base"`` is the four variants at a single default strength — the corpus axis
    of the epsilon sweep. ``"duplication"`` instead varies the *strength* of each
    duplication phenomenon (how many verbatim copies, how much of the corpus
    overlaps, how often a secret repeats) while epsilon is held fixed; this is the
    axis RQ1 is about. Duplicate corpus configurations are removed.

    **The duplication grid leads with the zero-duplication control.** Every
    duplicated cell is only interpretable against ``AUC_orig`` at the same pipeline,
    mode, epsilon and seed; without the control in the same grid — and therefore in
    the same results file — that subtraction has nothing to subtract.
    """
    configs: list[VariantCell] = []

    if grid in {"base", "all"}:
        configs += [
            VariantCell("original"),
            VariantCell(
                "full_duplicate",
                {"duplication_rate": spec.base_dup_rate, "n_copies": spec.base_dup_copies},
                axis="duplication_rate",
                strength=spec.base_dup_rate,
            ),
            VariantCell(
                "partial_overlap",
                {"overlap_rate": spec.base_overlap_rate, "n_hosts": spec.overlap_hosts},
                axis="overlap_rate",
                strength=spec.base_overlap_rate,
            ),
            VariantCell(
                "canary",
                {"n_canaries": spec.n_canaries, "n_repeats": spec.base_canary_repeats},
                axis="canary_n_repeats",
                strength=spec.base_canary_repeats,
            ),
        ]
    if grid in {"duplication", "all"}:
        configs += [
            VariantCell("original"),
            *(
                VariantCell(
                    "full_duplicate",
                    {"duplication_rate": r, "n_copies": spec.base_dup_copies},
                    axis="duplication_rate",
                    strength=r,
                )
                for r in spec.dup_rates
            ),
            *(
                VariantCell(
                    "full_duplicate",
                    {"duplication_rate": spec.base_dup_rate, "n_copies": c},
                    axis="n_copies",
                    strength=c,
                )
                for c in spec.dup_copies
            ),
            *(
                VariantCell(
                    "partial_overlap",
                    {"overlap_rate": r, "n_hosts": spec.overlap_hosts},
                    axis="overlap_rate",
                    strength=r,
                )
                for r in spec.overlap_rates
            ),
            *(
                VariantCell(
                    "canary",
                    {"n_canaries": spec.n_canaries, "n_repeats": n},
                    axis="canary_n_repeats",
                    strength=n,
                )
                for n in spec.canary_repeats
            ),
        ]

    if restrict:
        wanted = set(restrict)
        configs = [c for c in configs if c.name in wanted]

    deduped: list[VariantCell] = []
    seen: set[str] = set()
    for cell in configs:
        if cell.key() not in seen:
            seen.add(cell.key())
            deduped.append(cell)
    return deduped


def grid_from_args(args: Any) -> VariantGrid:
    """Build a :class:`VariantGrid` from parsed CLI arguments.

    Shared by every runner so the three sweeps expand the same grid from the same
    flags. A runner that builds its own ``VariantGrid()`` silently ignores the
    ``--dup-rates`` / ``--overlap-rates`` the user passed it.
    """
    spec = VariantGrid()
    return VariantGrid(
        dup_rates=parse_floats(getattr(args, "dup_rates", None) or "") or spec.dup_rates,
        dup_copies=parse_ints(getattr(args, "dup_copies", None) or "") or spec.dup_copies,
        overlap_rates=parse_floats(getattr(args, "overlap_rates", None) or "") or spec.overlap_rates,
        canary_repeats=parse_ints(getattr(args, "canary_repeats", None) or "") or spec.canary_repeats,
        base_dup_rate=getattr(args, "base_dup_rate", spec.base_dup_rate),
        base_dup_copies=getattr(args, "base_dup_copies", spec.base_dup_copies),
        base_overlap_rate=getattr(args, "base_overlap_rate", spec.base_overlap_rate),
        overlap_hosts=getattr(args, "overlap_hosts", spec.overlap_hosts),
        base_canary_repeats=getattr(args, "base_canary_repeats", spec.base_canary_repeats),
        n_canaries=getattr(args, "n_canaries", spec.n_canaries),
    )


def variant_seed(seed: int) -> int:
    """Derive the corpus-construction seed from the experiment seed.

    Variants and attack targets are both drawn from ``random.Random(seed)`` over a
    population of the same size, and ``random.sample`` takes the same algorithmic
    branch each time, so with a shared seed the two draws agree on a prefix: every
    document a variant duplicates is one of the first targets the attack scores.
    At a duplication rate of 0.1 that put 100 of 250 scored members in the
    duplicated set where an independent draw would have put about 25, and from a
    rate of 0.25 upward it put all of them there — which is why every rate at or
    above 0.3 measured the same treated population and differed only in corpus
    size.

    Offsetting the stream makes the two draws independent, so the fraction of
    scored targets that were duplicated is the duplication rate, as the name
    promises. The offset is a fixed odd constant rather than something derived
    from the rate: the corpus a given seed produces must not depend on who is
    looking at it.
    """
    return seed + VARIANT_SEED_OFFSET


def make_variant(name: str, params: dict[str, Any], seed: int) -> Any:
    """Build one dataset variant by name, from the single registry in ``datasets.variants``.

    ``seed`` is the experiment seed; the variant is built from
    :func:`variant_seed` of it, so corpus construction is independent of target
    sampling.
    """
    try:
        cls = VARIANTS[name]
    except KeyError:
        raise ValueError(f"Unknown variant: {name!r}; known: {sorted(VARIANTS)}") from None
    return cls(seed=variant_seed(seed), **params)


def make_variant_cell(cell: VariantCell, seed: int) -> Any:
    """Build the variant described by ``cell``."""
    return make_variant(cell.name, cell.params, seed)


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
    n_docs: int = 1
    mc_samples: int = 200
    # Grislain splits its budget between retrieval and generation; this is
    # retrieval's share, applied before the outer calibration solves the total.
    retrieval_share: float = 0.1
    # When set, Koga's voter count is raised per cell to the minimum that makes the
    # target feasible, capped here. Paying for 200 voters where 20 suffice is pure
    # cost; running 20 where 200 are needed produces a cell that measures nothing.
    auto_n_split_cap: int | None = None
    # Multiplier on the minimum voter count. The minimum is the count at which the
    # selection *could* succeed under complete unanimity, which is a necessary
    # condition the sizing treated as sufficient: at headroom 1.0 a cell needs
    # essentially every voter to agree, and with one document per voter that never
    # happens on free-form text. Raising it costs GPU time and nothing else — the
    # LD cutoff is 1 + log(2/delta_step)/eps_step, independent of the voter count,
    # and the accountant is never told the count either.
    n_split_headroom: float = 1.0
    # Overrides the sizing entirely, including the cap: every private cell gets
    # exactly this many voters. Used to hold the voter count constant across a
    # budget sweep, which is what removes it as a confound on the privacy-utility
    # slope. ``None`` leaves the automatic sizing in charge.
    n_split_fixed: int | None = None


def koga_voter_count(target: PrivacyTarget, params: PipelineParams) -> int:
    """Voters Koga gets for one cell, under whichever sizing policy is in force.

    Three policies, in precedence order:

    * ``n_split_fixed`` — exactly that count, for every budget including the
      non-private reference. Holding it constant is what makes a budget sweep's
      slope attributable to the noise rather than to the voter count.
    * ``auto_n_split_cap`` — ``headroom`` times the minimum feasible count, floored
      at ``n_split`` and capped. Headroom above 1.0 is the difference between a
      cell that needs unanimity and one that needs a plurality.
    * neither — the flat ``n_split``.

    None of this touches the guarantee: the vote cutoff does not depend on the
    voter count (:func:`~dp_rag_eval.privacy.ld_vote_threshold`) and the count is
    not an input to the composition either.
    """
    if params.n_split_fixed is not None:
        return params.n_split_fixed
    if not (params.auto_n_split_cap and target.is_private):
        return params.n_split

    minimum = min_voters_for(target)
    wanted = max(params.n_split, math.ceil(params.n_split_headroom * minimum))
    n_split = min(wanted, params.auto_n_split_cap)
    if wanted > params.auto_n_split_cap:
        logger.warning(
            "eps=%s: headroom %.2fx of %d voters wants %d but the cap is %d, so this cell runs at "
            "%.2fx and may still refuse every selection. Raise --auto-n-split-cap or accept it.",
            target.epsilon,
            params.n_split_headroom,
            minimum,
            wanted,
            params.auto_n_split_cap,
            n_split / minimum,
        )
    return n_split


def cell_config_suffix(name: str, target: PrivacyTarget, params: PipelineParams) -> str:
    """The part of a cell's identity that comes from the pipeline's own configuration.

    Only Koga contributes one, and only the voter count: it is the knob a sweep now
    varies deliberately, and two cells differing in it are different systems — one
    that answers and one that does not, at the same budget. Without it in the key,
    a fixed-count control sharing a directory with an auto-sized run is skipped as
    already done, and the command exits successfully having measured nothing. That
    happened once; the guard is here so it cannot happen twice.

    Knobs that do not vary within this project (``top_k``, ``mc_samples``,
    ``retrieval_share``) stay out, so that keys already written remain readable.
    """
    if name != "koga":
        return ""
    return f"|n_split={koga_voter_count(target, params)}"


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
    n_split = koga_voter_count(target, params) if name == "koga" else params.n_split

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
            n_docs=params.n_docs,
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

    Several commands may deliberately share one output directory — an epsilon sweep
    and a duplication sweep must land in the same file for their cells to be
    comparable at all. Cell keys carry the full configuration, so shared cells are
    skipped rather than re-run, and each command's ``meta`` is kept in
    ``meta_history`` instead of being overwritten by the next one.

    ``results.csv`` and ``report.md`` are refreshed after every cell, not only at the
    end. A multi-day sweep that only produced readable output on completion would be
    unmonitorable for its entire duration, and rsyncing a half-finished run would
    yield nothing but raw JSON. Set :attr:`report_fn` to a callable returning the
    rendered report to enable the ``report.md`` half.
    """

    def __init__(self, output_dir: Path, meta: dict[str, Any], csv_fields: list[str]) -> None:
        self.output_dir = output_dir
        self.meta = meta
        self.csv_fields = csv_fields
        self.report_fn: Callable[[], str] | None = None
        self.meta_history: list[dict[str, Any]] = []
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
                self.meta_history = list(saved.get("meta_history", []))
                previous = saved.get("meta")
                if previous and previous != meta and previous not in self.meta_history:
                    self.meta_history.append(previous)
                logger.info("Checkpoint: %d cells already done — resuming", len(self.done))
            except Exception:
                logger.warning("Could not read checkpoint at %s; starting fresh", self.results_path)
        self._reject_a_foreign_generator()

    def _reject_a_foreign_generator(self) -> None:
        """Refuse to add cells to a directory a different generator wrote.

        A cell key identifies a *configuration*, not a model, so two generators
        produce identical keys for the same configuration. Pointed at a shared
        directory, the second model's run would find every cell already done and
        exit successfully having measured nothing — the same silent-success failure
        the voter-count suffix exists to prevent, one level up.

        The model is deliberately kept **out** of the key rather than added to it.
        Adding it would change every key already written, so resuming any existing
        run would recompute finished cells from scratch. A second model is a second
        experiment and belongs in its own ``--output-dir``; this only makes the
        mistake loud.
        """
        mine = self.meta.get("generator")
        if not mine:
            return
        found = {m.get("generator") for m in self.meta_history if isinstance(m, dict)}
        found |= {r.get("generator") for r in self.rows}
        others = sorted(g for g in found if g and g != mine)
        if others:
            raise RuntimeError(
                f"{self.results_path} already holds cells produced by {', '.join(others)}, but this "
                f"run uses {mine}. Cell keys do not carry the model, so resuming here would skip "
                "every shared cell and measure nothing. Give each generator its own --output-dir."
            )

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
        payload = {"meta": self.meta, "meta_history": self.meta_history, "results": self.rows}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.results_path)
        self._write_derived()

    def _write_derived(self) -> None:
        """Rewrite ``results.csv`` and ``report.md`` from the rows held so far.

        Both are cheap next to a cell (a few hundred rows of text against minutes of
        GPU work), so they are rewritten every time rather than on an interval — that
        keeps ``rsync`` of a live run useful and removes "how stale is this" as a
        question. Rendering is wrapped: a formatting bug in a report must not lose a
        finished cell that has already been written to ``results.json``.
        """
        try:
            with (self.output_dir / "results.csv").open("w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=self.csv_fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
            if self.report_fn is not None:
                (self.output_dir / "report.md").write_text(self.report_fn(), encoding="utf-8")
        except Exception:
            logger.exception("Could not refresh results.csv/report.md; results.json is unaffected")

    def finalise(self, report: str | None = None) -> None:
        """Write the final ``results.csv`` and ``report.md``."""
        if report is not None:
            self.report_fn = lambda: report  # type: ignore[assignment,return-value]
        self._write_derived()


def seeded_sample(items: list[Document], k: int, seed: int) -> list[Document]:
    """Deterministic subsample used to pick attack targets."""
    return random.Random(seed).sample(items, min(k, len(items)))
