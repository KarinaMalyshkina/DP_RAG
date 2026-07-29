"""Read a running sweep's checkpoint and summarise where it has got to.

A sweep runs for days on rented hardware. The question that matters while it is
running is not "what is the AUC" but "is this still alive, how much is left, and is
anything quietly going wrong" — and the answer has to be available *without*
attaching to the tmux session, because attaching is how a sweep gets killed by a
stray Ctrl-C.

Everything here reads ``results.json``, which :class:`CheckpointedRun` rewrites
atomically after every finished cell, so it is always safe to read mid-run.

The failure this is really built to catch is the silent one. A Koga cell whose
Limited-Domain selection never fires produces an AUC of exactly 0.5 and a
``status=OK`` — indistinguishable at a glance from a mechanism that successfully
defeated the attack. ``ld_failure_rate`` is what separates the two, so it is
surfaced as a flag rather than left in a column of the CSV.
"""

from __future__ import annotations

import json
import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Above this share of failed Limited-Domain selections, the cell measured the
#: mechanism failing to operate rather than the leakage it removes (see docs/RUNBOOK_GPU.md, section 4a).
LD_FAILURE_ALARM = 0.95


def _fmt_duration(seconds: float) -> str:
    if seconds != seconds or seconds < 0:
        return "?"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@dataclass
class RunProgress:
    """Where one sweep output directory has got to."""

    path: Path
    n_done: int = 0
    planned: int | None = None
    by_status: dict[str, int] = field(default_factory=dict)
    by_pipeline: dict[str, int] = field(default_factory=dict)
    mean_seconds: float = float("nan")
    spent_seconds: float = 0.0
    eta_seconds: float = float("nan")
    updated: datetime | None = None
    recent: list[dict[str, Any]] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def remaining(self) -> int | None:
        if self.planned is None:
            return None
        return max(0, self.planned - self.n_done)

    @property
    def percent(self) -> float:
        if not self.planned:
            return float("nan")
        return 100.0 * self.n_done / self.planned


def _flags(rows: list[dict[str, Any]]) -> list[str]:
    """Problems worth interrupting a run for, in severity order."""
    flags: list[str] = []

    errors = [r for r in rows if r.get("status") == "ERROR"]
    if errors:
        sample = str(errors[-1].get("error", ""))[:120]
        flags.append(f"{len(errors)} cells ERRORed — most recent: {sample}")

    # An LD failure rate near 1 means no vote ever cleared the Limited-Domain
    # cutoff: the cell reports AUC 0.5 and status OK, but measured nothing.
    dead = [
        r
        for r in rows
        if isinstance(r.get("ld_failure_rate"), (int, float)) and float(r["ld_failure_rate"]) > LD_FAILURE_ALARM
    ]
    if dead:
        flags.append(
            f"{len(dead)} cells have ld_failure_rate > {LD_FAILURE_ALARM}: their AUC measures "
            "mechanism infeasibility, not privacy — check n_split before using them"
        )

    empty = [r for r in rows if isinstance(r.get("n_answers"), (int, float)) and float(r["n_answers"]) == 0]
    if empty:
        flags.append(f"{len(empty)} utility cells produced no answers at all")

    infeasible = sum(1 for r in rows if r.get("status") == "INFEASIBLE")
    if infeasible and infeasible == len(rows):
        flags.append("every cell so far is INFEASIBLE — the epsilon grid may be out of the mechanism's reach")

    return flags


def read_progress(run_dir: str | Path, *, recent: int = 5) -> RunProgress:
    """Summarise the checkpoint in ``run_dir`` without disturbing the run.

    Args:
        run_dir: A sweep output directory containing ``results.json``.
        recent: How many of the most recently finished cells to keep.

    Returns:
        A :class:`RunProgress`. A missing or unreadable checkpoint sets ``error``
        rather than raising — a watcher polling a directory that does not exist yet
        should say so, not crash.
    """
    path = Path(run_dir)
    results = path / "results.json"
    progress = RunProgress(path=path)

    if not results.exists():
        progress.error = f"no results.json in {path} (run not started, or wrong --output-dir)"
        return progress
    try:
        payload = json.loads(results.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - a partially written file is transient, not fatal
        progress.error = f"could not read {results}: {type(exc).__name__}: {exc}"
        return progress

    rows: list[dict[str, Any]] = payload.get("results", [])
    meta: dict[str, Any] = payload.get("meta", {})
    progress.n_done = len(rows)
    progress.planned = meta.get("planned_cells")
    progress.updated = datetime.fromtimestamp(results.stat().st_mtime, tz=timezone.utc)

    for row in rows:
        status = str(row.get("status", "?"))
        progress.by_status[status] = progress.by_status.get(status, 0) + 1
        pipeline = str(row.get("pipeline", "?"))
        progress.by_pipeline[pipeline] = progress.by_pipeline.get(pipeline, 0) + 1

    # INFEASIBLE cells cost no time, so averaging over them would flatter the ETA.
    timings = [float(r["seconds"]) for r in rows if isinstance(r.get("seconds"), (int, float)) and r["seconds"] > 0]
    progress.spent_seconds = sum(timings)
    if timings:
        progress.mean_seconds = statistics.fmean(timings)
        remaining = progress.remaining
        if remaining is not None:
            progress.eta_seconds = progress.mean_seconds * remaining

    progress.recent = rows[-recent:]
    progress.flags = _flags(rows)
    return progress


def render_progress(progress: RunProgress) -> str:
    """A plain-text block for a terminal, an SSH one-liner or a chat message."""
    if progress.error:
        return f"{progress.path}: {progress.error}"

    lines: list[str] = []
    if progress.planned:
        bar_width = 28
        filled = int(bar_width * progress.n_done / progress.planned) if progress.planned else 0
        bar = "#" * filled + "." * (bar_width - filled)
        lines.append(f"{progress.path}  [{bar}] {progress.n_done}/{progress.planned} ({progress.percent:.1f}%)")
    else:
        lines.append(f"{progress.path}  {progress.n_done} cells done (planned count not recorded)")

    status = "  ".join(f"{name}={count}" for name, count in sorted(progress.by_status.items()))
    lines.append(f"  status:    {status or '-'}")
    pipelines = "  ".join(f"{name}={count}" for name, count in sorted(progress.by_pipeline.items()))
    lines.append(f"  pipelines: {pipelines or '-'}")

    if not math.isnan(progress.mean_seconds):
        eta = _fmt_duration(progress.eta_seconds) if not math.isnan(progress.eta_seconds) else "? (no planned count)"
        lines.append(
            f"  timing:    {_fmt_duration(progress.mean_seconds)}/cell  |  "
            f"spent {_fmt_duration(progress.spent_seconds)}  |  ETA {eta}"
        )
    if progress.updated:
        age = (datetime.now(tz=timezone.utc) - progress.updated).total_seconds()
        stale = "  <-- no new cell for a while" if age > 3600 else ""
        lines.append(f"  last cell: {_fmt_duration(age)} ago{stale}")

    if progress.flags:
        lines.append("  FLAGS:")
        lines += [f"    ! {flag}" for flag in progress.flags]

    if progress.recent:
        lines.append("  recent:")
        for row in progress.recent:
            seconds = row.get("seconds", 0.0)
            seconds = float(seconds) if isinstance(seconds, (int, float)) else 0.0
            lines.append(f"    {str(row.get('status', '?')):<11} {_fmt_duration(seconds):>7}  {row.get('cell', '?')}")

    return "\n".join(lines)
