"""Tests for the live-progress reader.

The point of this module is to be readable *during* a multi-day run, so the cases
that matter are the awkward ones: a directory that does not exist yet, a checkpoint
being rewritten as it is read, and cells that report OK while having measured
nothing.
"""

from __future__ import annotations

import json
import math

import pytest

from dp_rag_eval.evaluation.progress import LD_FAILURE_ALARM, read_progress, render_progress


def write_run(path, rows, *, planned=None):
    path.mkdir(parents=True, exist_ok=True)
    meta = {"attack": "test"}
    if planned is not None:
        meta["planned_cells"] = planned
    (path / "results.json").write_text(json.dumps({"meta": meta, "results": rows}), encoding="utf-8")
    return path


def cell(name, status="OK", seconds=10.0, **extra):
    return {"cell": name, "pipeline": "koga", "status": status, "seconds": seconds, **extra}


def test_counts_and_eta_from_a_partially_finished_run(tmp_path):
    run = write_run(tmp_path / "r", [cell("a"), cell("b"), cell("c")], planned=10)

    progress = read_progress(run)

    assert progress.n_done == 3
    assert progress.planned == 10
    assert progress.remaining == 7
    assert progress.percent == pytest.approx(30.0)
    assert progress.mean_seconds == pytest.approx(10.0)
    assert progress.eta_seconds == pytest.approx(70.0)


def test_infeasible_cells_are_counted_but_do_not_flatter_the_eta(tmp_path):
    # INFEASIBLE cells are recorded in milliseconds; averaging over them would
    # predict a sweep far faster than it will actually be.
    rows = [cell("a", seconds=60.0)] + [cell(f"skip{i}", status="INFEASIBLE", seconds=0.0) for i in range(9)]
    run = write_run(tmp_path / "r", rows, planned=20)

    progress = read_progress(run)

    assert progress.n_done == 10
    assert progress.by_status == {"OK": 1, "INFEASIBLE": 9}
    assert progress.mean_seconds == pytest.approx(60.0)
    assert progress.eta_seconds == pytest.approx(600.0)


def test_missing_planned_count_leaves_eta_unknown_rather_than_wrong(tmp_path):
    run = write_run(tmp_path / "r", [cell("a")])

    progress = read_progress(run)

    assert progress.planned is None
    assert progress.remaining is None
    assert math.isnan(progress.eta_seconds)
    assert "1 cells done" in render_progress(progress)


def test_a_run_that_has_not_started_reports_that_instead_of_crashing(tmp_path):
    progress = read_progress(tmp_path / "never_created")

    assert progress.error and "no results.json" in progress.error
    assert "no results.json" in render_progress(progress)


def test_a_half_written_checkpoint_is_reported_not_raised(tmp_path):
    run = tmp_path / "r"
    run.mkdir()
    (run / "results.json").write_text('{"meta": {}, "results": [{"cell"', encoding="utf-8")

    progress = read_progress(run)

    assert progress.error and "could not read" in progress.error


# ---------------------------------------------------------------------------
# Flags — the failures that do not look like failures
# ---------------------------------------------------------------------------


def test_dead_ld_cells_are_flagged_even_though_their_status_is_ok(tmp_path):
    rows = [cell("a", ld_failure_rate=1.0, auc=0.5), cell("b", ld_failure_rate=0.0, auc=0.8)]
    run = write_run(tmp_path / "r", rows, planned=2)

    flags = read_progress(run).flags

    assert any("ld_failure_rate" in f for f in flags)
    assert any("1 cells" in f for f in flags)


def test_ld_failure_below_the_alarm_is_not_flagged(tmp_path):
    run = write_run(tmp_path / "r", [cell("a", ld_failure_rate=LD_FAILURE_ALARM - 0.01)], planned=1)

    assert read_progress(run).flags == []


def test_errors_are_flagged_with_the_most_recent_message(tmp_path):
    rows = [cell("a"), cell("b", status="ERROR", error="RuntimeError: CUDA out of memory")]
    run = write_run(tmp_path / "r", rows, planned=2)

    flags = read_progress(run).flags

    assert any("CUDA out of memory" in f for f in flags)


def test_an_all_infeasible_grid_is_flagged_as_out_of_reach(tmp_path):
    rows = [cell(f"c{i}", status="INFEASIBLE", seconds=0.0) for i in range(4)]
    run = write_run(tmp_path / "r", rows, planned=4)

    assert any("out of the mechanism's reach" in f for f in read_progress(run).flags)


def test_render_includes_progress_bar_statuses_and_recent_cells(tmp_path):
    run = write_run(tmp_path / "r", [cell("koga|unified|original|eps=4|seed=42")], planned=4)

    text = render_progress(read_progress(run))

    assert "1/4" in text and "25.0%" in text
    assert "OK=1" in text
    assert "koga|unified|original|eps=4|seed=42" in text
