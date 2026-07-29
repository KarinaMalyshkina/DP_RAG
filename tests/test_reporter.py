"""Tests for the degradation reporter — the layer that compares finished cells.

The arithmetic here is small but it is the arithmetic RQ1 is stated in, so the
values are checked against hand-computed numbers rather than against the
implementation's own output.
"""

from __future__ import annotations

import json
import math

import pytest

from dp_rag_eval.evaluation.reporter import (
    detect_metrics,
    load_rows,
    pair_against_anchor,
    pool_over_seeds,
    student_t_95,
)


def make_row(**overrides):
    """A result row with the columns the runners write."""
    row = {
        "cell": "x",
        "pipeline": "grislain",
        "mode": "unified",
        "epsilon": "4",
        "seed": 42,
        "variant": "original",
        "params": "-",
        "dup_axis": "none",
        "dup_strength": 0.0,
        "duplication_rate": 0.0,
        "n_copies": 0.0,
        "overlap_rate": 0.0,
        "n_hosts": 0.0,
        "n_canaries": 0.0,
        "canary_n_repeats": 0.0,
        "status": "OK",
        "auc": 0.5,
    }
    row.update(overrides)
    row["cell"] = f"{row['pipeline']}|{row['mode']}|{row['variant']}|{row['params']}|{row['epsilon']}|{row['seed']}"
    return row


def dup_row(**overrides):
    base = {
        "variant": "full_duplicate",
        "params": "duplication_rate=0.5,n_copies=1",
        "dup_axis": "duplication_rate",
        "dup_strength": 0.5,
        "duplication_rate": 0.5,
        "n_copies": 1.0,
    }
    base.update(overrides)
    return make_row(**base)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------


def test_delta_and_ratio_against_the_control():
    rows = [make_row(auc=0.60), dup_row(auc=0.75)]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert warnings == []
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.value == pytest.approx(0.75)
    assert pair.anchor_value == pytest.approx(0.60)
    assert pair.delta == pytest.approx(0.15)
    assert pair.ratio == pytest.approx(1.25)
    assert pair.dup_strength == pytest.approx(0.5)


def test_cells_are_paired_only_within_a_matching_configuration():
    # Same duplicated cell, but the control sits at a different epsilon: not a pair.
    rows = [make_row(epsilon="8", auc=0.60), dup_row(epsilon="4", auc=0.75)]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert pairs == []
    assert any("no 'original' control" in w for w in warnings)


def test_each_seed_is_paired_with_its_own_control():
    rows = [
        make_row(seed=42, auc=0.50),
        make_row(seed=43, auc=0.60),
        dup_row(seed=42, auc=0.70),
        dup_row(seed=43, auc=0.90),
    ]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert warnings == []
    deltas = {p.seed: p.delta for p in pairs}
    assert deltas[42] == pytest.approx(0.20)
    assert deltas[43] == pytest.approx(0.30)


def test_infeasible_rows_are_excluded_and_reported():
    rows = [
        make_row(auc=0.60),
        dup_row(auc=0.75),
        dup_row(params="duplication_rate=0.9", status="INFEASIBLE", auc=float("nan")),
    ]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert len(pairs) == 1
    assert any("status=INFEASIBLE" in w for w in warnings)


def test_infeasible_control_leaves_the_group_unpaired():
    rows = [make_row(status="INFEASIBLE", auc=float("nan")), dup_row(auc=0.75)]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert pairs == []
    assert any("no 'original' control" in w for w in warnings)


def test_zero_control_gives_an_undefined_ratio_not_a_crash():
    rows = [make_row(auc=0.0), dup_row(auc=0.4)]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert pairs[0].delta == pytest.approx(0.4)
    assert math.isnan(pairs[0].ratio)
    assert any("zero control value" in w for w in warnings)


def test_canary_rows_pair_on_the_exposure_metric():
    rows = [
        make_row(pipeline="naive", mode="exposure", epsilon="inf", mean_exposure=1.0, auc=None),
        dup_row(pipeline="naive", mode="exposure", epsilon="inf", mean_exposure=3.0, auc=None),
    ]
    pairs, _ = pair_against_anchor(rows, "mean_exposure")

    assert pairs[0].delta == pytest.approx(2.0)
    assert pairs[0].ratio == pytest.approx(3.0)


def test_canary_sweep_holds_the_secret_repeat_count_fixed():
    # In the canary sweep every cell carries an injected secret, so canary_n_repeats
    # is an axis of its own: a duplicated corpus at 5 repeats must not be compared
    # with a control at 1 repeat.
    def canary_cell(**kw):
        return make_row(mode="exposure", epsilon="inf", auc=None, **kw)

    rows = [
        canary_cell(canary_n_repeats=1.0, mean_exposure=1.0),
        canary_cell(canary_n_repeats=5.0, mean_exposure=4.0),
        canary_cell(variant="full_duplicate", params="d=0.5", canary_n_repeats=1.0, mean_exposure=2.0),
        canary_cell(variant="full_duplicate", params="d=0.5", canary_n_repeats=5.0, mean_exposure=7.0),
    ]
    pairs, warnings = pair_against_anchor(rows, "mean_exposure")

    assert warnings == []
    deltas = {p.canary_n_repeats: p.delta for p in pairs}
    assert deltas[1.0] == pytest.approx(1.0)  # 2.0 - 1.0, not 2.0 - 4.0
    assert deltas[5.0] == pytest.approx(3.0)  # 7.0 - 4.0


def test_membership_canary_variant_still_pairs_against_the_canary_free_control():
    # Here the control genuinely has no canaries, so canary_n_repeats is part of the
    # variant's identity and must NOT be held fixed, or nothing would ever pair.
    rows = [
        make_row(auc=0.50),
        make_row(variant="canary", params="n_canaries=10,n_repeats=5", canary_n_repeats=5.0, auc=0.65),
    ]
    pairs, warnings = pair_against_anchor(rows, "auc")

    assert warnings == []
    assert len(pairs) == 1
    assert pairs[0].delta == pytest.approx(0.15)


# ---------------------------------------------------------------------------
# Pooling over seeds
# ---------------------------------------------------------------------------


def test_pooled_mean_std_and_ci_match_hand_computation():
    # deltas are 0.20 and 0.30 -> mean 0.25, sample std 0.0707107, sem 0.05,
    # t(0.975, df=1) = 12.706 -> half-width 0.63530
    rows = [
        make_row(seed=42, auc=0.50),
        make_row(seed=43, auc=0.60),
        dup_row(seed=42, auc=0.70),
        dup_row(seed=43, auc=0.90),
    ]
    pairs, _ = pair_against_anchor(rows, "auc")
    pooled = {s.quantity: s for s in pool_over_seeds(pairs)}

    delta = pooled["delta"]
    assert delta.n_seeds == 2
    assert delta.mean == pytest.approx(0.25)
    assert delta.std == pytest.approx(0.0707107, abs=1e-6)
    assert delta.sem == pytest.approx(0.05)
    assert delta.ci_low == pytest.approx(0.25 - 12.706 * 0.05, abs=1e-6)
    assert delta.ci_high == pytest.approx(0.25 + 12.706 * 0.05, abs=1e-6)
    assert sorted(delta.seeds) == [42, 43]

    assert pooled["value"].mean == pytest.approx(0.80)
    assert pooled["ratio"].mean == pytest.approx((0.70 / 0.50 + 0.90 / 0.60) / 2)


def test_a_single_seed_has_a_mean_but_no_interval():
    rows = [make_row(auc=0.50), dup_row(auc=0.70)]
    pairs, _ = pair_against_anchor(rows, "auc")
    delta = next(s for s in pool_over_seeds(pairs) if s.quantity == "delta")

    assert delta.n_seeds == 1
    assert delta.mean == pytest.approx(0.20)
    assert math.isnan(delta.std)
    assert math.isnan(delta.ci_low)


def test_different_variants_pool_separately():
    rows = [make_row(seed=s, auc=0.5) for s in (42, 43)] + [
        dup_row(seed=42, auc=0.7),
        dup_row(seed=43, auc=0.7),
        dup_row(seed=42, variant="partial_overlap", params="overlap_rate=0.5", auc=0.6),
        dup_row(seed=43, variant="partial_overlap", params="overlap_rate=0.5", auc=0.6),
    ]
    pairs, _ = pair_against_anchor(rows, "auc")
    deltas = {(s.variant, s.quantity): s.mean for s in pool_over_seeds(pairs)}

    assert deltas[("full_duplicate", "delta")] == pytest.approx(0.2)
    assert deltas[("partial_overlap", "delta")] == pytest.approx(0.1)


def test_student_t_95_covers_the_seed_counts_this_framework_uses():
    assert student_t_95(1) == pytest.approx(12.706)
    assert student_t_95(2) == pytest.approx(4.303)  # 3 seeds
    assert student_t_95(1000) == pytest.approx(1.96, abs=0.01)
    assert math.isnan(student_t_95(0))


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_load_rows_merges_directories_and_dedupes_on_cell(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    for path, rows in ((first, [make_row(auc=0.5)]), (second, [make_row(auc=0.5), dup_row(auc=0.7)])):
        path.mkdir()
        (path / "results.json").write_text(json.dumps({"meta": {}, "results": rows}), encoding="utf-8")

    loaded = load_rows([first, second])

    assert len(loaded) == 2  # the shared control row is not duplicated
    assert {r["variant"] for r in loaded} == {"original", "full_duplicate"}


def test_load_rows_coerces_csv_strings_back_to_numbers(tmp_path):
    run = tmp_path / "csv_only"
    run.mkdir()
    (run / "results.csv").write_text(
        "cell,pipeline,mode,epsilon,seed,variant,params,status,auc,realized_epsilon\n"
        "c1,naive,unified,inf,42,original,-,OK,0.875,inf\n",
        encoding="utf-8",
    )

    row = load_rows([run])[0]

    assert row["auc"] == pytest.approx(0.875)
    assert row["seed"] == 42
    assert math.isinf(row["realized_epsilon"])
    assert detect_metrics([row]) == ["auc"]


def test_missing_source_is_skipped_not_fatal(tmp_path):
    assert load_rows([tmp_path / "nope"]) == []
