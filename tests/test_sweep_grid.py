"""Tests for grid expansion and variant construction in ``runner.sweep``.

Two properties here are load-bearing for the analysis and easy to break silently:
the duplication grid must contain its zero-duplication control, and duplication
strength must be readable from typed columns rather than from a params string.
"""

from __future__ import annotations

import pytest

from dp_rag_eval.datasets.variants import (
    VARIANTS,
    CanaryInjectionVariant,
    FullDuplicateVariant,
    OriginalVariant,
    PartialOverlapVariant,
)
from dp_rag_eval.runner.sweep import (
    VARIANT_COLUMNS,
    VariantCell,
    VariantGrid,
    grid_from_args,
    make_variant,
    variant_configs,
)


def test_duplication_grid_contains_the_zero_duplication_control():
    cells = variant_configs("duplication", VariantGrid())

    assert cells[0].name == "original"
    assert [c.name for c in cells].count("original") == 1
    # ... and the treatments it is the control for
    assert {"full_duplicate", "partial_overlap", "canary"} <= {c.name for c in cells}


def test_base_grid_still_has_all_four_variants_once():
    cells = variant_configs("base", VariantGrid())

    assert [c.name for c in cells] == ["original", "full_duplicate", "partial_overlap", "canary"]


def test_all_grid_deduplicates_shared_configurations():
    cells = variant_configs("all", VariantGrid())
    keys = [c.key() for c in cells]

    assert len(keys) == len(set(keys))
    assert keys.count("original:-") == 1


def test_restrict_filters_by_variant_name():
    cells = variant_configs("duplication", VariantGrid(), ["partial_overlap"])

    assert {c.name for c in cells} == {"partial_overlap"}


def test_strength_axes_are_labelled():
    cells = variant_configs("duplication", VariantGrid(dup_rates=[0.5], dup_copies=[3], overlap_rates=[0.2]))
    by_axis = {(c.name, c.axis): c.strength for c in cells}

    assert by_axis[("full_duplicate", "duplication_rate")] == pytest.approx(0.5)
    assert by_axis[("full_duplicate", "n_copies")] == pytest.approx(3)
    assert by_axis[("partial_overlap", "overlap_rate")] == pytest.approx(0.2)
    assert by_axis[("original", "none")] == pytest.approx(0.0)


def test_columns_are_numeric_and_complete_for_every_cell():
    for cell in variant_configs("all", VariantGrid()):
        columns = cell.columns()
        assert set(columns) == set(VARIANT_COLUMNS)
        assert isinstance(columns["variant"], str)
        for name in VARIANT_COLUMNS:
            if name not in {"variant", "params", "dup_axis"}:
                assert isinstance(columns[name], float), name


def test_control_columns_are_all_zero_so_it_joins_against_any_treatment():
    columns = VariantCell("original").columns()

    assert columns["variant"] == "original"
    assert columns["dup_axis"] == "none"
    assert columns["duplication_rate"] == 0.0
    assert columns["overlap_rate"] == 0.0
    assert columns["canary_n_repeats"] == 0.0


def test_canary_params_land_in_canary_n_repeats_not_n_repeats():
    # The canary runner has its own repeats axis; the two must not collide.
    columns = VariantCell("canary", {"n_canaries": 10, "n_repeats": 5}).columns()

    assert columns["canary_n_repeats"] == 5.0
    assert columns["n_canaries"] == 10.0
    assert "n_repeats" not in columns


# ---------------------------------------------------------------------------
# Variant construction goes through the single registry
# ---------------------------------------------------------------------------


def test_make_variant_builds_every_registered_variant():
    assert set(VARIANTS) == {"original", "full_duplicate", "partial_overlap", "canary"}
    expected = {
        "original": OriginalVariant,
        "full_duplicate": FullDuplicateVariant,
        "partial_overlap": PartialOverlapVariant,
        "canary": CanaryInjectionVariant,
    }
    for name, cls in expected.items():
        assert isinstance(make_variant(name, {}, seed=7), cls)


def test_make_variant_passes_seed_and_params_through():
    variant = make_variant("full_duplicate", {"duplication_rate": 0.4, "n_copies": 2}, seed=11)

    assert variant.duplication_rate == pytest.approx(0.4)
    assert variant.n_copies == 2
    assert variant.seed == 11


def test_make_variant_rejects_an_unknown_name_with_the_known_ones():
    with pytest.raises(ValueError, match="Unknown variant"):
        make_variant("nope", {}, seed=1)


def test_every_grid_cell_can_actually_be_built(sample_documents):
    for cell in variant_configs("all", VariantGrid()):
        corpus = make_variant(cell.name, cell.params, seed=3).apply(sample_documents)
        assert corpus and corpus[0].metadata["variant"] == cell.name


# ---------------------------------------------------------------------------
# Shared CLI -> grid conversion
# ---------------------------------------------------------------------------


class _Args:
    dup_rates = "0.25,0.75"
    overlap_rates = "0.5"
    base_dup_rate = 0.25


def test_grid_from_args_reads_cli_overrides_and_defaults_the_rest():
    grid = grid_from_args(_Args())

    assert grid.dup_rates == [0.25, 0.75]
    assert grid.overlap_rates == [0.5]
    assert grid.base_dup_rate == pytest.approx(0.25)
    # Unset flags keep the shared defaults rather than becoming empty.
    assert grid.dup_copies == VariantGrid().dup_copies
    assert grid.canary_repeats == VariantGrid().canary_repeats
