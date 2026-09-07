"""A cell's key must separate cells that measure different things.

Finished cells are keyed by configuration so a killed sweep resumes instead of
redoing GPU-hours. The cost of that is a failure mode with no symptom: if two cells
that differ share a key, the second is skipped as already done and the command exits
successfully having measured nothing. It is not a crash, not a warning, and not
visible in the results — the row from the *first* cell is simply read as if it were
the second's.

That happened: a fixed-voter control was pointed at the directory holding an
auto-sized sweep, and all of its cells were skipped, because the key carried the
budget and the seed but neither the voter count nor the answer length.
"""

from __future__ import annotations

import pytest

from dp_rag_eval.core.types import PrivacyTarget
from dp_rag_eval.runner.sweep import CheckpointedRun, PipelineParams, cell_config_suffix


def target(epsilon: float = 5.0, n_tokens: int = 32) -> PrivacyTarget:
    return PrivacyTarget(epsilon=epsilon, delta=1e-6, n_tokens=n_tokens)


def test_answer_length_separates_cells() -> None:
    """The same budget over 8 and over 32 tokens are different guarantees."""
    assert target(5.0, n_tokens=8).label() != target(5.0, n_tokens=32).label()


def test_budget_still_separates_cells() -> None:
    assert target(5.0).label() != target(8.0).label()
    assert target(float("inf")).label() != target(32.0).label()


def test_voter_count_separates_koga_cells() -> None:
    """The regression: auto-sized and fixed-count runs at one budget are two systems."""
    auto = PipelineParams(n_split=20, auto_n_split_cap=128)
    fixed = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_fixed=200)
    assert cell_config_suffix("koga", target(), auto) != cell_config_suffix("koga", target(), fixed)


def test_headroom_separates_koga_cells() -> None:
    plain = PipelineParams(n_split=20, auto_n_split_cap=1024, n_split_headroom=1.0)
    roomy = PipelineParams(n_split=20, auto_n_split_cap=1024, n_split_headroom=2.0)
    assert cell_config_suffix("koga", target(), plain) != cell_config_suffix("koga", target(), roomy)


def test_other_pipelines_keep_their_existing_keys() -> None:
    """Grislain and naive have no voter count, so their keys must not gain a suffix.

    Changing them would strand every checkpoint already written for those pipelines.
    """
    params = PipelineParams(n_split=20, auto_n_split_cap=128, n_split_fixed=200)
    assert cell_config_suffix("grislain", target(), params) == ""
    assert cell_config_suffix("naive", target(), params) == ""


def test_identical_configurations_still_share_a_key() -> None:
    """Resumability is the point of the key; it must survive the fix."""
    a = PipelineParams(n_split=20, auto_n_split_cap=128)
    b = PipelineParams(n_split=20, auto_n_split_cap=128)
    assert cell_config_suffix("koga", target(), a) == cell_config_suffix("koga", target(), b)
    assert target(5.0).label() == target(5.0).label()


# ---------------------------------------------------------------------------
# The model is not in the key, so it is guarded instead
# ---------------------------------------------------------------------------
#
# A second generator (Gemma 4, beside Qwen3.5) produces the *same* key for the same
# configuration. Adding the model to the key would fix that and invalidate every key
# already written, re-running finished cells. So the key stays as it is and the
# checkpoint refuses to mix two models in one directory.


def run_dir(tmp_path, generator: str):
    return CheckpointedRun(tmp_path, {"generator": generator}, ["pipeline", "generator", "status"])


def test_resuming_the_same_generator_is_allowed(tmp_path) -> None:
    first = run_dir(tmp_path, "Qwen/Qwen3.5-9B")
    first.record({"cell": "grislain|unified|original|-|eps=1,n_tokens=32|seed=42", "generator": "Qwen/Qwen3.5-9B"})

    resumed = run_dir(tmp_path, "Qwen/Qwen3.5-9B")
    assert resumed.is_done("grislain|unified|original|-|eps=1,n_tokens=32|seed=42")


def test_a_second_generator_is_refused(tmp_path) -> None:
    """Without this the run exits 0 having skipped every cell as already done."""
    first = run_dir(tmp_path, "Qwen/Qwen3.5-9B")
    first.record({"cell": "grislain|unified|original|-|eps=1,n_tokens=32|seed=42", "generator": "Qwen/Qwen3.5-9B"})

    with pytest.raises(RuntimeError, match="own --output-dir"):
        run_dir(tmp_path, "google/gemma-4-12B-it")


def test_the_refusal_names_both_models(tmp_path) -> None:
    first = run_dir(tmp_path, "Qwen/Qwen3.5-9B")
    first.record({"cell": "a", "generator": "Qwen/Qwen3.5-9B"})

    with pytest.raises(RuntimeError) as excinfo:
        run_dir(tmp_path, "google/gemma-4-12B-it")
    message = str(excinfo.value)
    assert "Qwen/Qwen3.5-9B" in message
    assert "google/gemma-4-12B-it" in message


def test_rows_written_before_the_column_existed_are_caught_by_meta(tmp_path) -> None:
    """Existing results files have no per-row generator — meta_history still names it."""
    first = run_dir(tmp_path, "Qwen/Qwen3.5-9B")
    first.record({"cell": "a"})  # no generator column, as older runs wrote it

    with pytest.raises(RuntimeError, match="Qwen/Qwen3.5-9B"):
        run_dir(tmp_path, "google/gemma-4-12B-it")


def test_a_directory_with_no_recorded_generator_still_resumes(tmp_path) -> None:
    """A checkpoint that predates the field must not become unreadable."""
    CheckpointedRun(tmp_path, {}, ["pipeline", "status"]).record({"cell": "a"})
    assert CheckpointedRun(tmp_path, {"generator": "google/gemma-4-12B-it"}, ["pipeline", "status"]).is_done("a")
