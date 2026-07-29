"""Sweep plumbing shared by the experiment scripts.

Both ``scripts/run_attack.py`` (membership inference) and ``scripts/run_canary.py``
(canary extraction) walk the same grid — seeds x dataset variants x epsilon x
pipeline — so the grid expansion, pipeline construction and resumable result
writing live here rather than being duplicated per script.
"""

from dp_rag_eval.runner.sweep import (
    VARIANT_COLUMNS,
    CheckpointedRun,
    PipelineParams,
    VariantCell,
    VariantGrid,
    build_pipeline,
    grid_from_args,
    make_variant,
    make_variant_cell,
    param_str,
    parse_floats,
    parse_ints,
    parse_strs,
    split_members,
    variant_configs,
)

__all__ = [
    "VARIANT_COLUMNS",
    "CheckpointedRun",
    "PipelineParams",
    "VariantCell",
    "VariantGrid",
    "build_pipeline",
    "grid_from_args",
    "make_variant",
    "make_variant_cell",
    "param_str",
    "parse_floats",
    "parse_ints",
    "parse_strs",
    "split_members",
    "variant_configs",
]
