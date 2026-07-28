"""Sweep plumbing shared by the experiment scripts.

Both ``scripts/run_attack.py`` (membership inference) and ``scripts/run_canary.py``
(canary extraction) walk the same grid — seeds x dataset variants x epsilon x
pipeline — so the grid expansion, pipeline construction and resumable result
writing live here rather than being duplicated per script.
"""

from dp_rag_eval.runner.sweep import (
    CheckpointedRun,
    PipelineParams,
    build_pipeline,
    make_variant,
    param_str,
    parse_floats,
    parse_ints,
    parse_strs,
    split_members,
    variant_configs,
)

__all__ = [
    "CheckpointedRun",
    "PipelineParams",
    "build_pipeline",
    "make_variant",
    "param_str",
    "parse_floats",
    "parse_ints",
    "parse_strs",
    "split_members",
    "variant_configs",
]
