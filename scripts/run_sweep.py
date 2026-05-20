"""CLI script to run a grid sweep over epsilon values and seeds."""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import logging
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dp_rag_eval.core.config import FullExperimentConfig
from dp_rag_eval.llm.base import set_seed
from dp_rag_eval.pipelines import create_pipeline
from dp_rag_eval.runner.experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a sweep of dp-rag-eval experiments")
    parser.add_argument("--config", type=str, required=True, help="Path to base YAML config file")
    parser.add_argument(
        "--sweep",
        type=str,
        action="append",
        default=[],
        help="Sweep parameter in format 'key=values', e.g., 'privacy.epsilon=[0.5,1,2,4,8]'",
    )
    return parser.parse_args()


def parse_sweep_params(sweep_args: list[str]) -> dict[str, list]:
    """Parse sweep arguments into a dict of parameter paths to value lists."""
    params: dict[str, list] = {}
    for arg in sweep_args:
        key, value_str = arg.split("=", 1)
        values = ast.literal_eval(value_str)
        if not isinstance(values, list):
            values = [values]
        params[key] = values
    return params


def set_nested(d: dict, key_path: str, value: object) -> None:
    """Set a nested dict value using dot-separated key path."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        base_raw = yaml.safe_load(f)

    sweep_params = parse_sweep_params(args.sweep)
    logging.info("Sweep parameters: %s", sweep_params)

    keys = list(sweep_params.keys())
    value_lists = [sweep_params[k] for k in keys]
    combinations = list(itertools.product(*value_lists))

    logging.info("Running %d experiment configurations", len(combinations))

    for combo_idx, combo in enumerate(combinations):
        raw = copy.deepcopy(base_raw)
        for key, value in zip(keys, combo):
            set_nested(raw, key, value)

        config = FullExperimentConfig(**raw)
        config.experiment.name = f"{config.experiment.name}_combo{combo_idx}"

        logging.info(
            "=== Sweep %d/%d: %s ===",
            combo_idx + 1,
            len(combinations),
            dict(zip(keys, combo)),
        )

        set_seed(config.experiment.seed)

        # Create LLM backend
        if config.pipeline.type == "koga":
            from dp_rag_eval.llm.local_vllm import LocalVLLMBackend

            llm_backend = LocalVLLMBackend(
                model_name=config.pipeline.model,
                device=config.pipeline.device,
            )
        else:
            from dp_rag_eval.llm.local_hf import LocalHFBackend

            llm_backend = LocalHFBackend(
                model_name=config.pipeline.model,
                device=config.pipeline.device,
            )

        pipeline = create_pipeline(config.pipeline, llm_backend)
        runner = ExperimentRunner(config=config, pipeline=pipeline)
        documents = runner.setup()
        results = runner.run(documents)
        logging.info("Sweep %d/%d complete: %s", combo_idx + 1, len(combinations), results["summary_path"])

    logging.info("All sweep runs complete.")


if __name__ == "__main__":
    main()
