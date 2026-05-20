"""CLI script to run a single experiment from a YAML config file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dp_rag_eval.core.config import FullExperimentConfig
from dp_rag_eval.llm.base import set_seed
from dp_rag_eval.pipelines import create_pipeline
from dp_rag_eval.runner.experiment import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single dp-rag-eval experiment")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--privacy.epsilon", type=float, default=None, dest="override_epsilon")
    parser.add_argument("--experiment.seed", type=int, default=None, dest="override_seed")
    return parser.parse_args()


def load_config(config_path: str, overrides: dict) -> FullExperimentConfig:
    """Load and validate config from YAML, applying CLI overrides."""
    with open(config_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if overrides.get("epsilon") is not None:
        raw.setdefault("privacy", {})["epsilon"] = overrides["epsilon"]
    if overrides.get("seed") is not None:
        raw.setdefault("experiment", {})["seed"] = overrides["seed"]

    return FullExperimentConfig(**raw)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()

    overrides = {
        "epsilon": args.override_epsilon,
        "seed": args.override_seed,
    }
    config = load_config(args.config, overrides)
    logging.info("Loaded config: %s", config.experiment.name)

    set_seed(config.experiment.seed)

    # Create LLM backend
    if config.pipeline.type == "koga":
        from dp_rag_eval.llm.local_vllm import LocalVLLMBackend

        llm_backend = LocalVLLMBackend(
            model_name=config.pipeline.model,
            device=config.pipeline.device,
            model_revision=config.pipeline.model_revision,
        )
    else:
        from dp_rag_eval.llm.local_hf import LocalHFBackend

        llm_backend = LocalHFBackend(
            model_name=config.pipeline.model,
            device=config.pipeline.device,
            model_revision=config.pipeline.model_revision,
        )

    pipeline = create_pipeline(config.pipeline, llm_backend)

    runner = ExperimentRunner(config=config, pipeline=pipeline)
    documents = runner.setup()
    results = runner.run(documents)
    logging.info("Experiment finished. Results: %s", results["summary_path"])


if __name__ == "__main__":
    main()
