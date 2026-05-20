"""Experiment reporting: JSON logging and summary generation."""

from __future__ import annotations

import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)


class ExperimentReporter:
    """Logs experiment results to JSON files with hardware metadata.

    Args:
        output_dir: Directory for experiment outputs.
        experiment_name: Name of the experiment.
    """

    def __init__(self, output_dir: str = "runs/", experiment_name: str = "experiment") -> None:
        self._output_dir = Path(output_dir) / experiment_name
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._results: list[dict[str, Any]] = []
        self._meta = self._collect_metadata()

    @staticmethod
    def _collect_metadata() -> dict[str, Any]:
        """Collect hardware and software metadata."""
        meta: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
            meta["cuda_version"] = torch.version.cuda
        return meta

    def log_run(
        self,
        pipeline_type: str,
        epsilon: float,
        delta: float,
        seed: int,
        dataset_variant: str,
        utility_metrics: dict[str, float],
        attack_results: list[dict[str, Any]] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Log a single experiment run.

        Args:
            pipeline_type: Pipeline used ("koga" or "grislain").
            epsilon: Privacy budget.
            delta: Privacy delta.
            seed: Random seed.
            dataset_variant: Dataset variant name.
            utility_metrics: Dict of metric_name → value.
            attack_results: List of attack result dicts.
            extra: Any additional metadata to log.
        """
        run_result: dict[str, Any] = {
            "pipeline": pipeline_type,
            "epsilon": epsilon,
            "delta": delta,
            "seed": seed,
            "dataset_variant": dataset_variant,
            "utility": utility_metrics,
            "attacks": attack_results or [],
        }
        if extra:
            run_result["extra"] = extra
        self._results.append(run_result)

        # Also write immediately to a per-run file
        run_file = self._output_dir / f"run_eps{epsilon}_seed{seed}.json"
        with open(run_file, "w", encoding="utf-8") as f:
            json.dump(run_result, f, indent=2, default=str)
        logger.info("Logged run to %s", run_file)

    def save_summary(self) -> Path:
        """Save a summary of all runs to a JSON file.

        Returns:
            Path to the summary file.
        """
        summary = {
            "metadata": self._meta,
            "n_runs": len(self._results),
            "runs": self._results,
        }
        summary_path = self._output_dir / "summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Saved experiment summary to %s (%d runs)", summary_path, len(self._results))
        return summary_path

    def save_metadata(self) -> Path:
        """Save hardware/software metadata to meta.json.

        Returns:
            Path to the metadata file.
        """
        meta_path = self._output_dir / "meta.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self._meta, f, indent=2)
        return meta_path
