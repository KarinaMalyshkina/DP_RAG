"""Experiment runner: orchestrates pipeline evaluation across configurations."""

from __future__ import annotations

import logging
from typing import Any

from dp_rag_eval.attacks.canary import CanaryExtractionAttack
from dp_rag_eval.attacks.membership import MembershipInferenceAttack
from dp_rag_eval.attacks.property import PropertyInferenceAttack
from dp_rag_eval.core.config import AttackConfig, FullExperimentConfig
from dp_rag_eval.core.types import Document, Query
from dp_rag_eval.datasets.loaders import create_loader
from dp_rag_eval.datasets.variants import create_variant
from dp_rag_eval.evaluation.metrics import compute_utility
from dp_rag_eval.evaluation.reporter import ExperimentReporter
from dp_rag_eval.llm.base import set_seed

logger = logging.getLogger(__name__)


def _create_attack(config: AttackConfig) -> Any:
    """Create an attack instance from configuration."""
    if config.type == "canary_extraction":
        return CanaryExtractionAttack()
    elif config.type == "membership_inference":
        return MembershipInferenceAttack(
            n_members=config.n_members,
            n_non_members=config.n_non_members,
        )
    elif config.type == "property_inference":
        return PropertyInferenceAttack()
    else:
        raise ValueError(f"Unknown attack type: {config.type!r}")


class ExperimentRunner:
    """Orchestrates experiment runs across epsilon values, seeds, and attacks.

    Args:
        config: Full experiment configuration.
        pipeline: An initialized DP-RAG pipeline (already created externally).
        queries: Optional list of evaluation queries.
    """

    def __init__(
        self,
        config: FullExperimentConfig,
        pipeline: Any,
        queries: list[Query] | None = None,
    ) -> None:
        self._config = config
        self._pipeline = pipeline
        self._queries = queries or []
        self._reporter = ExperimentReporter(
            output_dir=config.experiment.output_dir,
            experiment_name=config.experiment.name,
        )

    def setup(self) -> list[Document]:
        """Load dataset and apply variant.

        Returns:
            The processed document corpus.
        """
        set_seed(self._config.experiment.seed)

        # Load base dataset
        loader = create_loader(
            self._config.dataset.loader,
            max_docs=self._config.dataset.max_docs,
        )
        documents = loader.load()
        logger.info("Loaded %d base documents", len(documents))

        # Apply variant
        variant = create_variant(
            self._config.dataset.variant,
            self._config.dataset.variant_params,
        )
        documents = variant.apply(documents)
        logger.info("After variant '%s': %d documents", self._config.dataset.variant, len(documents))

        # Index in pipeline
        self._pipeline.index(documents)

        return documents

    def run(self, documents: list[Document]) -> dict[str, Any]:
        """Run the full experiment.

        For each epsilon value:
        1. Reset the pipeline
        2. Evaluate utility on queries
        3. Run configured attacks
        4. Log results

        Args:
            documents: The processed corpus (from setup()).

        Returns:
            Summary dict with all results.
        """
        epsilon_values = self._config.privacy.epsilon
        if isinstance(epsilon_values, float):
            epsilon_values = [epsilon_values]

        delta = self._config.privacy.delta
        all_results: list[dict[str, Any]] = []

        for eps in epsilon_values:
            logger.info("=== Running experiment with ε=%.4f, δ=%.2e ===", eps, delta)
            self._pipeline.reset()

            # Evaluate utility
            utility_metrics = self._evaluate_utility(eps, delta)

            # Run attacks
            attack_results = []
            for attack_config in self._config.attacks:
                self._pipeline.reset()
                attack = _create_attack(attack_config)
                attack.setup(self._pipeline, documents)
                result = attack.run(attack_config.n_queries, eps, delta)
                attack_results.append(
                    {
                        "type": result.attack_type,
                        "success_rate": result.success_rate,
                        "n_queries": result.n_queries,
                        "n_successes": result.n_successes,
                        "metadata": result.metadata,
                    }
                )
                logger.info(
                    "Attack %s: success_rate=%.4f (%d/%d)",
                    result.attack_type,
                    result.success_rate,
                    result.n_successes,
                    result.n_queries,
                )

            # Log
            self._reporter.log_run(
                pipeline_type=self._config.pipeline.type,
                epsilon=eps,
                delta=delta,
                seed=self._config.experiment.seed,
                dataset_variant=self._config.dataset.variant,
                utility_metrics=utility_metrics,
                attack_results=attack_results,
            )

            all_results.append(
                {
                    "epsilon": eps,
                    "utility": utility_metrics,
                    "attacks": attack_results,
                }
            )

        # Save summary
        self._reporter.save_metadata()
        summary_path = self._reporter.save_summary()
        logger.info("Experiment complete. Summary: %s", summary_path)

        return {"results": all_results, "summary_path": str(summary_path)}

    def _evaluate_utility(self, eps: float, delta: float) -> dict[str, float]:
        """Evaluate utility metrics on the query set.

        Args:
            eps: Privacy budget per query.
            delta: Privacy delta.

        Returns:
            Dict of metric_name → average score.
        """
        if not self._queries:
            return {}

        predictions: list[str] = []
        references: list[list[str]] = []

        for query in self._queries:
            result = self._pipeline.answer(query.text, eps, delta)
            predictions.append(result.text)
            references.append(query.gold_answers)

        metrics: dict[str, float] = {}
        for metric_name in self._config.metrics.utility:
            metrics[metric_name] = compute_utility(predictions, references, metric_name)
            logger.info("Utility %s: %.4f", metric_name, metrics[metric_name])

        return metrics
