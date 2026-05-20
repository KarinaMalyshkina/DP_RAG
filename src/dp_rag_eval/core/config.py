"""Pydantic configuration models for experiment definitions."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ExperimentConfig(BaseModel):
    """Top-level experiment metadata."""

    name: str = "experiment"
    seed: int = 42
    output_dir: str = "runs/"


class KogaPipelineParams(BaseModel):
    """Koga-specific pipeline parameters."""

    n_split: int = 50
    n_docs: int = 1
    max_length: int = 20
    below_threshold: bool = False
    threshold: float = 0.5
    sparse_eps: float = 1.0
    fail_mode: str = "zero"


class GrislainPipelineParams(BaseModel):
    """Grislain-specific pipeline parameters."""

    alpha: float = 4.0
    omega: float = 1.0
    temperature: float = 1.0
    max_new_tokens: int = 50
    top_p: float | None = None
    top_k: int | None = 10
    top_p_alpha: float = 4.0
    embedding_model: str = "Snowflake/snowflake-arctic-embed-m-v1.5"


class PipelineConfig(BaseModel):
    """Pipeline configuration."""

    type: Literal["koga", "grislain"] = "koga"
    model: str = "facebook/opt-1.3b"
    model_revision: str | None = None
    device: str = "cuda"
    koga: KogaPipelineParams = Field(default_factory=KogaPipelineParams)
    grislain: GrislainPipelineParams = Field(default_factory=GrislainPipelineParams)


class PrivacyConfig(BaseModel):
    """Privacy budget configuration."""

    epsilon: float | list[float] = 1.0
    delta: float = 1e-5


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    loader: str = "nq"
    variant: str = "original"
    variant_params: dict[str, Any] = Field(default_factory=dict)
    max_docs: int | None = None


class AttackConfig(BaseModel):
    """Single attack configuration."""

    type: str
    n_canaries: int = 100
    n_queries: int = 500
    n_members: int = 200
    n_non_members: int = 200
    canary_format: str = "The secret code is {secret}."
    copies_per_canary: int = 1


class MetricsConfig(BaseModel):
    """Metrics to compute."""

    utility: list[str] = Field(default_factory=lambda: ["exact_match", "f1"])
    privacy: list[str] = Field(default_factory=lambda: ["attack_success_rate", "tpr_at_fpr_01"])


class FullExperimentConfig(BaseModel):
    """Complete experiment configuration validated from YAML."""

    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    attacks: list[AttackConfig] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
