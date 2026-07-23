"""Strict, typed YAML configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from embedding_model.exceptions import ConfigurationError

PoolingName = Literal["mean", "cls", "max", "mean_sqrt_length"]
ObjectiveName = Literal[
    "multiple_negatives_ranking",
    "info_nce",
    "triplet",
    "cosine_regression",
    "distillation",
]


class StrictModel(BaseModel):
    """Reject unknown configuration keys to catch misspellings early."""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelConfig(StrictModel):
    """Architecture and inference contract for the local transformer."""

    encoder_type: Literal["local"] = "local"
    vocabulary_size: int = Field(default=4096, ge=8)
    pad_token_id: int = Field(default=0, ge=0)
    hidden_size: int = Field(default=128, gt=0)
    embedding_dimension: int = Field(default=128, gt=0)
    num_attention_heads: int = Field(default=4, gt=0)
    num_hidden_layers: int = Field(default=2, ge=1)
    intermediate_size: int = Field(default=256, gt=0)
    max_sequence_length: int = Field(default=128, ge=2, le=8192)
    pooling: PoolingName = "mean"
    normalize_embeddings: bool = True
    projection: bool = True
    layer_normalization: bool = False
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_architecture(self) -> ModelConfig:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.pad_token_id >= self.vocabulary_size:
            raise ValueError("pad_token_id must be smaller than vocabulary_size")
        if not self.projection and self.embedding_dimension != self.hidden_size:
            raise ValueError(
                "embedding_dimension must equal hidden_size when projection is disabled"
            )
        return self


class TrainingConfig(StrictModel):
    """Optimization, checkpoint, and reproducibility settings."""

    objective: ObjectiveName = "multiple_negatives_ranking"
    epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=32, ge=2)
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    learning_rate: float = Field(default=2e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    warmup_ratio: float = Field(default=0.1, ge=0.0, lt=1.0)
    max_gradient_norm: float = Field(default=1.0, gt=0.0)
    temperature: float = Field(default=0.05, gt=0.0)
    margin: float = Field(default=0.2, ge=0.0)
    mixed_precision: bool = False
    device: Literal["auto", "cpu", "cuda"] = "auto"
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    checkpoint_dir: Path = Path("artifacts/checkpoints")
    early_stopping_patience: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_hardware(self) -> TrainingConfig:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("device='cuda' was requested but CUDA is unavailable")
        if self.mixed_precision and (
            self.device == "cpu" or (self.device == "auto" and not torch.cuda.is_available())
        ):
            raise ValueError("mixed_precision requires an available CUDA device")
        return self


class ProjectConfig(StrictModel):
    """Top-level train configuration."""

    model: ModelConfig
    training: TrainingConfig


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration and surface actionable validation errors."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise ConfigurationError(f"configuration file does not exist: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ConfigurationError("configuration root must be a mapping")
        return ProjectConfig.model_validate(raw)
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"invalid YAML in {config_path}: {exc}") from exc
    except ValueError as exc:
        raise ConfigurationError(f"invalid configuration in {config_path}: {exc}") from exc
