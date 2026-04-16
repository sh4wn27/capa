"""Pydantic settings, hyperparameters, and project-wide paths."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Project root — everything else is relative to this
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent


class DataConfig(BaseSettings):
    """Paths and parameters for the data pipeline."""

    model_config = SettingsConfigDict(env_prefix="CAPA_DATA_")

    raw_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    bmt_filename: str = Field(default="bone-marrow.csv")
    hla_sequences_filename: str = Field(default="hla_sequences.json")

    val_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    test_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    random_seed: int = Field(default=42)

    @property
    def bmt_path(self) -> Path:
        """Full path to the UCI BMT CSV file."""
        return self.raw_dir / self.bmt_filename

    @property
    def hla_sequences_path(self) -> Path:
        """Full path to the HLA sequences JSON file."""
        return self.raw_dir / self.hla_sequences_filename


class EmbeddingConfig(BaseSettings):
    """ESM-2 model and embedding cache configuration."""

    model_config = SettingsConfigDict(env_prefix="CAPA_EMBED_")

    esm_model_name: str = Field(default="facebook/esm2_t33_650M_UR50D")
    embedding_dim: int = Field(default=1280)
    cache_path: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "embeddings.h5")
    batch_size: int = Field(default=8)
    device: str = Field(default="cpu")  # override to "cuda" / "mps" as needed


class ModelConfig(BaseSettings):
    """Architecture hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="CAPA_MODEL_")

    # HLA loci to include
    hla_loci: list[str] = Field(default=["A", "B", "C", "DRB1", "DQB1"])

    # Cross-attention interaction network
    interaction_heads: int = Field(default=8)
    interaction_layers: int = Field(default=2)
    interaction_dim: int = Field(default=128)
    interaction_pos_embed: bool = Field(default=False)  # V2: learned per-locus positional embeddings
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)

    # V2: partial ESM-2 fine-tuning (0 = fully frozen)
    esm_finetune_layers: int = Field(default=0, ge=0)

    # Clinical covariate embedding
    clinical_dim: int = Field(default=32)

    # DeepHit survival head
    time_bins: int = Field(default=100)
    num_events: int = Field(default=3)  # GvHD, relapse, TRM


class TrainingConfig(BaseSettings):
    """Training loop hyperparameters."""

    model_config = SettingsConfigDict(env_prefix="CAPA_TRAIN_")

    learning_rate: float = Field(default=1e-4, gt=0.0)
    weight_decay: float = Field(default=1e-4, ge=0.0)
    batch_size: int = Field(default=32, gt=0)
    max_epochs: int = Field(default=200, gt=0)
    patience: int = Field(default=20, gt=0)      # early-stopping patience
    lr_patience: int = Field(default=10, gt=0)   # ReduceLROnPlateau patience
    lr_factor: float = Field(default=0.5, gt=0.0, lt=1.0)  # ReduceLROnPlateau decay
    random_seed: int = Field(default=42)

    # Gradient clipping
    max_grad_norm: float = Field(default=1.0, gt=0.0)

    # DeepHit loss weights
    alpha: float = Field(default=0.5, ge=0.0, le=1.0)  # ranking loss weight
    sigma: float = Field(default=0.1, gt=0.0)           # ranking loss bandwidth

    runs_dir: Path = Field(default=PROJECT_ROOT / "runs")


class CAPAConfig(BaseSettings):
    """Top-level config aggregating all sub-configs."""

    model_config = SettingsConfigDict(env_prefix="CAPA_")

    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)


def _flatten_yaml(d: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Recursively flatten a nested YAML dict into CAPA_* env-var style keys.

    Parameters
    ----------
    d : dict
        Nested dict from parsed YAML.
    prefix : str
        Running key prefix (e.g. ``"CAPA_MODEL_"``).

    Returns
    -------
    dict[str, str]
        Flat mapping ``{ENV_VAR_NAME: str_value}`` ready for ``os.environ``.
    """
    result: dict[str, str] = {}
    for k, v in d.items():
        full_key = f"{prefix}{k.upper()}"
        if isinstance(v, dict):
            result.update(_flatten_yaml(v, full_key + "_"))
        elif isinstance(v, list):
            # pydantic-settings reads lists from JSON-encoded env vars
            import json
            result[full_key] = json.dumps(v)
        else:
            result[full_key] = str(v)
    return result


def get_config(config_file: str | Path | None = None) -> CAPAConfig:
    """Return the global CAPA configuration.

    Priority (highest → lowest):
    1. Environment variables (``CAPA_*``)
    2. Values from *config_file* (YAML), if provided
    3. Pydantic field defaults

    Parameters
    ----------
    config_file : str or Path or None
        Path to a YAML config file (e.g. ``configs/default.yaml``).
        Values are injected as env vars *only if the env var is not already set*,
        so explicit env overrides always win.
    """
    if config_file is not None:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML config loading. "
                "Install it with: uv add pyyaml"
            ) from exc

        with open(config_file) as fh:
            raw: dict[str, Any] = yaml.safe_load(fh) or {}

        flat = _flatten_yaml(raw, prefix="CAPA_")
        for env_key, env_val in flat.items():
            if env_key not in os.environ:
                os.environ[env_key] = env_val
        logger.debug("Loaded config from %s (%d keys)", config_file, len(flat))

    return CAPAConfig()
