"""YAML configuration loading, validation, resolution, and hashing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

FORMAL_SEEDS = (42, 1453, 666, 114514, 1919810)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SourceReference(StrictModel):
    path: str
    sha256: str
    cell: int | None = None
    role: str


class DatasetConfig(StrictModel):
    key: Literal["act4", "jml4", "bil6"]
    display_name: str
    manifest: str
    classes: tuple[str, ...]


class SpectralConfig(StrictModel):
    method: Literal["arakelov_green"]
    epsilon: float = Field(ge=0)
    noise_threshold: float = Field(ge=0)
    top_k: int = Field(gt=0)
    implementation_version: str
    source: SourceReference


class ModelConfig(StrictModel):
    name: Literal["gnn_alpha", "treelstm_arch3", "mlp"]
    parameters: dict[str, object]
    sources: tuple[SourceReference, ...]


class TrainerConfig(StrictModel):
    epochs: int = Field(gt=0)
    patience: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    learning_rate: float = Field(gt=0)
    weight_decay: float = Field(ge=0)
    optimizer: Literal["adam", "adamw"]
    scheduler: Literal["none", "cosine"]
    eta_min: float = Field(default=0.0, ge=0)
    gradient_clip: float | None = Field(default=None, gt=0)
    jitter_std: float = Field(default=0.0, ge=0)
    jitter_clip: float = Field(default=0.005, ge=0)
    label_smoothing: float = Field(default=0.0, ge=0, lt=1)
    num_workers: int = Field(default=0, ge=0)
    selection_rule: Literal["accuracy", "accuracy_macro_f1_loss"]


class ExperimentConfig(StrictModel):
    schema_version: Literal[1]
    dataset: Literal["act4", "jml4", "bil6"]
    spectral: Literal["arakelov_green"]
    model: Literal["gnn_alpha", "treelstm_arch3", "mlp"]
    trainer: TrainerConfig
    seeds: tuple[int, ...] = FORMAL_SEEDS

    @model_validator(mode="after")
    def validate_formal_seeds(self) -> ExperimentConfig:
        if self.seeds != FORMAL_SEEDS:
            raise ValueError(f"formal seeds must be exactly {FORMAL_SEEDS}")
        return self


class ResolvedExperiment(StrictModel):
    schema_version: Literal[1]
    dataset: DatasetConfig
    spectral: SpectralConfig
    model: ModelConfig
    trainer: TrainerConfig
    seeds: tuple[int, ...]
    source_hashes: dict[str, str]

    @property
    def config_hash(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def find_config_root(experiment_path: Path) -> Path:
    for candidate in (experiment_path.parent, *experiment_path.parents):
        if candidate.name == "configs":
            return candidate
    raise ValueError(f"{experiment_path} is not inside a configs directory")


def resolve_experiment(path: Path) -> ResolvedExperiment:
    experiment_path = path.resolve()
    root = find_config_root(experiment_path)
    raw_experiment = ExperimentConfig.model_validate(_load_yaml(experiment_path))
    dataset_path = root / "datasets" / f"{raw_experiment.dataset}.yaml"
    spectral_path = root / "spectral" / f"{raw_experiment.spectral}.yaml"
    model_path = root / "models" / f"{raw_experiment.model}.yaml"

    resolved = ResolvedExperiment(
        schema_version=1,
        dataset=DatasetConfig.model_validate(_load_yaml(dataset_path)),
        spectral=SpectralConfig.model_validate(_load_yaml(spectral_path)),
        model=ModelConfig.model_validate(_load_yaml(model_path)),
        trainer=raw_experiment.trainer,
        seeds=raw_experiment.seeds,
        source_hashes={
            str(candidate.relative_to(root.parent).as_posix()): sha256_path(candidate)
            for candidate in (
                experiment_path,
                dataset_path,
                spectral_path,
                model_path,
            )
        },
    )
    if resolved.dataset.key != raw_experiment.dataset:
        raise ValueError("dataset component key does not match the experiment")
    if resolved.spectral.method != raw_experiment.spectral:
        raise ValueError("spectral component method does not match the experiment")
    if resolved.model.name != raw_experiment.model:
        raise ValueError("model component name does not match the experiment")
    return resolved


def resolved_yaml(config: ResolvedExperiment) -> str:
    payload = config.model_dump(mode="json")
    payload["config_hash"] = config.config_hash
    return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
