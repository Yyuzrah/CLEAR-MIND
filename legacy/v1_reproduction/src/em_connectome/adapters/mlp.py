"""64-dimensional spectrum adapter used by the fixed MLP."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from em_connectome.data.records import NeuronRecord


@dataclass(frozen=True)
class MLPSample:
    sample_id: str
    features: NDArray[np.float64]
    label: int


@dataclass(frozen=True)
class SpectrumStandardizer:
    """Train-only per-dimension normalization from the MLP result notebook."""

    mean: NDArray[np.float64]
    std: NDArray[np.float64]

    @classmethod
    def fit(cls, features: NDArray[np.float64]) -> SpectrumStandardizer:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or len(matrix) == 0:
            raise ValueError("training features must be a non-empty matrix")
        mean = matrix.mean(axis=0, keepdims=True)
        std = matrix.std(axis=0, keepdims=True)
        std[std < 1e-8] = 1.0
        return cls(mean=mean, std=std)

    def transform(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        matrix = np.asarray(features, dtype=np.float64)
        return (matrix - self.mean) / self.std


class MLPAdapter:
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim

    def transform(self, record: NeuronRecord) -> MLPSample:
        if record.spectrum is None:
            raise ValueError(f"{record.sample_id} has no spectrum")
        spectrum = np.asarray(record.spectrum, dtype=np.float64)
        if spectrum.shape != (self.feature_dim,):
            raise ValueError(
                f"{record.sample_id} spectrum has shape {spectrum.shape}; "
                f"expected ({self.feature_dim},)"
            )
        if not np.isfinite(spectrum).all():
            raise ValueError(f"{record.sample_id} spectrum contains a non-finite value")
        return MLPSample(
            sample_id=record.sample_id,
            features=spectrum.copy(),
            label=record.label_id,
        )

    def transform_many(self, records: Iterable[NeuronRecord]) -> list[MLPSample]:
        return [self.transform(record) for record in records]
