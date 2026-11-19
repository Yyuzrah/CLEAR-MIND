"""Canonical sample object passed from the data layer to every adapter."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .manifest import ManifestEntry, resolve_data_path
from .swc import SWCTree, parse_swc


@dataclass(frozen=True)
class NeuronRecord:
    sample_id: str
    dataset: str
    swc_path: Path
    swc_sha256: str
    label_id: int
    label_name: str
    fold: int
    phase: str
    morphology: SWCTree
    spectrum: NDArray[np.float64] | None = None

    def with_spectrum(self, spectrum: NDArray[np.float64]) -> NeuronRecord:
        values = np.asarray(spectrum, dtype=np.float64)
        if values.ndim != 1:
            raise ValueError("spectrum must be a one-dimensional vector")
        return replace(self, spectrum=values)


def load_neuron_record(data_root: Path, entry: ManifestEntry) -> NeuronRecord:
    path = resolve_data_path(data_root, entry)
    return NeuronRecord(
        sample_id=entry.sample_id,
        dataset=entry.dataset,
        swc_path=path,
        swc_sha256=entry.content_sha256,
        label_id=entry.label_id,
        label_name=entry.label_name,
        fold=entry.fold,
        phase=entry.phase,
        morphology=parse_swc(path),
    )
