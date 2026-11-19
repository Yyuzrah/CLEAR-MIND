"""Extension interface and provenance helpers for spectrum extractors."""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
from numpy.typing import NDArray

from em_connectome.data.records import NeuronRecord


@dataclass(frozen=True)
class SpectrumParameters:
    """Parameters fixed by the version-one result notebooks."""

    method: str = "arakelov_green"
    epsilon: float = 20.0
    noise_threshold: float = 5.0
    top_k: int = 64

    def __post_init__(self) -> None:
        if self.method != "arakelov_green":
            raise ValueError("SpectrumParameters only describes arakelov_green")
        if self.epsilon < 0:
            raise ValueError("epsilon must be non-negative")
        if self.noise_threshold < 0:
            raise ValueError("noise_threshold must be non-negative")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")

    def canonical_payload(self) -> dict[str, object]:
        return asdict(self)


class SpectrumExtractor(ABC):
    """Minimal plugin interface for present and future spectrum methods."""

    method: str
    implementation_version: str

    @abstractmethod
    def extract(self, record: NeuronRecord) -> NDArray[np.float64]:
        """Return one deterministic one-dimensional spectrum."""


ExtractorFactory = Callable[[SpectrumParameters], SpectrumExtractor]
_REGISTRY: dict[str, ExtractorFactory] = {}


def register_spectrum_method(name: str, factory: ExtractorFactory) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Spectrum method already registered: {name}")
    _REGISTRY[name] = factory


def create_spectrum_extractor(parameters: SpectrumParameters) -> SpectrumExtractor:
    try:
        factory = _REGISTRY[parameters.method]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "<none>"
        raise KeyError(
            f"Unknown spectrum method {parameters.method!r}; registered: {available}"
        ) from exc
    return factory(parameters)


def spectrum_cache_key(
    *,
    swc_sha256: str,
    manifest_sha256: str,
    parameters: SpectrumParameters,
    implementation_version: str,
) -> str:
    """Hash every input that can change a cached version-one spectrum."""

    payload = {
        "swc_sha256": swc_sha256,
        "manifest_sha256": manifest_sha256,
        "method": parameters.method,
        "parameters": parameters.canonical_payload(),
        "implementation_version": implementation_version,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()
