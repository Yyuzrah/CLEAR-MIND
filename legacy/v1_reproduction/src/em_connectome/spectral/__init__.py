"""Spectrum extraction interfaces and Arakelov–Green implementation."""

from .arakelov_green import ArakelovGreenSpectrum
from .base import (
    SpectrumExtractor,
    SpectrumParameters,
    create_spectrum_extractor,
    spectrum_cache_key,
)
from .cache import SpectrumBundle, build_or_load_spectra, select_smoke_entries

__all__ = [
    "ArakelovGreenSpectrum",
    "SpectrumBundle",
    "SpectrumExtractor",
    "SpectrumParameters",
    "build_or_load_spectra",
    "create_spectrum_extractor",
    "select_smoke_entries",
    "spectrum_cache_key",
]
