"""Manifest-aligned local spectrum cache."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from em_connectome.data.manifest import ManifestEntry
from em_connectome.data.records import load_neuron_record

from .arakelov_green import ArakelovGreenSpectrum
from .base import SpectrumParameters, spectrum_cache_key


@dataclass(frozen=True)
class SpectrumBundle:
    entries: list[ManifestEntry]
    spectra: NDArray[np.float64]
    cache_path: Path
    cache_hit: bool
    feature_set_hash: str


def select_smoke_entries(
    entries: Iterable[ManifestEntry],
    *,
    per_class_per_split: int = 2,
) -> list[ManifestEntry]:
    """Select a deterministic class-balanced subset in manifest order."""

    if per_class_per_split <= 0:
        raise ValueError("per_class_per_split must be positive")
    rows = list(entries)
    selected: list[ManifestEntry] = []
    counts: dict[tuple[str, int], int] = {}
    for entry in rows:
        group = (entry.phase, entry.label_id)
        if counts.get(group, 0) >= per_class_per_split:
            continue
        selected.append(entry)
        counts[group] = counts.get(group, 0) + 1
    expected_groups = {
        (phase, label)
        for phase in ("train", "test")
        for label in {entry.label_id for entry in rows}
    }
    missing = expected_groups.difference(counts)
    if missing:
        raise ValueError(f"smoke subset is missing split/label groups: {sorted(missing)}")
    return selected


def _feature_set_hash(cache_keys: list[str], sample_ids: list[str]) -> str:
    payload = json.dumps(
        {"cache_keys": cache_keys, "sample_ids": sample_ids},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_or_load_spectra(
    *,
    data_root: Path,
    entries: Iterable[ManifestEntry],
    manifest_sha256: str,
    cache_root: Path,
    parameters: SpectrumParameters,
    force: bool = False,
) -> SpectrumBundle:
    rows = list(entries)
    if not rows:
        raise ValueError("cannot build spectra for an empty manifest selection")
    extractor = ArakelovGreenSpectrum(parameters)
    cache_keys = [
        spectrum_cache_key(
            swc_sha256=row.content_sha256,
            manifest_sha256=manifest_sha256,
            parameters=parameters,
            implementation_version=extractor.implementation_version,
        )
        for row in rows
    ]
    sample_ids = [row.sample_id for row in rows]
    set_hash = _feature_set_hash(cache_keys, sample_ids)
    dataset = rows[0].dataset
    if any(row.dataset != dataset for row in rows):
        raise ValueError("a spectrum bundle cannot mix datasets")
    cache_directory = cache_root.resolve() / "spectral" / dataset
    cache_directory.mkdir(parents=True, exist_ok=True)
    cache_path = cache_directory / f"{set_hash}.npz"
    metadata_path = cache_directory / f"{set_hash}.json"

    if cache_path.is_file() and metadata_path.is_file() and not force:
        with np.load(cache_path, allow_pickle=False) as payload:
            spectra = np.asarray(payload["spectra"], dtype=np.float64)
            cached_sample_ids = payload["sample_ids"].astype(str).tolist()
            cached_content_hashes = payload["content_sha256"].astype(str).tolist()
            cached_keys = payload["cache_keys"].astype(str).tolist()
        expected_shape = (len(rows), parameters.top_k)
        if spectra.shape != expected_shape:
            raise ValueError(
                f"malformed spectrum cache {cache_path}: "
                f"expected {expected_shape}, found {spectra.shape}"
            )
        if cached_sample_ids != sample_ids:
            raise ValueError(f"sample alignment mismatch in {cache_path}")
        if cached_content_hashes != [row.content_sha256 for row in rows]:
            raise ValueError(f"content-hash alignment mismatch in {cache_path}")
        if cached_keys != cache_keys:
            raise ValueError(f"cache-key alignment mismatch in {cache_path}")
        return SpectrumBundle(rows, spectra, cache_path, True, set_hash)

    spectra_rows: list[NDArray[np.float64]] = []
    for row in rows:
        record = load_neuron_record(data_root, row)
        spectrum = extractor.extract(record)
        if spectrum.shape != (parameters.top_k,):
            raise ValueError(f"invalid spectrum shape for {row.sample_id}: {spectrum.shape}")
        if not np.isfinite(spectrum).all() or float(spectrum.sum()) == 0.0:
            raise ValueError(f"invalid spectrum for {row.sample_id}")
        spectra_rows.append(spectrum)
    spectra = np.asarray(spectra_rows, dtype=np.float64)
    np.savez_compressed(
        cache_path,
        spectra=spectra,
        sample_ids=np.asarray(sample_ids),
        content_sha256=np.asarray([row.content_sha256 for row in rows]),
        cache_keys=np.asarray(cache_keys),
    )
    metadata = {
        "schema_version": 1,
        "dataset": dataset,
        "rows": len(rows),
        "feature_set_hash": set_hash,
        "manifest_sha256": manifest_sha256,
        "method": parameters.method,
        "parameters": parameters.canonical_payload(),
        "implementation_version": extractor.implementation_version,
        "cache_path": cache_path.name,
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return SpectrumBundle(rows, spectra, cache_path, False, set_hash)
