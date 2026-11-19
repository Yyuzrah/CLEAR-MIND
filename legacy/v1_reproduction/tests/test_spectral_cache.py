from __future__ import annotations

import hashlib
import shutil
from pathlib import Path

import numpy as np

from em_connectome.data.manifest import ManifestEntry
from em_connectome.spectral import SpectrumParameters, build_or_load_spectra

FIXTURE = Path(__file__).parent / "fixtures" / "simple.swc"


def test_spectrum_cache_is_manifest_and_sample_aligned(tmp_path: Path) -> None:
    relative = Path("raw") / "allen_cell_type" / "swc" / FIXTURE.name
    destination = tmp_path / "data" / relative
    destination.parent.mkdir(parents=True)
    shutil.copyfile(FIXTURE, destination)
    content_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    entry = ManifestEntry(
        dataset="act4",
        sample_id="act4/simple.swc",
        phase="train",
        source_index=0,
        fold=0,
        label_id=0,
        label_name="Isocortex_layer23",
        relative_path=relative.as_posix(),
        filename=FIXTURE.name,
        is_reg=False,
        content_sha256=content_hash,
    )

    first = build_or_load_spectra(
        data_root=tmp_path / "data",
        entries=[entry],
        manifest_sha256="a" * 64,
        cache_root=tmp_path / "cache",
        parameters=SpectrumParameters(),
    )
    second = build_or_load_spectra(
        data_root=tmp_path / "data",
        entries=[entry],
        manifest_sha256="a" * 64,
        cache_root=tmp_path / "cache",
        parameters=SpectrumParameters(),
    )
    changed_manifest = build_or_load_spectra(
        data_root=tmp_path / "data",
        entries=[entry],
        manifest_sha256="b" * 64,
        cache_root=tmp_path / "cache",
        parameters=SpectrumParameters(),
    )

    assert not first.cache_hit
    assert second.cache_hit
    assert first.cache_path == second.cache_path
    assert changed_manifest.cache_path != first.cache_path
    np.testing.assert_array_equal(first.spectra, second.spectra)
    assert first.spectra.shape == (1, 64)
