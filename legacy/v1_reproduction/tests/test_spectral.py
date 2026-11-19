from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from em_connectome.data import load_manifest, load_neuron_record, sha256_file
from em_connectome.data.swc import parse_swc
from em_connectome.spectral import (
    ArakelovGreenSpectrum,
    SpectrumParameters,
    spectrum_cache_key,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_PATH = REPOSITORY_ROOT / "tests" / "golden" / "arakelov_green_v1.json"


def load_golden() -> dict[str, object]:
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


def test_golden_provenance_points_to_preserved_notebook_code() -> None:
    golden = load_golden()
    notebook_path = REPOSITORY_ROOT / golden["source_notebook"]
    assert golden["source_cell"] == 6
    assert sha256_file(notebook_path) == golden["repository_notebook_sha256"]
    assert golden["parameters"] == {
        "epsilon": 20.0,
        "noise_threshold": 5.0,
        "top_k": 64,
    }


def test_arakelev_green_matches_notebook_on_synthetic_fixture() -> None:
    golden = load_golden()
    fixture = golden["fixtures"][0]
    fixture_path = REPOSITORY_ROOT / fixture["path"]
    assert sha256_file(fixture_path) == fixture["content_sha256"]

    actual = ArakelovGreenSpectrum().extract_tree(parse_swc(fixture_path))
    expected = np.asarray(fixture["values"], dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-9)
    assert actual.shape == (64,)
    assert np.isfinite(actual).all()


def test_real_notebook_golden_vectors_when_data_root_is_available() -> None:
    raw_root = os.environ.get("EM_CONNECTOME_DATA_ROOT")
    if not raw_root:
        pytest.skip("set EM_CONNECTOME_DATA_ROOT to validate real notebook golden vectors")
    data_root = Path(raw_root)
    golden = load_golden()
    extractor = ArakelovGreenSpectrum()
    manifests = {
        key: {
            row.sample_id: row
            for row in load_manifest(REPOSITORY_ROOT / "manifests" / f"{key}.csv")
        }
        for key in ("act4", "jml4", "bil6")
    }

    for expected_record in golden["records"]:
        dataset = expected_record["sample_id"].split("/", maxsplit=1)[0]
        entry = manifests[dataset][expected_record["sample_id"]]
        record = load_neuron_record(data_root, entry)
        assert record.swc_sha256 == expected_record["content_sha256"]
        actual = extractor.extract(record)
        expected = np.asarray(expected_record["values"], dtype=np.float64)
        np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-8)


def test_cache_key_covers_content_manifest_parameters_and_version() -> None:
    base = {
        "swc_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
        "parameters": SpectrumParameters(),
        "implementation_version": ArakelovGreenSpectrum.implementation_version,
    }
    key = spectrum_cache_key(**base)
    assert len(key) == 64
    assert key != spectrum_cache_key(**{**base, "swc_sha256": "c" * 64})
    assert key != spectrum_cache_key(**{**base, "manifest_sha256": "d" * 64})
    assert key != spectrum_cache_key(**{**base, "parameters": SpectrumParameters(epsilon=19.0)})
    assert key != spectrum_cache_key(**{**base, "implementation_version": "next"})
