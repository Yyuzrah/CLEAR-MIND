from __future__ import annotations

import json
from pathlib import Path

from em_connectome.data import DATASETS, load_manifest, sha256_file, validate_entries

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIRECTORY = REPOSITORY_ROOT / "manifests"


def test_locked_manifests_match_registry_and_hashes() -> None:
    lock = json.loads((MANIFEST_DIRECTORY / "manifest-lock.json").read_text(encoding="utf-8"))
    assert lock["schema_version"] == 1
    assert set(lock["datasets"]) == set(DATASETS)

    for key, spec in DATASETS.items():
        manifest_path = MANIFEST_DIRECTORY / f"{key}.csv"
        rows = load_manifest(manifest_path)
        summary = validate_entries(rows, spec)
        locked = lock["datasets"][key]

        assert sha256_file(manifest_path) == locked["manifest_sha256"]
        assert summary["rows"] == spec.expected_total == locked["rows"]
        assert summary["split_counts"] == {
            "train": spec.expected_train,
            "test": spec.expected_test,
        }


def test_every_manifest_path_is_relative_and_split_is_canonical() -> None:
    for key, spec in DATASETS.items():
        rows = load_manifest(MANIFEST_DIRECTORY / f"{key}.csv")
        assert all(not Path(row.relative_path).is_absolute() for row in rows)
        assert all(row.phase == ("train" if row.fold < 8 else "test") for row in rows)
        assert len({row.sample_id for row in rows}) == spec.expected_total


def test_bil_manifest_is_the_exact_registered_cohort() -> None:
    rows = load_manifest(MANIFEST_DIRECTORY / "bil6.csv")
    assert len(rows) == 1200
    assert sum(row.phase == "train" for row in rows) == 958
    assert sum(row.phase == "test" for row in rows) == 242
    assert all(row.is_reg and row.filename.lower().endswith("_reg.swc") for row in rows)
