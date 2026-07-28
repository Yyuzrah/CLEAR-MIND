"""Regenerate the locked version-one manifests from the canonical metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from em_connectome.data.manifest import (  # noqa: E402
    build_entries,
    validate_entries,
    write_manifest,
    write_manifest_lock,
)
from em_connectome.data.registry import DATASETS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Directory containing info/ and raw/ (canonical source: external/treemoco/data)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "manifests",
        help="Manifest output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_directory = args.output_dir.resolve()
    lock: dict[str, object] = {
        "schema_version": 1,
        "manifest_set": "v1-arakelov-green",
        "metadata_policy": "champion exact-loader metadata; exact raw filename; folds 0-7/8-9",
        "datasets": {},
    }

    for key, spec in DATASETS.items():
        entries = build_entries(args.data_root, spec)
        manifest_path = output_directory / f"{key}.csv"
        manifest_hash = write_manifest(manifest_path, entries)
        summary = validate_entries(entries, spec)
        lock["datasets"][key] = {
            "display_name": spec.display_name,
            "manifest": manifest_path.name,
            "manifest_sha256": manifest_hash,
            "metadata_filename": spec.metadata_filename,
            "metadata_sha256": spec.metadata_sha256,
            **summary,
        }
        print(
            f"{key}: {summary['split_counts']['train']} train / "
            f"{summary['split_counts']['test']} test; {manifest_hash}"
        )

    write_manifest_lock(output_directory / "manifest-lock.json", lock)
    print(json.dumps(lock, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
