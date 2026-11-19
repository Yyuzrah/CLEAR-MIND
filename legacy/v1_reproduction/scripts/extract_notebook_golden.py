"""Extract spectrum golden vectors from the preserved MLP code cell."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from scipy.linalg import eigh, pinvh

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = (
    REPOSITORY_ROOT / "notebooks" / "reference" / "V.alpha_MLP_clean_champion_reproduction.ipynb"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "golden" / "arakelov_green_v1.json",
    )
    return parser.parse_args()


def _load_reference_class():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][6]["source"])
    class_source = source.split("\ndef manifest_fingerprint", maxsplit=1)[0]
    namespace = {
        "Path": Path,
        "nx": nx,
        "np": np,
        "eigh": eigh,
        "pinvh": pinvh,
    }
    exec(compile(class_source, f"{NOTEBOOK}:cell-6", "exec"), namespace)
    return namespace["RawExactSpectralExtractor"]


def main() -> int:
    args = parse_args()
    reference_extractor = _load_reference_class()
    fixture_path = REPOSITORY_ROOT / "tests" / "fixtures" / "simple.swc"
    fixture_vector = reference_extractor(fixture_path).signature(
        epsilon=20.0,
        tau=5.0,
        top_k=64,
    )
    records: list[dict[str, object]] = []
    for dataset in ("act4", "jml4", "bil6"):
        manifest_path = REPOSITORY_ROOT / "manifests" / f"{dataset}.csv"
        import csv

        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        chosen = [
            next(row for row in rows if row["phase"] == "train"),
            next(row for row in rows if row["phase"] == "test"),
        ]
        for row in chosen:
            swc_path = args.data_root / Path(row["relative_path"])
            vector = reference_extractor(swc_path).signature(
                epsilon=20.0,
                tau=5.0,
                top_k=64,
            )
            records.append(
                {
                    "sample_id": row["sample_id"],
                    "phase": row["phase"],
                    "content_sha256": row["content_sha256"],
                    "values": [float(value) for value in vector],
                }
            )

    payload = {
        "schema_version": 1,
        "source_notebook": NOTEBOOK.relative_to(REPOSITORY_ROOT).as_posix(),
        "historical_source_notebook_sha256": (
            "48c6f729efcb8aa4a9096712e5e8c3a8057801adfcdbad8c274cfa9556bd0907"
        ),
        "repository_notebook_sha256": hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest(),
        "source_cell": 6,
        "method": "arakelov_green",
        "parameters": {
            "epsilon": 20.0,
            "noise_threshold": 5.0,
            "top_k": 64,
        },
        "fixtures": [
            {
                "path": fixture_path.relative_to(REPOSITORY_ROOT).as_posix(),
                "content_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
                "values": [float(value) for value in fixture_vector],
            }
        ],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} notebook-derived vectors to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
