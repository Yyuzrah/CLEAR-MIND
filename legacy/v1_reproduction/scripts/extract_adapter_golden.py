"""Generate adapter golden hashes from the locked notebook and V.beta source."""

from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh, pinvh

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TREE_NOTEBOOK = REPOSITORY_ROOT / "notebooks" / "reference" / "V.alpha_TreeLSTM_arch3.ipynb"
FIXTURE = REPOSITORY_ROOT / "tests" / "fixtures" / "simple.swc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vbeta-source", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY_ROOT / "tests" / "golden" / "adapters_v1.json",
    )
    return parser.parse_args()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def load_vbeta(path: Path):
    spec = importlib.util.spec_from_file_location("_legacy_vbeta_gnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tree_reference() -> dict[str, object]:
    notebook = json.loads(TREE_NOTEBOOK.read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][3]["source"])
    namespace = {
        "List": list,
        "F": F,
        "dataclass": dataclass,
        "eigh": eigh,
        "glob": glob,
        "nn": nn,
        "np": np,
        "nx": nx,
        "os": os,
        "pd": pd,
        "pinvh": pinvh,
        "torch": torch,
        "tqdm": lambda iterable, **_: iterable,
    }
    exec(compile(source, f"{TREE_NOTEBOOK}:cell-3", "exec"), namespace)
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        shutil.copyfile(FIXTURE, directory / FIXTURE.name)
        metadata = directory / "metadata.csv"
        pd.DataFrame(
            [
                {
                    "structure_merge__acronym": "Isocortex_layer23",
                    "model__fold": 0.0,
                    "swc__fname": FIXTURE.name,
                }
            ]
        ).to_csv(metadata, index=False)
        dataset = namespace["MorphologyDataset"](
            "ACT-4",
            str(directory),
            str(metadata),
            phase="train",
        )
        sample = dataset[0]
    return {
        "x_shape": list(sample["x"].shape),
        "x_sha256": array_hash(sample["x"].numpy()),
        "parent_indices": sample["parent_indices"].tolist(),
        "levels": [level.tolist() for level in sample["levels"]],
        "root_idx": int(sample["root_idx"]),
        "spectral_sha256": array_hash(sample["spectral_sig"].numpy()),
        "label": int(sample["label"]),
    }


def main() -> int:
    args = parse_args()
    vbeta = load_vbeta(args.vbeta_source)
    parsed = vbeta.parse_swc_features(FIXTURE)
    rng = np.random.default_rng(
        vbeta.stable_seed(
            "ACT-4",
            FIXTURE.name,
            "random",
            1024,
            vbeta.FEATURE_VERSION,
        )
    )
    sampled = vbeta.random_sample_indices(len(parsed["features"]), 1024, rng)
    point_cloud = np.asarray(parsed["features"][sampled].T, dtype=np.float32)
    tree_parent = vbeta.sampled_tree_parent(parsed, sampled).astype(np.int64)

    payload = {
        "schema_version": 1,
        "fixture": FIXTURE.relative_to(REPOSITORY_ROOT).as_posix(),
        "fixture_sha256": hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        "gnn": {
            "source": str(args.vbeta_source),
            "source_sha256": hashlib.sha256(args.vbeta_source.read_bytes()).hexdigest(),
            "feature_version": vbeta.FEATURE_VERSION,
            "num_points": 1024,
            "sampled_indices_sha256": array_hash(sampled),
            "point_cloud_shape": list(point_cloud.shape),
            "point_cloud_sha256": array_hash(point_cloud),
            "tree_parent_sha256": array_hash(tree_parent),
        },
        "treelstm": {
            "source": TREE_NOTEBOOK.relative_to(REPOSITORY_ROOT).as_posix(),
            "source_sha256": hashlib.sha256(TREE_NOTEBOOK.read_bytes()).hexdigest(),
            "source_cell": 3,
            **tree_reference(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote adapter golden data to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
