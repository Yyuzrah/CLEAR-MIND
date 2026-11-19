"""User-facing reproducibility gates independent of pytest."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from em_connectome.adapters import GNNAdapter, TreeLSTMAdapter
from em_connectome.data import (
    DATASETS,
    NeuronRecord,
    audit_local_files,
    load_manifest,
    load_neuron_record,
    parse_swc,
    sha256_file,
    validate_entries,
)
from em_connectome.spectral import ArakelovGreenSpectrum


def _array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def verify_data(
    repository_root: Path,
    data_root: Path,
    datasets: list[str] | None = None,
) -> dict[str, object]:
    keys = datasets or list(DATASETS)
    result: dict[str, object] = {"gate": "data", "datasets": {}, "ok": True}
    for key in keys:
        spec = DATASETS[key]
        path = repository_root / "manifests" / f"{key}.csv"
        entries = load_manifest(path)
        manifest_summary = validate_entries(entries, spec)
        local = audit_local_files(data_root, entries, verify_hashes=True)
        result["datasets"][key] = {
            "manifest_sha256": sha256_file(path),
            "manifest": manifest_summary,
            "local": {
                "rows": local["rows"],
                "missing_count": local["missing_count"],
                "hash_mismatch_count": local["hash_mismatch_count"],
                "missing_examples": local["missing_sample_ids"][:10],
                "hash_mismatch_examples": local["hash_mismatch_sample_ids"][:10],
                "ok": local["ok"],
            },
        }
        result["ok"] = bool(result["ok"] and local["ok"])
    return result


def verify_features(
    repository_root: Path,
    data_root: Path,
) -> dict[str, object]:
    golden_path = repository_root / "tests" / "golden" / "arakelov_green_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    notebook = repository_root / golden["source_notebook"]
    provenance_ok = sha256_file(notebook) == golden["repository_notebook_sha256"]
    manifests = {
        key: {
            row.sample_id: row
            for row in load_manifest(repository_root / "manifests" / f"{key}.csv")
        }
        for key in DATASETS
    }
    extractor = ArakelovGreenSpectrum()
    records: list[dict[str, object]] = []
    for expected in golden["records"]:
        dataset = expected["sample_id"].split("/", maxsplit=1)[0]
        entry = manifests[dataset][expected["sample_id"]]
        record = load_neuron_record(data_root, entry)
        actual = extractor.extract(record)
        target = np.asarray(expected["values"], dtype=np.float64)
        matches = bool(np.allclose(actual, target, rtol=1e-10, atol=1e-8))
        records.append(
            {
                "sample_id": entry.sample_id,
                "content_sha256_ok": record.swc_sha256 == expected["content_sha256"],
                "max_absolute_error": float(np.max(np.abs(actual - target))),
                "matches": matches,
            }
        )
    ok = provenance_ok and all(row["content_sha256_ok"] and row["matches"] for row in records)
    return {
        "gate": "features",
        "repository_notebook_sha256_ok": provenance_ok,
        "records": records,
        "ok": ok,
    }


def verify_adapters(repository_root: Path) -> dict[str, object]:
    golden_path = repository_root / "tests" / "golden" / "adapters_v1.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    fixture = repository_root / golden["fixture"]
    fixture_hash_ok = sha256_file(fixture) == golden["fixture_sha256"]
    morphology = parse_swc(fixture)
    record = NeuronRecord(
        sample_id="act4/simple.swc",
        dataset="act4",
        swc_path=fixture,
        swc_sha256=sha256_file(fixture),
        label_id=0,
        label_name="Isocortex_layer23",
        fold=0,
        phase="train",
        morphology=morphology,
    ).with_spectrum(ArakelovGreenSpectrum().extract_tree(morphology))

    gnn = GNNAdapter().transform(record)
    tree = TreeLSTMAdapter().transform(record)
    gnn_checks = {
        "point_cloud": _array_hash(gnn.point_cloud) == golden["gnn"]["point_cloud_sha256"],
        "tree_parent": _array_hash(gnn.tree_parent) == golden["gnn"]["tree_parent_sha256"],
        "spectrum_label_alignment": gnn.label == record.label_id
        and np.array_equal(gnn.spectrum, record.spectrum.astype(np.float32)),
    }
    tree_checks = {
        "node_features": _array_hash(tree.x.numpy()) == golden["treelstm"]["x_sha256"],
        "parent_indices": tree.parent_indices.tolist() == golden["treelstm"]["parent_indices"],
        "levels": [level.tolist() for level in tree.levels] == golden["treelstm"]["levels"],
        "root": tree.root_idx == golden["treelstm"]["root_idx"],
        "spectrum": _array_hash(tree.spectral_sig.numpy()) == golden["treelstm"]["spectral_sha256"],
        "label": tree.label == golden["treelstm"]["label"],
    }
    return {
        "gate": "adapters",
        "fixture_sha256_ok": fixture_hash_ok,
        "gnn": gnn_checks,
        "treelstm": tree_checks,
        "ok": fixture_hash_ok and all(gnn_checks.values()) and all(tree_checks.values()),
    }


def verify_results(
    repository_root: Path,
    summary_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Validate a complete formal matrix against the published v1 tolerances."""

    targets_path = repository_root / "reports" / "v1_regression_targets.yaml"
    targets = yaml.safe_load(targets_path.read_text(encoding="utf-8"))
    if not isinstance(targets, dict) or targets.get("schema_version") != 1:
        raise ValueError(f"invalid regression target file: {targets_path}")
    expected_seeds = [int(seed) for seed in targets["seeds"]]
    experiments = targets.get("experiments")
    if not isinstance(experiments, dict) or not experiments:
        raise ValueError(f"no experiments declared in {targets_path}")

    paths = summary_paths or [repository_root / "reports" / "v1_formal_results.csv"]
    resolved_paths = [
        path.resolve() if path.is_absolute() else (repository_root / path).resolve()
        for path in paths
    ]
    required = {
        "dataset",
        "model",
        "seed",
        "mode",
        "train_count",
        "test_count",
        "best_test_accuracy",
        "config_hash",
        "manifest_sha256",
        "feature_set_hash",
    }
    rows: list[dict[str, str]] = []
    for path in resolved_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = required.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{path} is missing result columns: {sorted(missing)}")
            rows.extend(reader)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        key = f"{row['dataset']}_{row['model']}"
        grouped.setdefault(key, []).append(row)

    unexpected = sorted(set(grouped).difference(experiments))
    result_rows: dict[str, object] = {}
    overall_ok = not unexpected
    for key, target in experiments.items():
        if not isinstance(target, dict):
            raise ValueError(f"regression target {key!r} must be a mapping")
        experiment_rows = grouped.get(key, [])
        seeds = [int(row["seed"]) for row in experiment_rows]
        accuracies = [float(row["best_test_accuracy"]) for row in experiment_rows]
        finite_accuracies = all(
            math.isfinite(value) and 0.0 <= value <= 1.0 for value in accuracies
        )
        mean_accuracy = float(np.mean(accuracies)) if accuracies else float("nan")
        tolerance = float(target["mean_accuracy_absolute_tolerance"])
        if tolerance < 0:
            raise ValueError(f"negative tolerance for {key}")
        target_mean = float(target["mean_accuracy"])

        checks = {
            "five_unique_formal_seeds": (
                sorted(seeds) == sorted(expected_seeds) and len(seeds) == len(set(seeds))
            ),
            "formal_mode": bool(experiment_rows)
            and all(row["mode"] == "formal" for row in experiment_rows),
            "split_counts": bool(experiment_rows)
            and all(
                int(row["train_count"]) == int(target["train_count"])
                and int(row["test_count"]) == int(target["test_count"])
                for row in experiment_rows
            ),
            "config_hash": bool(experiment_rows)
            and {row["config_hash"] for row in experiment_rows} == {target["config_hash"]},
            "manifest_sha256": bool(experiment_rows)
            and {row["manifest_sha256"] for row in experiment_rows} == {target["manifest_sha256"]},
            "feature_set_hash": bool(experiment_rows)
            and {row["feature_set_hash"] for row in experiment_rows}
            == {target["feature_set_hash"]},
            "finite_accuracies": finite_accuracies,
            "mean_accuracy_within_tolerance": bool(accuracies)
            and abs(mean_accuracy - target_mean) <= tolerance,
        }
        experiment_ok = all(checks.values())
        overall_ok = overall_ok and experiment_ok
        result_rows[key] = {
            "rows": len(experiment_rows),
            "mean_accuracy": mean_accuracy if accuracies else None,
            "target_mean_accuracy": target_mean,
            "absolute_difference": (abs(mean_accuracy - target_mean) if accuracies else None),
            "absolute_tolerance": tolerance,
            "checks": checks,
            "ok": experiment_ok,
        }

    return {
        "gate": "results",
        "summaries": [{"path": str(path), "sha256": sha256_file(path)} for path in resolved_paths],
        "expected_runs": len(experiments) * len(expected_seeds),
        "found_runs": len(rows),
        "unexpected_experiments": unexpected,
        "experiments": result_rows,
        "ok": overall_ok and len(rows) == len(experiments) * len(expected_seeds),
    }
