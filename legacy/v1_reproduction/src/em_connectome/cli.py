"""Command-line interface for the version-one reproduction pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from em_connectome.config import FORMAL_SEEDS, resolve_experiment
from em_connectome.data import (
    DATASETS,
    audit_local_files,
    get_dataset_spec,
    load_manifest,
    sha256_file,
    validate_entries,
)
from em_connectome.spectral import (
    SpectrumParameters,
    build_or_load_spectra,
    select_smoke_entries,
)
from em_connectome.training import run_training
from em_connectome.verify import (
    verify_adapters,
    verify_data,
    verify_features,
    verify_results,
)


def default_repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def require_data_root(value: str | None) -> Path:
    raw = value or os.environ.get("EM_CONNECTOME_DATA_ROOT")
    if not raw:
        raise ValueError(
            "provide --data-root or set EM_CONNECTOME_DATA_ROOT to a directory "
            "containing raw/ (and info/ only when regenerating manifests)"
        )
    root = Path(raw).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"data root does not exist: {root}")
    return root


def _common_paths(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    repository = Path(args.repository_root).resolve()
    cache = Path(args.cache_root or repository / "cache").resolve()
    output = Path(args.output_root or repository / "runs").resolve()
    return repository, cache, output


def _print(payload: object) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


def command_data_audit(args: argparse.Namespace) -> int:
    repository, _, _ = _common_paths(args)
    data_root = require_data_root(args.data_root)
    spec = get_dataset_spec(args.dataset)
    manifest_path = repository / "manifests" / f"{spec.key}.csv"
    entries = load_manifest(manifest_path)
    manifest = validate_entries(entries, spec)
    local = audit_local_files(data_root, entries, verify_hashes=not args.skip_hashes)
    payload = {
        "dataset": spec.key,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest": manifest,
        "local": {
            "rows": local["rows"],
            "missing_count": local["missing_count"],
            "hash_mismatch_count": local["hash_mismatch_count"],
            "missing_examples": local["missing_sample_ids"][:10],
            "hash_mismatch_examples": local["hash_mismatch_sample_ids"][:10],
            "ok": local["ok"],
        },
    }
    _print(payload)
    return 0 if local["ok"] else 1


def command_data_build(args: argparse.Namespace) -> int:
    repository, cache_root, _ = _common_paths(args)
    data_root = require_data_root(args.data_root)
    spec = get_dataset_spec(args.dataset)
    manifest_path = repository / "manifests" / f"{spec.key}.csv"
    entries = load_manifest(manifest_path)
    validate_entries(entries, spec)
    audit = audit_local_files(data_root, entries, verify_hashes=True)
    if not audit["ok"]:
        _print(audit)
        return 1
    index_path = cache_root / "data" / spec.key / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "dataset": spec.key,
        "data_root": str(data_root),
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "rows": [
            {
                "sample_id": entry.sample_id,
                "relative_path": entry.relative_path,
                "content_sha256": entry.content_sha256,
                "fold": entry.fold,
                "phase": entry.phase,
                "label_id": entry.label_id,
            }
            for entry in entries
        ],
    }
    index_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _print({"dataset": spec.key, "rows": len(entries), "index": str(index_path), "ok": True})
    return 0


def command_spectral_build(args: argparse.Namespace) -> int:
    repository, cache_root, _ = _common_paths(args)
    data_root = require_data_root(args.data_root)
    spec = get_dataset_spec(args.dataset)
    manifest_path = repository / "manifests" / f"{spec.key}.csv"
    entries = load_manifest(manifest_path)
    selected = (
        select_smoke_entries(entries, per_class_per_split=1) if args.mode == "smoke" else entries
    )
    bundle = build_or_load_spectra(
        data_root=data_root,
        entries=selected,
        manifest_sha256=sha256_file(manifest_path),
        cache_root=cache_root,
        parameters=SpectrumParameters(),
        force=args.force,
    )
    _print(
        {
            "dataset": spec.key,
            "mode": args.mode,
            "rows": len(bundle.entries),
            "cache": str(bundle.cache_path),
            "cache_hit": bundle.cache_hit,
            "feature_set_hash": bundle.feature_set_hash,
            "ok": True,
        }
    )
    return 0


def command_train(args: argparse.Namespace) -> int:
    repository, cache_root, output_root = _common_paths(args)
    data_root = require_data_root(args.data_root)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repository / config_path
    config = resolve_experiment(config_path)
    seeds = [args.seed] if args.seed is not None else list(config.seeds)
    results = []
    for seed in seeds:
        result = run_training(
            config=config,
            data_root=data_root,
            repository_root=repository,
            cache_root=cache_root,
            output_root=output_root,
            seed=seed,
            mode=args.mode,
            device_name=args.device,
            force_spectra=args.force_spectra,
        )
        results.append(
            {
                "run_directory": str(result.run_directory),
                "metrics": result.metrics,
            }
        )
    _print({"runs": results, "ok": True})
    return 0


def _experiment_paths(repository: Path, experiment: str) -> list[Path]:
    directory = repository / "configs" / "experiments"
    if experiment == "all":
        return sorted(directory.glob("*.yaml"))
    candidate = Path(experiment)
    if candidate.suffix:
        path = candidate if candidate.is_absolute() else repository / candidate
    else:
        path = directory / f"{experiment}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"experiment config not found: {path}")
    return [path]


def command_reproduce(args: argparse.Namespace) -> int:
    repository, cache_root, output_root = _common_paths(args)
    data_root = require_data_root(args.data_root)
    rows: list[dict[str, object]] = []
    for config_path in _experiment_paths(repository, args.experiment):
        config = resolve_experiment(config_path)
        seeds = [config.seeds[0]] if args.mode == "smoke" else list(config.seeds)
        for seed in seeds:
            result = run_training(
                config=config,
                data_root=data_root,
                repository_root=repository,
                cache_root=cache_root,
                output_root=output_root,
                seed=seed,
                mode=args.mode,
                device_name=args.device,
            )
            rows.append(
                {
                    **result.metrics,
                    "run_directory": str(result.run_directory),
                }
            )
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary_path = output_root / f"reproduction_summary_{args.mode}_{timestamp}.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _print({"mode": args.mode, "runs": len(rows), "summary": str(summary_path), "ok": True})
    return 0


def command_verify(args: argparse.Namespace) -> int:
    repository, _, _ = _common_paths(args)
    targets = ["data", "features", "adapters", "results"] if args.target == "all" else [args.target]
    results = []
    for target in targets:
        if target == "adapters":
            results.append(verify_adapters(repository))
            continue
        if target == "results":
            summaries = [Path(path) for path in args.summary] if args.summary else None
            results.append(verify_results(repository, summaries))
            continue
        data_root = require_data_root(args.data_root)
        if target == "data":
            datasets = [get_dataset_spec(args.dataset).key] if args.dataset else None
            results.append(verify_data(repository, data_root, datasets))
        elif target == "features":
            results.append(verify_features(repository, data_root))
    ok = all(result["ok"] for result in results)
    _print({"results": results, "ok": ok})
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="em-connectome")
    parser.add_argument(
        "--repository-root",
        default=str(default_repository_root()),
        help="repository containing configs/, manifests/, tests/, and runs/",
    )
    parser.add_argument("--cache-root")
    parser.add_argument("--output-root")
    subcommands = parser.add_subparsers(dest="command", required=True)

    data = subcommands.add_parser("data")
    data_subcommands = data.add_subparsers(dest="data_command", required=True)
    audit = data_subcommands.add_parser("audit")
    audit.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    audit.add_argument("--data-root")
    audit.add_argument("--skip-hashes", action="store_true")
    audit.set_defaults(handler=command_data_audit)
    build = data_subcommands.add_parser("build")
    build.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    build.add_argument("--data-root")
    build.set_defaults(handler=command_data_build)

    spectral = subcommands.add_parser("spectral")
    spectral_subcommands = spectral.add_subparsers(
        dest="spectral_command",
        required=True,
    )
    spectral_build = spectral_subcommands.add_parser("build")
    spectral_build.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    spectral_build.add_argument("--data-root")
    spectral_build.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    spectral_build.add_argument("--force", action="store_true")
    spectral_build.set_defaults(handler=command_spectral_build)

    train = subcommands.add_parser("train")
    train.add_argument("--config", required=True)
    train.add_argument("--data-root")
    train.add_argument("--seed", type=int, choices=FORMAL_SEEDS)
    train.add_argument("--mode", choices=("formal", "smoke"), default="formal")
    train.add_argument("--device", default="auto")
    train.add_argument("--force-spectra", action="store_true")
    train.set_defaults(handler=command_train)

    reproduce = subcommands.add_parser("reproduce")
    reproduce.add_argument("--experiment", default="all")
    reproduce.add_argument("--data-root")
    reproduce.add_argument("--mode", choices=("formal", "smoke"), default="smoke")
    reproduce.add_argument("--device", default="auto")
    reproduce.set_defaults(handler=command_reproduce)

    verify = subcommands.add_parser("verify")
    verify.add_argument(
        "target",
        choices=("data", "features", "adapters", "results", "all"),
    )
    verify.add_argument("--dataset", choices=sorted(DATASETS))
    verify.add_argument("--data-root")
    verify.add_argument(
        "--summary",
        action="append",
        help=(
            "formal reproduction CSV to verify; repeat for separately run experiments "
            "(defaults to reports/v1_formal_results.csv)"
        ),
    )
    verify.set_defaults(handler=command_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
