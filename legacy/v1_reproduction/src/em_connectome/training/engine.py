"""Unified, configuration-driven training and result archival."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import platform
import random
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, Dataset

from em_connectome.adapters import (
    GNNAdapter,
    MLPAdapter,
    SpectrumStandardizer,
    TreeLSTMAdapter,
    collate_gnn_samples,
    collate_treelstm_samples,
)
from em_connectome.config import FORMAL_SEEDS, ResolvedExperiment, resolved_yaml
from em_connectome.data import (
    ManifestEntry,
    load_manifest,
    load_neuron_record,
    sha256_file,
)
from em_connectome.models import MLP, GNNAlpha, TreeLSTMArch3
from em_connectome.spectral import (
    SpectrumParameters,
    build_or_load_spectra,
    select_smoke_entries,
)

RunMode = Literal["formal", "smoke"]

# One CLI invocation runs the five seeds of a config consecutively. Reuse the
# immutable adapted datasets across those seeds, then evict them when the next
# dataset/model/config is prepared so full reproduction stays memory-bounded.
_PREPARED_DATA_CACHE: dict[
    tuple[str, str, str, str, RunMode],
    tuple[Dataset, Dataset, object | None],
] = {}


@dataclass(frozen=True)
class RunResult:
    run_directory: Path
    metrics: dict[str, object]


@dataclass(frozen=True)
class _Evaluation:
    loss: float
    accuracy: float
    macro_f1: float
    sample_ids: list[str]
    labels: list[int]
    predictions: list[int]


class _ListDataset(Dataset):
    def __init__(self, samples: list[object]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> object:
        return self.samples[index]


class _MLPDataset(Dataset):
    def __init__(
        self,
        sample_ids: list[str],
        features: np.ndarray,
        labels: np.ndarray,
    ):
        self.sample_ids = sample_ids
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {
            "sample_id": self.sample_ids[index],
            "features": self.features[index],
            "label": self.labels[index],
        }


def set_global_seed(seed: int, *, deterministic_algorithms: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(
        deterministic_algorithms,
        warn_only=deterministic_algorithms,
    )
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str = "auto") -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _repository_root(config: ResolvedExperiment, manifest_path: Path) -> Path:
    del config
    return manifest_path.resolve().parents[1]


def _records_with_spectra(
    data_root: Path,
    entries: list[ManifestEntry],
    spectra: np.ndarray,
) -> list:
    if len(entries) != len(spectra):
        raise ValueError("manifest/spectrum row count mismatch")
    records = []
    for entry, spectrum in zip(entries, spectra, strict=True):
        record = load_neuron_record(data_root, entry).with_spectrum(spectrum)
        if record.sample_id != entry.sample_id:
            raise AssertionError("sample ID changed while attaching a spectrum")
        records.append(record)
    return records


def _prepare_datasets(
    config: ResolvedExperiment,
    records: list,
    *,
    mode: RunMode,
) -> tuple[Dataset, Dataset, object | None]:
    train_records = [record for record in records if record.phase == "train"]
    test_records = [record for record in records if record.phase == "test"]
    if not train_records or not test_records:
        raise ValueError("both train and test records are required")
    model_name = config.model.name

    if model_name == "mlp":
        adapter = MLPAdapter(feature_dim=int(config.model.parameters["input_dim"]))
        train_samples = adapter.transform_many(train_records)
        test_samples = adapter.transform_many(test_records)
        train_matrix = np.asarray([sample.features for sample in train_samples], dtype=np.float64)
        test_matrix = np.asarray([sample.features for sample in test_samples], dtype=np.float64)
        standardizer = SpectrumStandardizer.fit(train_matrix)
        return (
            _MLPDataset(
                [sample.sample_id for sample in train_samples],
                standardizer.transform(train_matrix),
                np.asarray([sample.label for sample in train_samples], dtype=np.int64),
            ),
            _MLPDataset(
                [sample.sample_id for sample in test_samples],
                standardizer.transform(test_matrix),
                np.asarray([sample.label for sample in test_samples], dtype=np.int64),
            ),
            standardizer,
        )
    if model_name == "gnn_alpha":
        formal_points = int(config.model.parameters["num_points"])
        point_count = min(formal_points, 64) if mode == "smoke" else formal_points
        adapter = GNNAdapter(
            num_points=point_count,
            spectral_dim=int(config.model.parameters["spectral_dim"]),
        )
        return (
            _ListDataset(adapter.transform_many(train_records)),
            _ListDataset(adapter.transform_many(test_records)),
            None,
        )
    if model_name == "treelstm_arch3":
        adapter = TreeLSTMAdapter(spectral_dim=int(config.model.parameters["spectral_dim"]))
        return (
            _ListDataset(adapter.transform_many(train_records)),
            _ListDataset(adapter.transform_many(test_records)),
            None,
        )
    raise ValueError(f"unsupported model: {model_name}")


def _collate_for(model_name: str):
    if model_name == "gnn_alpha":
        return collate_gnn_samples
    if model_name == "treelstm_arch3":
        return collate_treelstm_samples
    return None


def _build_model(config: ResolvedExperiment) -> nn.Module:
    model = config.model
    num_classes = len(config.dataset.classes)
    if model.name == "mlp":
        return MLP(
            num_classes=num_classes,
            input_dim=int(model.parameters["input_dim"]),
            hidden_dims=tuple(int(value) for value in model.parameters["hidden_dims"]),
            dropout=float(model.parameters["dropout"]),
        )
    if model.name == "gnn_alpha":
        return GNNAlpha(
            num_classes=num_classes,
            k=int(model.parameters["k"]),
            spectral_dim=int(model.parameters["spectral_dim"]),
        )
    if model.name == "treelstm_arch3":
        return TreeLSTMArch3(
            num_classes=num_classes,
            node_feat_dim=int(model.parameters["node_feat_dim"]),
            hidden_dim=int(model.parameters["hidden_dim"]),
            spectral_dim=int(model.parameters["spectral_dim"]),
        )
    raise ValueError(f"unsupported model: {model.name}")


def _make_loaders(
    config: ResolvedExperiment,
    train_dataset: Dataset,
    test_dataset: Dataset,
    seed: int,
    mode: RunMode,
) -> tuple[DataLoader, DataLoader]:
    requested = config.trainer.batch_size
    batch_size = min(requested, len(train_dataset)) if mode == "smoke" else requested
    # MLP and the formal GNN runner bind shuffling to an explicit generator.
    # The TreeLSTM notebook intentionally uses Torch's global RNG; model
    # initialization happens after loader construction and before iteration,
    # so sharing that RNG is required to reproduce its batch order.
    generator = (
        None if config.model.name == "treelstm_arch3" else torch.Generator().manual_seed(seed)
    )
    collate = _collate_for(config.model.name)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=config.trainer.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=min(requested, len(test_dataset)) if mode == "smoke" else requested,
        shuffle=False,
        num_workers=config.trainer.num_workers,
        collate_fn=collate,
        drop_last=False,
    )
    return train_loader, test_loader


def _batch_parts(
    model_name: str,
    model: nn.Module,
    batch: object,
    device: torch.device,
    *,
    training: bool,
    jitter_std: float,
    jitter_clip: float,
) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    if model_name == "mlp":
        labels = batch["label"].to(device)
        logits = model(batch["features"].to(device))
        return logits, labels, list(batch["sample_id"])
    if model_name == "gnn_alpha":
        points = batch["point_cloud"].to(device).clone()
        if training and jitter_std > 0:
            noise = torch.randn_like(points[:, :3]) * jitter_std
            points[:, :3] += noise.clamp(-jitter_clip, jitter_clip)
        labels = batch["label"].to(device)
        logits = model(points, batch["spectral_sig"].to(device))
        return logits, labels, list(batch["sample_id"])
    if model_name == "treelstm_arch3":
        tree_batch = batch.to(device)
        return model(tree_batch), tree_batch.labels, tree_batch.sample_ids
    raise ValueError(model_name)


@torch.no_grad()
def _evaluate(
    config: ResolvedExperiment,
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> _Evaluation:
    model.eval()
    loss_sum = 0.0
    seen = 0
    sample_ids: list[str] = []
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for batch in loader:
        logits, labels, ids = _batch_parts(
            config.model.name,
            model,
            batch,
            device,
            training=False,
            jitter_std=0.0,
            jitter_clip=0.0,
        )
        loss_sum += float(criterion(logits, labels)) * len(labels)
        seen += len(labels)
        predictions = logits.argmax(dim=1)
        sample_ids.extend(ids)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(predictions.detach().cpu().tolist())
    if seen == 0:
        raise ValueError("evaluation loader is empty")
    labels_array = np.asarray(labels_all, dtype=np.int64)
    predictions_array = np.asarray(predictions_all, dtype=np.int64)
    return _Evaluation(
        loss=loss_sum / seen,
        accuracy=float(np.mean(labels_array == predictions_array)),
        macro_f1=float(
            f1_score(
                labels_array,
                predictions_array,
                labels=range(len(config.dataset.classes)),
                average="macro",
                zero_division=0,
            )
        ),
        sample_ids=sample_ids,
        labels=labels_all,
        predictions=predictions_all,
    )


def _is_better(
    current: _Evaluation,
    best: _Evaluation | None,
    selection_rule: str,
) -> bool:
    if best is None:
        return True
    if current.accuracy != best.accuracy:
        return current.accuracy > best.accuracy
    if selection_rule == "accuracy_macro_f1_loss":
        if current.macro_f1 != best.macro_f1:
            return current.macro_f1 > best.macro_f1
        return current.loss < best.loss
    return False


def _environment_payload(device: torch.device) -> dict[str, object]:
    packages = {}
    for name in (
        "em-connectome",
        "networkx",
        "numpy",
        "pandas",
        "pydantic",
        "PyYAML",
        "scikit-learn",
        "scipy",
        "torch",
    ):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu_name": (torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }


def _run_directory(
    output_root: Path,
    config: ResolvedExperiment,
    seed: int,
    mode: RunMode,
) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    digest = hashlib.sha256(f"{config.config_hash}:{seed}:{mode}".encode()).hexdigest()[:12]
    path = output_root.resolve() / f"{timestamp}_{config.dataset.key}_{config.model.name}_{digest}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def _write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_training(
    *,
    config: ResolvedExperiment,
    data_root: Path,
    repository_root: Path,
    cache_root: Path,
    output_root: Path,
    seed: int,
    mode: RunMode = "formal",
    device_name: str = "auto",
    force_spectra: bool = False,
) -> RunResult:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"seed must be in the formal protocol: {FORMAL_SEEDS}")
    device = resolve_device(device_name)
    set_global_seed(
        seed,
        deterministic_algorithms=config.model.name == "mlp",
    )

    manifest_path = repository_root.resolve() / config.dataset.manifest
    entries = load_manifest(manifest_path)
    manifest_sha256 = sha256_file(manifest_path)
    selected_entries = (
        select_smoke_entries(entries, per_class_per_split=1) if mode == "smoke" else entries
    )
    parameters = SpectrumParameters(
        method=config.spectral.method,
        epsilon=config.spectral.epsilon,
        noise_threshold=config.spectral.noise_threshold,
        top_k=config.spectral.top_k,
    )
    spectrum_bundle = build_or_load_spectra(
        data_root=data_root,
        entries=selected_entries,
        manifest_sha256=manifest_sha256,
        cache_root=cache_root,
        parameters=parameters,
        force=force_spectra,
    )
    prepared_key = (
        str(data_root.resolve()),
        spectrum_bundle.feature_set_hash,
        config.model.name,
        config.config_hash,
        mode,
    )
    prepared = _PREPARED_DATA_CACHE.get(prepared_key)
    if prepared is None:
        records = _records_with_spectra(
            data_root,
            spectrum_bundle.entries,
            spectrum_bundle.spectra,
        )
        prepared = _prepare_datasets(
            config,
            records,
            mode=mode,
        )
        _PREPARED_DATA_CACHE.clear()
        _PREPARED_DATA_CACHE[prepared_key] = prepared
    train_dataset, test_dataset, normalizer = prepared
    train_loader, test_loader = _make_loaders(
        config,
        train_dataset,
        test_dataset,
        seed,
        mode,
    )

    model = _build_model(config).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.trainer.label_smoothing)
    if config.trainer.optimizer == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.trainer.learning_rate,
            weight_decay=config.trainer.weight_decay,
        )
    epochs = min(config.trainer.epochs, 2) if mode == "smoke" else config.trainer.epochs
    patience = min(config.trainer.patience, 2) if mode == "smoke" else config.trainer.patience
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=config.trainer.eta_min,
        )
        if config.trainer.scheduler == "cosine"
        else None
    )

    best: _Evaluation | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale_epochs = 0
    history: list[dict[str, object]] = []
    log_lines: list[str] = []
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        for batch in train_loader:
            logits, labels, _ = _batch_parts(
                config.model.name,
                model,
                batch,
                device,
                training=True,
                jitter_std=config.trainer.jitter_std,
                jitter_clip=config.trainer.jitter_clip,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(logits, labels)
            loss.backward()
            if config.trainer.gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.trainer.gradient_clip,
                )
            optimizer.step()
            train_loss_sum += float(loss.detach()) * len(labels)
            train_seen += len(labels)
        if scheduler is not None:
            scheduler.step()
        current = _evaluate(config, model, test_loader, criterion, device)
        improved = _is_better(
            current,
            best,
            config.trainer.selection_rule,
        )
        if improved:
            best = copy.deepcopy(current)
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_seen,
            "test_loss": current.loss,
            "test_accuracy": current.accuracy,
            "test_macro_f1": current.macro_f1,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "selected_checkpoint": improved,
        }
        history.append(row)
        message = (
            f"epoch={epoch:03d} train_loss={row['train_loss']:.6f} "
            f"test_loss={current.loss:.6f} test_accuracy={current.accuracy:.6f} "
            f"test_macro_f1={current.macro_f1:.6f} "
            f"{'BEST' if improved else f'patience={stale_epochs}/{patience}'}"
        )
        print(message)
        log_lines.append(message)
        if stale_epochs >= patience:
            break

    if best is None or best_state is None:
        raise RuntimeError("training completed without a checkpoint")
    model.load_state_dict(best_state)
    final = _evaluate(config, model, test_loader, criterion, device)
    # The selected state must reproduce the evaluation captured at selection.
    if final.predictions != best.predictions:
        raise RuntimeError("restored checkpoint predictions do not match the selected epoch")

    run_directory = _run_directory(output_root, config, seed, mode)
    runtime_config = yaml.safe_load(resolved_yaml(config))
    runtime_config["runtime"] = {
        "mode": mode,
        "seed": seed,
        "device": str(device),
        "epochs": epochs,
        "patience": patience,
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "spectrum_cache_hit": spectrum_bundle.cache_hit,
        "spectrum_feature_set_hash": spectrum_bundle.feature_set_hash,
    }
    (run_directory / "config.resolved.yaml").write_text(
        yaml.safe_dump(runtime_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (run_directory / "manifest_hash.json").write_text(
        json.dumps(
            {
                "manifest": config.dataset.manifest,
                "manifest_sha256": manifest_sha256,
                "selected_rows": len(selected_entries),
                "feature_set_hash": spectrum_bundle.feature_set_hash,
                "spectrum_cache": str(spectrum_bundle.cache_path),
                "source_hashes": config.source_hashes,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (run_directory / "environment.json").write_text(
        json.dumps(_environment_payload(device), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(
        run_directory / "history.csv",
        list(history[0]),
        history,
    )
    metrics = {
        "dataset": config.dataset.key,
        "model": config.model.name,
        "seed": seed,
        "mode": mode,
        "train_count": len(train_dataset),
        "test_count": len(test_dataset),
        "epochs_ran": len(history),
        "best_epoch": best_epoch,
        "best_test_accuracy": best.accuracy,
        "best_test_macro_f1": best.macro_f1,
        "best_test_loss": best.loss,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "config_hash": config.config_hash,
        "manifest_sha256": manifest_sha256,
        "feature_set_hash": spectrum_bundle.feature_set_hash,
    }
    _write_csv(run_directory / "metrics.csv", list(metrics), [metrics])
    prediction_rows = [
        {
            "sample_id": sample_id,
            "true_label_id": label,
            "true_label_name": config.dataset.classes[label],
            "predicted_label_id": prediction,
            "predicted_label_name": config.dataset.classes[prediction],
        }
        for sample_id, label, prediction in zip(
            best.sample_ids,
            best.labels,
            best.predictions,
            strict=True,
        )
    ]
    _write_csv(
        run_directory / "predictions.csv",
        list(prediction_rows[0]),
        prediction_rows,
    )
    (run_directory / "train.log").write_text(
        "\n".join(log_lines) + "\n",
        encoding="utf-8",
    )
    checkpoint: dict[str, object] = {
        "state_dict": best_state,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_evaluation": asdict(best),
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash,
        "manifest_sha256": manifest_sha256,
        "feature_set_hash": spectrum_bundle.feature_set_hash,
    }
    if isinstance(normalizer, SpectrumStandardizer):
        checkpoint["feature_mean"] = normalizer.mean
        checkpoint["feature_std"] = normalizer.std
    torch.save(checkpoint, run_directory / "checkpoint.pt")
    return RunResult(run_directory=run_directory, metrics=metrics)
