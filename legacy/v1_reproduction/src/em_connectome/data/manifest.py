"""Build, load, hash, and audit exact-filename dataset manifests."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from .registry import DatasetSpec

MANIFEST_COLUMNS = (
    "dataset",
    "sample_id",
    "phase",
    "source_index",
    "fold",
    "label_id",
    "label_name",
    "relative_path",
    "filename",
    "is_reg",
    "content_sha256",
)


def sha256_file(path: Path) -> str:
    """Hash a file without loading it fully into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_label(value: object) -> str:
    """Match the label normalization used by the result notebooks."""

    return str(value).replace("2/3", "23")


@dataclass(frozen=True)
class ManifestEntry:
    dataset: str
    sample_id: str
    phase: str
    source_index: int
    fold: int
    label_id: int
    label_name: str
    relative_path: str
    filename: str
    is_reg: bool
    content_sha256: str

    @classmethod
    def from_csv_row(cls, row: dict[str, str]) -> ManifestEntry:
        missing = [column for column in MANIFEST_COLUMNS if column not in row]
        if missing:
            raise ValueError(f"Manifest row is missing columns: {missing}")
        return cls(
            dataset=row["dataset"],
            sample_id=row["sample_id"],
            phase=row["phase"],
            source_index=int(row["source_index"]),
            fold=int(row["fold"]),
            label_id=int(row["label_id"]),
            label_name=row["label_name"],
            relative_path=row["relative_path"],
            filename=row["filename"],
            is_reg=row["is_reg"].strip().lower() == "true",
            content_sha256=row["content_sha256"],
        )

    def to_csv_row(self) -> dict[str, object]:
        row = asdict(self)
        row["is_reg"] = "true" if self.is_reg else "false"
        return row


def _read_metadata(metadata_path: Path) -> Iterator[tuple[int, dict[str, str]]]:
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"structure_merge__acronym", "model__fold", "swc__fname"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{metadata_path} is missing columns: {sorted(missing)}")
        yield from enumerate(reader)


def build_entries(data_root: Path, spec: DatasetSpec) -> list[ManifestEntry]:
    """Select the exact version-one cohort and hash each selected SWC."""

    root = data_root.resolve()
    metadata_path = root / "info" / spec.metadata_filename
    raw_directory = root / spec.raw_subdirectory
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    if not raw_directory.is_dir():
        raise FileNotFoundError(f"SWC directory not found: {raw_directory}")

    actual_metadata_hash = sha256_file(metadata_path)
    if actual_metadata_hash != spec.metadata_sha256:
        raise ValueError(
            f"Metadata hash mismatch for {spec.key}: expected {spec.metadata_sha256}, "
            f"found {actual_metadata_hash}. See docs/DATA_PROVENANCE.md."
        )

    label_map = {label: index for index, label in enumerate(spec.classes)}
    entries: list[ManifestEntry] = []
    for source_index, row in _read_metadata(metadata_path):
        label_name = canonical_label(row["structure_merge__acronym"])
        if label_name not in label_map:
            continue

        fold = int(float(row["model__fold"]))
        if fold not in range(10):
            raise ValueError(f"Invalid fold {fold} at metadata row {source_index}")
        phase = "train" if fold < 8 else "test"
        filename = str(row["swc__fname"])
        path = raw_directory / filename
        if not path.is_file():
            continue

        is_reg = filename.lower().endswith("_reg.swc")
        entries.append(
            ManifestEntry(
                dataset=spec.key,
                sample_id=f"{spec.key}/{filename}",
                phase=phase,
                source_index=source_index,
                fold=fold,
                label_id=label_map[label_name],
                label_name=label_name,
                relative_path=path.relative_to(root).as_posix(),
                filename=filename,
                is_reg=is_reg,
                content_sha256=sha256_file(path),
            )
        )

    validate_entries(entries, spec)
    return entries


def validate_entries(entries: Iterable[ManifestEntry], spec: DatasetSpec) -> dict[str, object]:
    """Validate cohort size, labels, folds, IDs, and train/test isolation."""

    rows = list(entries)
    if not rows:
        raise ValueError(f"Manifest for {spec.key} is empty")
    if any(row.dataset != spec.key for row in rows):
        raise ValueError(f"Manifest contains a row from another dataset than {spec.key}")

    sample_ids = [row.sample_id for row in rows]
    paths = [row.relative_path for row in rows]
    hashes = [row.content_sha256 for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Duplicate sample_id in {spec.key}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"Duplicate SWC path in {spec.key}")
    if len(hashes) != len(set(hashes)):
        raise ValueError(f"Duplicate SWC content in {spec.key}")

    train = [row for row in rows if row.phase == "train"]
    test = [row for row in rows if row.phase == "test"]
    if len(train) != spec.expected_train or len(test) != spec.expected_test:
        raise ValueError(
            f"{spec.key} split mismatch: expected {spec.expected_train}/{spec.expected_test}, "
            f"found {len(train)}/{len(test)}"
        )
    if any(row.fold not in range(8) for row in train):
        raise ValueError(f"{spec.key} training split contains fold 8 or 9")
    if any(row.fold not in {8, 9} for row in test):
        raise ValueError(f"{spec.key} test split contains a fold below 8")
    if {row.label_id for row in rows} != set(range(len(spec.classes))):
        raise ValueError(f"{spec.key} is missing one or more configured labels")
    if any(spec.classes[row.label_id] != row.label_name for row in rows):
        raise ValueError(f"{spec.key} label IDs do not match the registry order")
    if spec.require_registered_swc and not all(row.is_reg for row in rows):
        raise ValueError(f"{spec.key} contains a non-registered SWC")

    return {
        "rows": len(rows),
        "split_counts": dict(sorted(Counter(row.phase for row in rows).items())),
        "fold_counts": {
            str(key): value for key, value in sorted(Counter(row.fold for row in rows).items())
        },
        "label_counts": dict(sorted(Counter(row.label_name for row in rows).items())),
        "registered_fraction": sum(row.is_reg for row in rows) / len(rows),
    }


def write_manifest(path: Path, entries: Iterable[ManifestEntry]) -> str:
    """Write a deterministic CSV and return its SHA-256."""

    rows = list(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(row.to_csv_row() for row in rows)
    return sha256_file(path)


def load_manifest(path: Path) -> list[ManifestEntry]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [ManifestEntry.from_csv_row(row) for row in csv.DictReader(handle)]


def resolve_data_path(data_root: Path, entry: ManifestEntry) -> Path:
    """Resolve a manifest path while preventing traversal outside data_root."""

    root = data_root.resolve()
    candidate = (root / Path(entry.relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Manifest path escapes the data root: {entry.relative_path}") from exc
    return candidate


def audit_local_files(
    data_root: Path,
    entries: Iterable[ManifestEntry],
    *,
    verify_hashes: bool = True,
) -> dict[str, object]:
    """Audit file presence and, by default, byte identity against a manifest."""

    rows = list(entries)
    missing: list[str] = []
    mismatched: list[str] = []
    for row in rows:
        path = resolve_data_path(data_root, row)
        if not path.is_file():
            missing.append(row.sample_id)
        elif verify_hashes and sha256_file(path) != row.content_sha256:
            mismatched.append(row.sample_id)
    return {
        "rows": len(rows),
        "missing_count": len(missing),
        "hash_mismatch_count": len(mismatched),
        "missing_sample_ids": missing,
        "hash_mismatch_sample_ids": mismatched,
        "ok": not missing and not mismatched,
    }


def write_manifest_lock(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
