"""Versioned dataset definitions shared by every downstream model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    """Static definition of a version-one classification cohort."""

    key: str
    display_name: str
    metadata_filename: str
    metadata_sha256: str
    raw_subdirectory: Path
    classes: tuple[str, ...]
    expected_train: int
    expected_test: int
    require_registered_swc: bool = False

    @property
    def expected_total(self) -> int:
        return self.expected_train + self.expected_test


# The metadata hashes below identify the result-producing exact-loader copy at
# external/treemoco/data. Raw SWC content is byte-identical to EM/data for all
# selected samples. See docs/DATA_PROVENANCE.md for the legacy JML split.
DATASETS: dict[str, DatasetSpec] = {
    "act4": DatasetSpec(
        key="act4",
        display_name="ACT-4",
        metadata_filename="ACT_info_swc_10folds.csv",
        metadata_sha256="978c83d2d5af95041e675e1df7965e42d98558821e9f0f4cae8833f8fd46343d",
        raw_subdirectory=Path("raw") / "allen_cell_type" / "swc",
        classes=(
            "Isocortex_layer23",
            "Isocortex_layer4",
            "Isocortex_layer5",
            "Isocortex_layer6",
        ),
        expected_train=400,
        expected_test=95,
    ),
    "jml4": DatasetSpec(
        key="jml4",
        display_name="JML-4",
        metadata_filename="JML_info_swc_10folds.csv",
        metadata_sha256="957403ccdd45d6de20599d12349e4a9f787dde46534f88287b5c1472150a42a8",
        raw_subdirectory=Path("raw") / "janelia_mouselight" / "swc",
        classes=(
            "Isocortex_layer23",
            "Isocortex_layer5",
            "Isocortex_layer6",
            "VPM",
        ),
        expected_train=332,
        expected_test=76,
    ),
    "bil6": DatasetSpec(
        key="bil6",
        display_name="BIL-6",
        metadata_filename="BIL_info_swc_10folds.csv",
        metadata_sha256="19dbeeab1b23f7ce87ebbe098b3bf63164c8ba619c1d7605281714f75357fe08",
        raw_subdirectory=Path("raw") / "bil" / "swc",
        classes=(
            "CP",
            "Isocortex_layer23",
            "Isocortex_layer4",
            "Isocortex_layer5",
            "Isocortex_layer6",
            "VPM",
        ),
        expected_train=958,
        expected_test=242,
        require_registered_swc=True,
    ),
}


def get_dataset_spec(key: str) -> DatasetSpec:
    """Return a dataset definition using a case-insensitive key."""

    normalized = key.strip().lower().replace("-", "")
    aliases = {
        "act": "act4",
        "act4": "act4",
        "jml": "jml4",
        "jml4": "jml4",
        "bil": "bil6",
        "bil6": "bil6",
    }
    try:
        return DATASETS[aliases[normalized]]
    except KeyError as exc:
        valid = ", ".join(DATASETS)
        raise KeyError(f"Unknown dataset {key!r}; expected one of: {valid}") from exc
