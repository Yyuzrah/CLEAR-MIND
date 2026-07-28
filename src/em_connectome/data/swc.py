"""Strict, dependency-light SWC parsing for the shared data layer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SWCTree:
    """Array representation of one SWC morphology in file order."""

    node_ids: NDArray[np.int64]
    node_types: NDArray[np.int64]
    coordinates: NDArray[np.float64]
    radii: NDArray[np.float64]
    parent_ids: NDArray[np.int64]
    parent_indices: NDArray[np.int64]
    root_index: int

    def __post_init__(self) -> None:
        size = len(self.node_ids)
        if self.coordinates.shape != (size, 3):
            raise ValueError("coordinates must have shape (n_nodes, 3)")
        for name in ("node_types", "radii", "parent_ids", "parent_indices"):
            if len(getattr(self, name)) != size:
                raise ValueError(f"{name} length does not match node_ids")
        if not 0 <= self.root_index < size:
            raise ValueError("root_index is outside the node array")


def parse_swc(path: Path) -> SWCTree:
    """Parse the seven standard SWC columns and preserve row order exactly."""

    node_ids: list[int] = []
    node_types: list[int] = []
    coordinates: list[tuple[float, float, float]] = []
    radii: list[float] = []
    parent_ids: list[int] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 7:
                raise ValueError(f"{path}:{line_number}: expected at least seven SWC columns")
            try:
                node_ids.append(int(float(fields[0])))
                node_types.append(int(float(fields[1])))
                coordinates.append((float(fields[2]), float(fields[3]), float(fields[4])))
                radii.append(float(fields[5]))
                parent_ids.append(int(float(fields[6])))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number}: invalid numeric SWC field") from exc

    if not node_ids:
        raise ValueError(f"{path} contains no SWC nodes")
    if len(node_ids) != len(set(node_ids)):
        raise ValueError(f"{path} contains duplicate node IDs")

    id_to_index = {node_id: index for index, node_id in enumerate(node_ids)}
    parent_indices: list[int] = []
    explicit_roots: list[int] = []
    for index, parent_id in enumerate(parent_ids):
        if parent_id == -1:
            parent_indices.append(-1)
            explicit_roots.append(index)
        elif parent_id in id_to_index:
            parent_indices.append(id_to_index[parent_id])
        else:
            # The historical loaders treated a missing parent as disconnected.
            parent_indices.append(-1)

    # Historical notebook parsers overwrote the root for every -1 row.
    root_index = explicit_roots[-1] if explicit_roots else 0
    return SWCTree(
        node_ids=np.asarray(node_ids, dtype=np.int64),
        node_types=np.asarray(node_types, dtype=np.int64),
        coordinates=np.asarray(coordinates, dtype=np.float64),
        radii=np.asarray(radii, dtype=np.float64),
        parent_ids=np.asarray(parent_ids, dtype=np.int64),
        parent_indices=np.asarray(parent_indices, dtype=np.int64),
        root_index=root_index,
    )
