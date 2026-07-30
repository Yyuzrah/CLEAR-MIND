"""Deterministic point-cloud adapter for the formal GNN-alpha runner."""

from __future__ import annotations

import hashlib
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from em_connectome.data.records import NeuronRecord

FEATURE_VERSION = "vbeta_topology6_bounded_fps_v1"
BIL_SOURCE_POLICY = "strict_csv_reg_v1"


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha1("|".join(map(str, parts)).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


class _UnionFind:
    def __init__(self, size: int):
        self.parent = np.arange(size, dtype=np.int64)
        self.rank = np.zeros(size, dtype=np.int8)

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = int(self.parent[item])
        return item

    def union(self, left: int, right: int) -> None:
        root_left, root_right = self.find(left), self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


@dataclass(frozen=True)
class GNNSample:
    sample_id: str
    point_cloud: NDArray[np.float32]
    tree_parent: NDArray[np.int64]
    spectrum: NDArray[np.float32]
    label: int


class GNNAdapter:
    """Match the deterministic ``baseline`` point cache used for five seeds.

    The partial V.alpha notebook sampled with global NumPy state. The formal
    five-seed BIL result used the later deterministic baseline cache. This
    adapter keeps the V.alpha centering/unit-sphere/random-1024 behavior while
    deriving a stable per-sample RNG exactly as that runner did.
    """

    def __init__(self, num_points: int = 1024, spectral_dim: int = 64):
        if num_points <= 0:
            raise ValueError("num_points must be positive")
        self.num_points = num_points
        self.spectral_dim = spectral_dim

    def transform(self, record: NeuronRecord) -> GNNSample:
        if record.spectrum is None:
            raise ValueError(f"{record.sample_id} has no spectrum")
        spectrum = np.asarray(record.spectrum, dtype=np.float32)
        if spectrum.shape != (self.spectral_dim,):
            raise ValueError(
                f"{record.sample_id} spectrum has shape {spectrum.shape}; "
                f"expected ({self.spectral_dim},)"
            )

        parsed = self._parse_features(record)
        display_name = {
            "act4": "ACT-4",
            "jml4": "JML-4",
            "bil6": "BIL-6",
        }[record.dataset]
        base_version = (
            f"{FEATURE_VERSION}_{BIL_SOURCE_POLICY}"
            if record.dataset == "bil6"
            else FEATURE_VERSION
        )
        rng = np.random.default_rng(
            stable_seed(
                display_name,
                record.swc_path.name,
                "random",
                self.num_points,
                base_version,
            )
        )
        node_count = len(parsed["features"])
        sampled = rng.choice(
            node_count,
            size=self.num_points,
            replace=node_count < self.num_points,
        ).astype(np.int64)
        tree_parent = self._sampled_tree_parent(parsed, sampled)
        return GNNSample(
            sample_id=record.sample_id,
            point_cloud=np.asarray(parsed["features"][sampled].T, dtype=np.float32),
            tree_parent=tree_parent,
            spectrum=spectrum.copy(),
            label=record.label_id,
        )

    def transform_many(self, records: Iterable[NeuronRecord]) -> list[GNNSample]:
        return [self.transform(record) for record in records]

    @staticmethod
    def _parse_features(record: NeuronRecord) -> dict[str, NDArray]:
        tree = record.morphology
        ids = np.asarray(tree.node_ids, dtype=np.int64)
        xyz_all = np.asarray(tree.coordinates, dtype=np.float64)
        parent_ids_all = np.asarray(tree.parent_ids, dtype=np.int64)
        if len(ids) < 2:
            raise ValueError(f"{record.sample_id} has fewer than two nodes")

        id_to_index = {int(node_id): index for index, node_id in enumerate(ids)}
        raw_parent = np.full(len(ids), -1, dtype=np.int64)
        union_find = _UnionFind(len(ids))
        for child, parent_id in enumerate(parent_ids_all):
            parent = id_to_index.get(int(parent_id), -1)
            if parent >= 0 and parent != child:
                raw_parent[child] = parent
                union_find.union(child, parent)

        roots = np.asarray(
            [union_find.find(index) for index in range(len(ids))],
            dtype=np.int64,
        )
        component_ids, component_counts = np.unique(roots, return_counts=True)
        largest_root = int(component_ids[np.argmax(component_counts)])
        keep = np.flatnonzero(roots == largest_root)
        old_to_new = np.full(len(ids), -1, dtype=np.int64)
        old_to_new[keep] = np.arange(len(keep), dtype=np.int64)
        xyz = xyz_all[keep]
        parent_ids = parent_ids_all[keep]
        raw_parent_kept = raw_parent[keep]
        parent = np.where(raw_parent_kept >= 0, old_to_new[raw_parent_kept], -1)

        declared_roots = np.flatnonzero(parent_ids == -1)
        root = int(declared_roots[0]) if len(declared_roots) else 0
        adjacency: list[list[int]] = [[] for _ in range(len(keep))]
        for child, parent_index in enumerate(parent):
            if parent_index >= 0 and parent_index != child:
                adjacency[child].append(int(parent_index))
                adjacency[int(parent_index)].append(child)

        degree = np.asarray([len(neighbors) for neighbors in adjacency], dtype=np.float32)
        rooted_parent = np.full(len(keep), -1, dtype=np.int64)
        depth = np.zeros(len(keep), dtype=np.float64)
        branch_order = np.zeros(len(keep), dtype=np.float32)
        order: list[int] = []
        queue: deque[int] = deque([root])
        visited = np.zeros(len(keep), dtype=bool)
        visited[root] = True
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in adjacency[node]:
                if visited[child]:
                    continue
                visited[child] = True
                rooted_parent[child] = node
                depth[child] = depth[node] + float(np.linalg.norm(xyz[child] - xyz[node]))
                branch_order[child] = branch_order[node] + float(degree[node] >= 3)
                queue.append(child)
        if not visited.all():
            raise ValueError(f"{record.sample_id} largest component is disconnected")

        centered = xyz - xyz[root]
        radius = max(float(np.linalg.norm(centered, axis=1).max()), 1e-5)
        normalized_xyz = centered / radius
        degree_feature = np.clip(degree, 0, 6) / 6.0
        depth_feature = depth / max(float(depth.max()), 1e-5)
        branch_feature = branch_order / max(float(branch_order.max()), 1.0)
        features = np.column_stack(
            (normalized_xyz, degree_feature, depth_feature, branch_feature)
        ).astype(np.float32)
        return {
            "features": features,
            "rooted_parent": rooted_parent,
            "bfs_order": np.asarray(order, dtype=np.int64),
        }

    @staticmethod
    def _sampled_tree_parent(
        parsed: dict[str, NDArray],
        sampled: NDArray[np.int64],
    ) -> NDArray[np.int64]:
        node_count = len(parsed["features"])
        first_position = np.full(node_count, -1, dtype=np.int64)
        for position, node in enumerate(sampled):
            if first_position[node] < 0:
                first_position[node] = position
        nearest_selected = np.full(node_count, -1, dtype=np.int64)
        parent_positions = np.arange(len(sampled), dtype=np.int64)
        rooted_parent = parsed["rooted_parent"]
        for node in parsed["bfs_order"]:
            parent = int(rooted_parent[node])
            inherited = nearest_selected[parent] if parent >= 0 else -1
            position = int(first_position[node])
            if position >= 0:
                parent_positions[position] = inherited if inherited >= 0 else position
                nearest_selected[node] = position
            else:
                nearest_selected[node] = inherited
        for position, node in enumerate(sampled):
            canonical = int(first_position[node])
            if canonical != position:
                parent_positions[position] = parent_positions[canonical]
        return parent_positions


def collate_gnn_samples(batch: list[GNNSample]) -> dict[str, object]:
    return {
        "sample_id": [sample.sample_id for sample in batch],
        "point_cloud": torch.from_numpy(np.stack([sample.point_cloud for sample in batch])),
        "tree_parent": torch.from_numpy(np.stack([sample.tree_parent for sample in batch])),
        "spectral_sig": torch.from_numpy(np.stack([sample.spectrum for sample in batch])),
        "label": torch.tensor([sample.label for sample in batch], dtype=torch.long),
    }
