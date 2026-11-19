"""Tree-structured adapter and batching for TreeLSTM-arch3."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx
import numpy as np
import torch

from em_connectome.data.records import NeuronRecord
from em_connectome.spectral.arakelov_green import ArakelovGreenSpectrum


@dataclass(frozen=True)
class TreeLSTMSample:
    sample_id: str
    x: torch.Tensor
    parent_indices: torch.Tensor
    levels: list[torch.Tensor]
    root_idx: int
    spectral_sig: torch.Tensor
    label: int


@dataclass(frozen=True)
class TreeBatch:
    sample_ids: list[str]
    x: torch.Tensor
    parent_indices: torch.Tensor
    levels: list[torch.Tensor]
    batch_indices: torch.Tensor
    root_indices: torch.Tensor
    spectral_sig: torch.Tensor
    labels: torch.Tensor
    total_nodes: int

    def to(self, device: torch.device) -> TreeBatch:
        return TreeBatch(
            sample_ids=self.sample_ids,
            x=self.x.to(device),
            parent_indices=self.parent_indices.to(device),
            levels=[level.to(device) for level in self.levels],
            batch_indices=self.batch_indices.to(device),
            root_indices=self.root_indices.to(device),
            spectral_sig=self.spectral_sig.to(device),
            labels=self.labels.to(device),
            total_nodes=self.total_nodes,
        )


class TreeLSTMAdapter:
    def __init__(self, spectral_dim: int = 64):
        self.spectral_dim = spectral_dim

    def transform(self, record: NeuronRecord) -> TreeLSTMSample:
        if record.spectrum is None:
            raise ValueError(f"{record.sample_id} has no spectrum")
        spectrum = np.asarray(record.spectrum, dtype=np.float32)
        if spectrum.shape != (self.spectral_dim,):
            raise ValueError(
                f"{record.sample_id} spectrum has shape {spectrum.shape}; "
                f"expected ({self.spectral_dim},)"
            )

        skeleton, coordinates, root = ArakelovGreenSpectrum._build_skeleton(record.morphology)
        if len(skeleton) < 2:
            raise ValueError(f"{record.sample_id} skeleton has fewer than two nodes")
        node_types = {
            int(node): int(record.morphology.node_types[index])
            for index, node in enumerate(record.morphology.node_ids)
        }
        nx.set_node_attributes(
            skeleton,
            {node: node_types.get(int(node), 3) for node in skeleton},
            "type",
        )
        node_list = list(skeleton.nodes())
        mapping = {node: index for index, node in enumerate(node_list)}

        directed = nx.DiGraph()
        directed.add_nodes_from(node_list)
        for parent, child in nx.bfs_edges(skeleton, source=root):
            directed.add_edge(child, parent)

        in_degrees = dict(directed.in_degree())
        queue = [node for node in directed.nodes() if in_degrees[node] == 0]
        levels: list[list[int]] = []
        while queue:
            levels.append(queue)
            next_queue: list[int] = []
            for node in queue:
                for successor in directed.successors(node):
                    in_degrees[successor] -= 1
                    if in_degrees[successor] == 0:
                        next_queue.append(successor)
            queue = next_queue

        parent_indices = np.zeros(len(node_list), dtype=np.int64)
        for child, parent in directed.edges():
            parent_indices[mapping[child]] = mapping[parent]
        parent_indices[mapping[root]] = len(node_list)

        features = np.zeros((len(node_list), 5), dtype=np.float64)
        root_coordinates = coordinates[root]
        for old_id, new_id in mapping.items():
            features[new_id, 0:3] = coordinates[old_id] - root_coordinates
            features[new_id, 3] = skeleton.nodes[old_id].get("type", 3)
            features[new_id, 4] = skeleton.degree[old_id]
        max_radius = np.max(np.linalg.norm(features[:, 0:3], axis=1)) + 1e-5
        features[:, 0:3] /= max_radius
        features[:, 4] /= 10.0

        mapped_levels = [
            torch.tensor([mapping[node] for node in level], dtype=torch.long) for level in levels
        ]
        return TreeLSTMSample(
            sample_id=record.sample_id,
            x=torch.tensor(features, dtype=torch.float32),
            parent_indices=torch.tensor(parent_indices, dtype=torch.long),
            levels=mapped_levels,
            root_idx=mapping[root],
            spectral_sig=torch.tensor(spectrum, dtype=torch.float32),
            label=record.label_id,
        )

    def transform_many(self, records: Iterable[NeuronRecord]) -> list[TreeLSTMSample]:
        return [self.transform(record) for record in records]


def collate_treelstm_samples(batch: list[TreeLSTMSample]) -> TreeBatch:
    if not batch:
        raise ValueError("cannot collate an empty TreeLSTM batch")
    total_nodes = sum(sample.x.shape[0] for sample in batch)
    x_list: list[torch.Tensor] = []
    batched_parents: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    root_indices: list[int] = []
    spectral_signatures: list[torch.Tensor] = []
    labels: list[int] = []
    offsets: list[int] = []

    offset = 0
    for graph_index, sample in enumerate(batch):
        node_count = sample.x.shape[0]
        x_list.append(sample.x)
        batch_indices.append(torch.full((node_count,), graph_index, dtype=torch.long))
        parent_indices = sample.parent_indices.clone()
        has_local_parent = parent_indices < node_count
        parent_indices[has_local_parent] += offset
        parent_indices[~has_local_parent] = total_nodes
        batched_parents.append(parent_indices)
        root_indices.append(sample.root_idx + offset)
        spectral_signatures.append(sample.spectral_sig)
        labels.append(sample.label)
        offsets.append(offset)
        offset += node_count

    max_depth = max(len(sample.levels) for sample in batch)
    batched_levels: list[torch.Tensor] = []
    for depth in range(max_depth):
        nodes = [
            sample.levels[depth] + offsets[index]
            for index, sample in enumerate(batch)
            if depth < len(sample.levels)
        ]
        batched_levels.append(torch.cat(nodes))

    return TreeBatch(
        sample_ids=[sample.sample_id for sample in batch],
        x=torch.cat(x_list),
        parent_indices=torch.cat(batched_parents),
        levels=batched_levels,
        batch_indices=torch.cat(batch_indices),
        root_indices=torch.tensor(root_indices, dtype=torch.long),
        spectral_sig=torch.stack(spectral_signatures),
        labels=torch.tensor(labels, dtype=torch.long),
        total_nodes=total_nodes,
    )
