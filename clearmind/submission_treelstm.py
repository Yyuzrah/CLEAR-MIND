# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Tree-LSTM model, SWC parser and Submitted AG extractor of the submission.

SpectralTopologyExtractor computes the AG spectral signature (pinvh with its default cutoff).
MorphologyDataset parses SWC files into node features, parent indices and levels.
TopologicalTreeLSTM fuses the Tree-LSTM root state with the spectral signature.
"""
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import pinvh, eigh
from tqdm import tqdm
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {DEVICE}")


class SpectralTopologyExtractor:
    """AG spectral signature of one neuron tree.

    nx_tree has edge attribute 'weight' (segment length), nodes_coords maps
    node id to xyz coordinates, root_node is the soma node.
    """

    def __init__(self, nx_tree: nx.Graph, nodes_coords: dict, root_node: int):
        self.G = nx_tree.copy()
        self.nodes_info = nodes_coords
        self.root_node = root_node

        self._skeletonize_graph()

        # Copy taken before get_spectral_signature adds leaf-root edges, so it stays a tree.
        self.skeleton_tree = self.G.copy()

        self.node_list = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}

    def _skeletonize_graph(self):
        """Remove non-root degree-2 nodes; the two edges merge into one whose length is their sum.

        If the merged edge already exists, the shorter length is kept.
        """
        deg2_nodes = [n for n in self.G.nodes() if self.G.degree(n) == 2 and n != self.root_node]
        for n in deg2_nodes:
            neighbors = list(self.G.neighbors(n))
            if len(neighbors) == 2:
                u, v = neighbors
                new_w = self.G[u][n]['weight'] + self.G[n][v]['weight']
                if self.G.has_edge(u, v):
                    self.G[u][v]['weight'] = min(self.G[u][v]['weight'], new_w)
                else:
                    self.G.add_edge(u, v, weight=new_w)
                self.G.remove_node(n)

    def get_spectral_signature(self, epsilon: float = 20.0, tau: float = 5.0, top_k: int = 64) -> np.ndarray:
        """Return the top_k absolute eigenvalues of the AG distance matrix, largest first.

        epsilon: a leaf whose Euclidean distance to the root is below epsilon gets an
            edge to the root with that distance as length (self.G is modified in place).
        tau: bridges shorter than tau get conductance 1e9.
        top_k: signature length; zero-padded when the graph has fewer nodes.
        """
        # nx.Graph is simple: for a leaf already adjacent to the root, add_edge overwrites that edge's weight.
        leaves = [n for n, d in self.G.degree() if d == 1 and n != self.root_node]
        for leaf in leaves:
            dist = np.linalg.norm(self.nodes_info[leaf] - self.nodes_info[self.root_node])
            if dist < epsilon:
                self.G.add_edge(leaf, self.root_node, weight=dist)

        bridges = set(nx.bridges(self.G))
        n = len(self.node_list)
        if n < 2:
            return np.zeros(top_k)

        # Conductance matrix: 1/length per edge, 1e9 for bridges shorter than tau.
        W = np.zeros((n, n))
        for u, v, data in self.G.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            is_noise = ((u, v) in bridges or (v, u) in bridges) and data['weight'] < tau

            # Lower bound on length to avoid division by zero.
            weight = max(data['weight'], 1e-5)
            W[i, j] = W[j, i] = 1e9 if is_noise else (1.0 / weight)

        D = np.diag(np.sum(W, axis=1))

        # Pseudoinverse of the weighted Laplacian D - W with the default pinvh cutoff.
        L_pinv = pinvh(D - W)
        diag = np.diag(L_pinv)
        # D_AG[i, j] = L+[i, i] + L+[j, j] - 2 L+[i, j].
        D_AG = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * L_pinv

        eigenvalues = np.sort(np.abs(eigh(D_AG, eigvals_only=True)))[::-1]
        sig = np.zeros(top_k)
        actual_k = min(top_k, len(eigenvalues))
        sig[:actual_k] = eigenvalues[:actual_k]
        return sig


class ChildSumTreeLSTMCell(nn.Module):
    """Child-Sum Tree-LSTM cell applied to all nodes of one level at once.

    A single forget gate is computed from the summed child hidden state and
    applied to the summed child memory.
    """

    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.W_iou = nn.Linear(x_size, 3 * h_size, bias=True)
        self.U_iou = nn.Linear(h_size, 3 * h_size, bias=False)
        self.W_f = nn.Linear(x_size, h_size, bias=True)
        self.U_f = nn.Linear(h_size, h_size, bias=False)

    def forward(self, x: torch.Tensor, h_children_sum: torch.Tensor, c_children_sum: torch.Tensor):
        iou = self.W_iou(x) + self.U_iou(h_children_sum)
        i, o, u = torch.chunk(iou, 3, dim=1)
        i, o, u = torch.sigmoid(i), torch.sigmoid(o), torch.tanh(u)

        f = torch.sigmoid(self.W_f(x) + self.U_f(h_children_sum))

        c = i * u + f * c_children_sum
        h = o * torch.tanh(c)
        return h, c


class NativeTreeLSTM(nn.Module):
    """Bottom-up Tree-LSTM over a batched forest, one cell call per level."""

    def __init__(self, x_size: int, h_size: int):
        super().__init__()
        self.h_size = h_size
        self.cell = ChildSumTreeLSTMCell(x_size, h_size)

    def forward(self, x: torch.Tensor, parent_indices: torch.Tensor, levels: List[torch.Tensor], total_nodes: int):
        """x: (N, x_size); parent_indices: (N,), roots point to N; levels: leaves first.

        Returns the hidden state of every node, shape (N, h_size), with N = total_nodes.
        """
        # Row total_nodes collects the roots' outputs and is never read.
        h_sum = torch.zeros(total_nodes + 1, self.h_size, device=x.device)
        c_sum = torch.zeros(total_nodes + 1, self.h_size, device=x.device)
        h_out = torch.zeros(total_nodes, self.h_size, device=x.device)

        for level_nodes in levels:
            x_level = x[level_nodes]
            h_sum_level = h_sum[level_nodes]
            c_sum_level = c_sum[level_nodes]

            h_lvl, c_lvl = self.cell(x_level, h_sum_level, c_sum_level)
            h_out[level_nodes] = h_lvl

            # Add each node's h and c to the sums of its parent.
            p_indices = parent_indices[level_nodes]
            h_sum.index_add_(0, p_indices, h_lvl)
            c_sum.index_add_(0, p_indices, c_lvl)

        return h_out


@dataclass
class GraphBatch:
    """A batch of B trees stored as one forest with N = total_nodes nodes.

    x: (N, 5) node features; parent_indices: (N,), roots point to N;
    levels: node indices by height, leaves first; root_indices: (B,);
    spectral_sig: (B, 64); labels: (B,).
    """
    x: torch.Tensor
    parent_indices: torch.Tensor
    levels: List[torch.Tensor]
    root_indices: torch.Tensor
    spectral_sig: torch.Tensor
    labels: torch.Tensor
    total_nodes: int

    def to(self, device: torch.device):
        return GraphBatch(
            x=self.x.to(device),
            parent_indices=self.parent_indices.to(device),
            levels=[lvl.to(device) for lvl in self.levels],
            root_indices=self.root_indices.to(device),
            spectral_sig=self.spectral_sig.to(device),
            labels=self.labels.to(device),
            total_nodes=self.total_nodes
        )


class MorphologyDataset(torch.utils.data.Dataset):
    """Neurons of ACT-4, BIL-6 or JML-4 listed in csv_path, read from swc_dir.

    phase "train" keeps rows with model__fold < 8, phase "val" keeps model__fold >= 8.
    Each item holds node features, parent indices, levels, root index,
    spectral signature and label.
    """

    def __init__(self, dataset_name: str, swc_dir: str, csv_path: str, phase: str = "train"):
        valid_classes = {
            "ACT-4": ["Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6"],
            "BIL-6": ["CP", "Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6", "VPM"],
            "JML-4": ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"]
        }
        target_classes = valid_classes[dataset_name]
        self.label_map = {cls_name: i for i, cls_name in enumerate(target_classes)}

        df = pd.read_csv(csv_path)

        # Rows whose SWC file is not in swc_dir are skipped.
        local_swc_files = set(glob.glob(os.path.join(swc_dir, "*.swc")))

        self.data_list = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {dataset_name} ({phase})"):
            label = row['structure_merge__acronym']
            # "Isocortex_layer2/3" -> "Isocortex_layer23"
            if isinstance(label, str):
                label = label.replace("2/3", "23")
            if label not in target_classes:
                continue

            fold = float(row['model__fold'])

            if phase == "train" and fold >= 8.0:
                continue
            if phase == "val" and fold < 8.0:
                continue

            fname_csv = str(row['swc__fname'])
            exact_swc_path = os.path.join(swc_dir, fname_csv)

            if exact_swc_path not in local_swc_files:
                continue

            swc_path = exact_swc_path

            G_tree, nodes_info, root_node = self._parse_swc(swc_path)

            if len(G_tree) < 2: continue

            engine = SpectralTopologyExtractor(G_tree, nodes_info, root_node)
            spectral_sig = engine.get_spectral_signature()
            if np.sum(spectral_sig) == 0: continue

            di_g = nx.DiGraph()
            di_g.add_nodes_from(engine.node_list)

            # Edges point from child to parent in the skeleton tree rooted at root_node.
            for u, v in nx.bfs_edges(engine.skeleton_tree, source=engine.root_node):
                di_g.add_edge(v, u)

            # Levels by height: a node is added once all its children are processed (leaves first).
            in_degrees = dict(di_g.in_degree())
            queue = [n for n in di_g.nodes() if in_degrees[n] == 0]
            levels = []
            while queue:
                levels.append(queue)
                next_q = []
                for n in queue:
                    for succ in di_g.successors(n):
                        in_degrees[succ] -= 1
                        if in_degrees[succ] == 0:
                            next_q.append(succ)
                queue = next_q

            # The root's parent index is the number of nodes (the extra slot used in NativeTreeLSTM).
            mapping = engine.node_to_idx
            parent_indices = np.zeros(len(engine.node_list), dtype=np.int64)
            for child, parent in di_g.edges():
                parent_indices[mapping[child]] = mapping[parent]
            parent_indices[mapping[engine.root_node]] = len(engine.node_list)

            # Node features: xyz relative to the root, SWC type (default 3), degree in the skeleton tree.
            feats = np.zeros((len(engine.node_list), 5))
            root_coords = nodes_info[engine.root_node]

            for old_id, new_id in mapping.items():
                feats[new_id, 0:3] = nodes_info[old_id] - root_coords
                feats[new_id, 3] = engine.skeleton_tree.nodes[old_id].get('type', 3)
                feats[new_id, 4] = engine.skeleton_tree.degree[old_id]

            # Coordinates divided by the largest distance to the root, degree divided by 10.
            max_radius = np.max(np.linalg.norm(feats[:, 0:3], axis=1)) + 1e-5
            feats[:, 0:3] /= max_radius
            feats[:, 4] /= 10.0

            mapped_levels = [torch.tensor([mapping[n] for n in lvl], dtype=torch.long) for lvl in levels]

            self.data_list.append({
                'x': torch.tensor(feats, dtype=torch.float32),
                'parent_indices': torch.tensor(parent_indices, dtype=torch.long),
                'levels': mapped_levels,
                'root_idx': mapping[engine.root_node],
                'spectral_sig': torch.tensor(spectral_sig, dtype=torch.float32),
                'label': self.label_map[label]
            })

    def _parse_swc(self, filepath: str):
        """Read an SWC file into an undirected graph with edge weight = segment length.

        The root is the node with parent -1 (the first node if there is none).
        Only the largest connected component is kept.
        Returns (G, nodes_info, root_node); nodes_info maps node id to xyz.
        """
        G = nx.Graph()
        nodes_info, edges = {}, []
        root_node = None
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                # SWC columns: id, type, x, y, z, radius, parent id.
                parts = line.split()
                if len(parts) < 7: continue
                nid, ntype = int(parts[0]), int(parts[1])
                x, y, z, pid = float(parts[2]), float(parts[3]), float(parts[4]), int(parts[6])
                nodes_info[nid] = np.array([x, y, z])
                G.add_node(nid, type=ntype)
                if pid == -1:
                    root_node = nid
                else:
                    edges.append((nid, pid))

        if root_node is None and len(G) > 0:
            root_node = list(G.nodes())[0]

        # Lengths below 1e-5 are raised to 1e-5 (duplicate points).
        for u, v in edges:
            if u in nodes_info and v in nodes_info:
                G.add_edge(u, v, weight=max(np.linalg.norm(nodes_info[u]-nodes_info[v]), 1e-5))

        if len(G) > 0:
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc).copy()
            if root_node not in G:
                root_node = list(G.nodes())[0]
        return G, nodes_info, root_node

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]


def graph_collate_fn(batch: List[dict]) -> GraphBatch:
    """Concatenate the trees of a batch into one forest (see GraphBatch)."""
    total_nodes = sum(b['x'].shape[0] for b in batch)
    x_list, batched_parents, root_indices, spectral_sigs, labels = [], [], [], [], []

    offset = 0
    offsets = []
    for b in batch:
        n = b['x'].shape[0]
        x_list.append(b['x'])

        p_idx = b['parent_indices'].clone()
        mask = p_idx < n
        p_idx[mask] += offset

        # Root entries point to the shared slot total_nodes.
        p_idx[~mask] = total_nodes
        batched_parents.append(p_idx)

        root_indices.append(b['root_idx'] + offset)
        spectral_sigs.append(b['spectral_sig'])
        labels.append(b['label'])

        offsets.append(offset)
        offset += n

    # Batch level d is the union of level d of every tree.
    max_depth = max(len(b['levels']) for b in batch)
    batched_levels = []
    for d in range(max_depth):
        lvl_nodes = []
        for i, b in enumerate(batch):
            if d < len(b['levels']):
                lvl_nodes.append(b['levels'][d] + offsets[i])
        batched_levels.append(torch.cat(lvl_nodes))

    return GraphBatch(
        x=torch.cat(x_list),
        parent_indices=torch.cat(batched_parents),
        levels=batched_levels,
        root_indices=torch.tensor(root_indices, dtype=torch.long),
        spectral_sig=torch.stack(spectral_sigs),
        labels=torch.tensor(labels, dtype=torch.long),
        total_nodes=total_nodes
    )


class TopologicalTreeLSTM(nn.Module):
    """Classifier on the Tree-LSTM root state and the spectral signature.

    Each input is projected to hidden_dim // 2; the concatenation goes to an MLP.
    """

    def __init__(self, num_classes: int, node_feat_dim: int = 5, hidden_dim: int = 128, spectral_dim: int = 64):
        super().__init__()
        self.treelstm = NativeTreeLSTM(x_size=node_feat_dim, h_size=hidden_dim)

        self.graph_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU()
        )

        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, batch: GraphBatch):
        h_all = self.treelstm(batch.x, batch.parent_indices, batch.levels, batch.total_nodes)
        h_root = h_all[batch.root_indices]

        h_graph = self.graph_proj(h_root)
        h_spec = self.spectral_proj(batch.spectral_sig)

        h_fused = torch.cat([h_graph, h_spec], dim=-1)
        return self.classifier(h_fused)


def train_and_evaluate(model: nn.Module, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, epochs: int = 150, patience: int = 25) -> float:
    """Train with AdamW and a cosine learning-rate schedule, evaluating on val_loader each epoch.

    Stops after `patience` epochs without a higher accuracy on val_loader and
    returns the highest accuracy on val_loader over all epochs.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_data in train_loader:
            batch = batch_data.to(DEVICE)
            optimizer.zero_grad()

            logits = model(batch)
            loss = criterion(logits, batch.labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch_data in val_loader:
                batch = batch_data.to(DEVICE)
                logits = model(batch)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch.labels).sum().item()
                total += batch.labels.size(0)

        acc = correct / total
        avg_loss = total_loss / len(train_loader)

        if acc > best_acc:
            best_acc = acc
            epochs_no_improve = 0
            status = "Best Record Updated"
        else:
            epochs_no_improve += 1
            status = f"Patience: {epochs_no_improve}/{patience}"

        print(f"Epoch {epoch:03d}/{epochs} | Train Loss: {avg_loss:.4f} | Val Acc: {acc:.4f} | Best Acc: {best_acc:.4f} | {status}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    return best_acc
