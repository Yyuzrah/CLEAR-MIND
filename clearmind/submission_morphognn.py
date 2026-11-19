# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""MorphoGNN model, point-cloud dataset and spectral extractor of the submission.

TropicalGeometricExtractor computes the AG spectrum of a neuron (Submitted AG, scipy pinv).
TropicalMorphoGNN fuses an EdgeConv point-cloud backbone with that spectrum.
MorphoPointDataset reads SWC files and builds (point cloud, spectrum, label) samples.
"""
import os
import glob
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import pinv, eigh
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F


class TropicalGeometricExtractor:
    """Submitted AG spectral signature of one neuron tree."""

    def __init__(self, nx_tree: nx.Graph, nodes_coords: dict, root_node: int):
        self.G = nx_tree.copy()
        self.nodes_info = nodes_coords
        self.root_node = root_node
        self._skeletonize_graph()
        self.node_list = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}

    def _skeletonize_graph(self):
        """Contract degree-2 nodes other than the root.

        The two incident edge lengths are summed; if the endpoints are already
        adjacent, the shorter length is kept.
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
        """Return the top_k largest eigenvalue magnitudes of M_AG, zero-padded.

        Leaves within distance epsilon of the root are joined to the root.
        Bridges shorter than tau get conductance 1e9, all other edges 1/length.
        M_AG is the effective-resistance matrix computed from the pseudoinverse
        of the weighted Laplacian (scipy pinv with its default cutoff).
        """
        leaves = [n for n, d in self.G.degree() if d == 1 and n != self.root_node]
        for leaf in leaves:
            dist = np.linalg.norm(self.nodes_info[leaf] - self.nodes_info[self.root_node])
            if dist < epsilon:
                self.G.add_edge(leaf, self.root_node, weight=dist)

        bridges = set(nx.bridges(self.G))
        n = len(self.node_list)
        if n < 2: return np.zeros(top_k)

        # Conductance matrix W
        W = np.zeros((n, n))
        for u, v, data in self.G.edges(data=True):
            i, j = self.node_to_idx[u], self.node_to_idx[v]
            is_noise = ((u, v) in bridges or (v, u) in bridges) and data['weight'] < tau

            # Avoid division by zero for zero-length edges
            weight = max(data['weight'], 1e-5)
            W[i, j] = W[j, i] = 1e9 if is_noise else (1.0 / weight)

        D = np.diag(np.sum(W, axis=1))
        L_pinv = pinv(D - W)
        diag = np.diag(L_pinv)
        # M_AG[i, j] = L+[i, i] + L+[j, j] - 2 L+[i, j]
        D_AG = diag[:, np.newaxis] + diag[np.newaxis, :] - 2 * L_pinv

        eigenvalues = np.sort(np.abs(eigh(D_AG, eigvals_only=True)))[::-1]
        sig = np.zeros(top_k)
        actual_k = min(top_k, len(eigenvalues))
        sig[:actual_k] = eigenvalues[:actual_k]
        return sig


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest neighbors of each point.

    x: (B, C, N). Returns (B, N, k); each point counts as its own neighbor.
    """
    # Negative squared Euclidean distance, so topk selects the nearest points
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (B, N, k)
    return idx

def get_graph_feature(x: torch.Tensor, k: int = 16, idx: torch.Tensor = None) -> torch.Tensor:
    """EdgeConv input: (x_j - x_i, x_i) for the k neighbors j of each point i.

    x: (B, C, N). Returns (B, 2C, N, k).
    """
    batch_size, _, num_points = x.size()
    if idx is None:
        idx = knn(x, k=k)

    # Flat indices into the (B * N, C) feature table
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


class TropicalMorphoGNN(nn.Module):
    """EdgeConv point-cloud backbone with late fusion of the projected AG spectrum."""

    def __init__(self, num_classes: int, k: int = 16, spectral_dim: int = 64):
        super().__init__()
        self.k = k

        # EdgeConv backbone
        self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(256, 1024, kernel_size=1, bias=False), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))

        # Projection of the AG spectrum
        self.spectral_proj = nn.Sequential(
            nn.Linear(spectral_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )

        # Classifier on the fused feature (2048 + 128 = 2176)
        self.linear1 = nn.Linear(2048 + 128, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.5)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=0.5)
        self.linear3 = nn.Linear(256, num_classes)

    def forward(self, x_pc: torch.Tensor, x_spec: torch.Tensor) -> torch.Tensor:
        """
        x_pc: (B, 3, 1024) point cloud, centered at the root and scaled to the unit ball
        x_spec: (B, 64) AG spectrum at the original scale
        Returns (B, num_classes) logits.
        """
        batch_size = x_pc.size(0)

        # Five-layer EdgeConv network (App. B.6): conv1-conv4 on dynamic k-NN graphs, max-pooled over the k neighbors; conv5 maps to 1024 channels
        x = get_graph_feature(x_pc, k=self.k)
        x = self.conv1(x).max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv2(x).max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv3(x).max(dim=-1, keepdim=False)[0]
        x = get_graph_feature(x, k=self.k)
        x = self.conv4(x).max(dim=-1, keepdim=False)[0]

        x = self.conv5(x)

        # Global max and mean pooling over points
        x_max = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x_avg = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        h_pointcloud = torch.cat((x_max, x_avg), dim=1) # (B, 2048)

        # Late fusion with the projected spectrum
        h_spectral = self.spectral_proj(x_spec) # (B, 128)
        h_fused = torch.cat([h_pointcloud, h_spectral], dim=-1) # (B, 2176)

        x = F.leaky_relu(self.bn6(self.linear1(h_fused)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        return self.linear3(x)


class MorphoPointDataset(torch.utils.data.Dataset):
    """Point clouds and AG spectra for ACT-4, BIL-6 or JML-4.

    Rows with model__fold < 8 form the training split, rows with fold >= 8 the test split.
    """

    def __init__(self, dataset_name: str, swc_dir: str, csv_path: str, phase: str = "train", num_points: int = 1024):
        self.num_points = num_points
        valid_classes = {
            "ACT-4": ["Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6"],
            "BIL-6": ["CP", "Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6", "VPM"],
            "JML-4": ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"]
        }
        target_classes = valid_classes[dataset_name]
        self.label_map = {cls_name: i for i, cls_name in enumerate(target_classes)}

        df = pd.read_csv(csv_path)
        local_swc_files = set(glob.glob(os.path.join(swc_dir, "*.swc")))
        self.data_list = []

        self.data_list = []
        print(f"[{phase.upper()}] {dataset_name}: sampling {num_points} points per neuron and computing AG spectra")

        for _, row in tqdm(df.iterrows(), total=len(df)):
            # "layer2/3" in the CSV is written "layer23" in the class names
            label = str(row['structure_merge__acronym']).replace("2/3", "23")
            if label not in target_classes: continue

            fold = float(row['model__fold'])
            if phase == "train" and fold >= 8.0: continue
            if phase == "test" and fold < 8.0: continue

            # Use only SWC files present under the exact CSV file name
            fname_csv = str(row['swc__fname'])
            exact_swc_path = os.path.join(swc_dir, fname_csv)

            if exact_swc_path not in local_swc_files:
                continue

            swc_path = exact_swc_path

            G_tree, nodes_info, root_node = self._parse_swc(swc_path)
            if len(G_tree) < 2: continue

            # AG spectrum on the original coordinates (keeps the absolute scale)
            engine = TropicalGeometricExtractor(G_tree, nodes_info, root_node)
            spectral_sig = engine.get_spectral_signature(top_k=64)
            if np.sum(spectral_sig) == 0: continue

            # Point cloud: move the root to the origin and scale to the unit ball.
            # Absolute size is carried only by spectral_sig.
            coords = np.array([nodes_info[n] for n in G_tree.nodes()])
            root_coord = nodes_info[root_node]

            coords = coords - root_coord

            max_radius = np.max(np.linalg.norm(coords, axis=1)) + 1e-5
            coords = coords / max_radius

            n_actual = coords.shape[0]
            if n_actual >= self.num_points:
                # Enough nodes: sample without replacement, add Gaussian jitter
                # (std 1e-3) during training only.
                indices = np.random.choice(n_actual, self.num_points, replace=False)
                sampled_coords = coords[indices, :]
                if phase == "train":
                    sampled_coords += np.random.normal(0, 1e-3, size=sampled_coords.shape)
            else:
                # Too few nodes: sample with replacement and always add jitter,
                # since duplicate points give zero distance in the k-NN graph.
                indices = np.random.choice(n_actual, self.num_points, replace=True)
                sampled_coords = coords[indices, :]
                jitter = np.random.normal(0, 1e-3, size=sampled_coords.shape)
                sampled_coords += jitter

            self.data_list.append({
                'point_cloud': torch.tensor(sampled_coords, dtype=torch.float32).transpose(0, 1), # (3, N)
                'spectral_sig': torch.tensor(spectral_sig, dtype=torch.float32),
                'label': self.label_map[label]
            })

    def _parse_swc(self, filepath: str):
        """Read an SWC file into a graph whose edge weights are segment lengths.

        Returns the largest connected component, the node coordinates and the
        root (the node with parent -1, otherwise the first node).
        """
        G = nx.Graph()
        nodes_info, edges = {}, []
        root_node = None
        with open(filepath, 'r') as f:
            for line in f:
                if not line.strip() or line.startswith('#'): continue
                parts = line.split()
                if len(parts) < 7: continue
                nid, x, y, z, pid = int(parts[0]), float(parts[2]), float(parts[3]), float(parts[4]), int(parts[6])
                nodes_info[nid] = np.array([x, y, z])
                G.add_node(nid)
                if pid == -1: root_node = nid
                else: edges.append((nid, pid))
        if root_node is None and len(G) > 0: root_node = list(G.nodes())[0]
        for u, v in edges:
            if u in nodes_info and v in nodes_info:
                G.add_edge(u, v, weight=max(np.linalg.norm(nodes_info[u]-nodes_info[v]), 1e-5))
        if len(G) > 0:
            lcc = max(nx.connected_components(G), key=len)
            G = G.subgraph(lcc).copy()
            if root_node not in G: root_node = list(G.nodes())[0]
        return G, nodes_info, root_node

    def __len__(self): return len(self.data_list)
    def __getitem__(self, idx): return self.data_list[idx]

def pointcloud_collate_fn(batch: list) -> dict:
    """Stack a list of samples into batched tensors."""
    pc = torch.stack([b['point_cloud'] for b in batch])
    sig = torch.stack([b['spectral_sig'] for b in batch])
    lbl = torch.tensor([b['label'] for b in batch], dtype=torch.long)
    return {'point_cloud': pc, 'spectral_sig': sig, 'label': lbl}
