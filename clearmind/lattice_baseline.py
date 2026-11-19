# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Explicit lattice baseline of Table 3: period matrix, Babai rounding and tropical distances.

PureTropicalJacobianExtractor is the extractor described in the appendix. It joins leaves
near the root to the root, builds the period matrix of the resulting cycles, and returns the
log-spectrum of the double-centered Gram matrix of the resulting distance.
"""
import os
import glob
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy.linalg import pinvh, eigh
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
import torch
import torch.nn as nn
import warnings


class PureTropicalJacobianExtractor:
    """Spectral signature of a weighted tree after closing leaves near the root into cycles.

    nx_tree: tree with edge attribute 'weight' (length). nodes_coords: node -> coordinate array.
    root_node: node that closure edges attach to.
    """
    def __init__(self, nx_tree: nx.Graph, nodes_coords: dict, root_node: int):
        self.G = nx_tree.copy()
        self.nodes_info = nodes_coords
        self.root_node = root_node

        # Remove degree-2 nodes; the reduced tree is kept for path and BFS computations.
        self._skeletonize_graph()
        self.skeleton_tree = self.G.copy()

        self.node_list = list(self.G.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.node_list)}

    def _skeletonize_graph(self):
        """Remove each degree-2 node other than the root and join its two neighbors by one edge.

        The new edge weight is the sum of the two removed weights; if the neighbors are
        already adjacent, the smaller weight is kept.
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
        """Return log1p of the top_k absolute eigenvalues of the double-centered Gram matrix of the distance.

        Every leaf within Euclidean distance epsilon of the root is joined to the root, giving
        g cycles (the closure edges are added to self.G). For g = 0 the distance is the
        shortest-path distance of the tree. For g > 0 it is the torus distance obtained by
        Babai rounding with the period matrix Q, plus the path length over bridges of weight
        at least tau. The output has length top_k and is zero-padded.
        """
        # Closure edges leaf-root with weight equal to the Euclidean distance.
        leaves = [n for n, d in self.G.degree() if d == 1 and n != self.root_node]
        closure_edges = []
        for leaf in leaves:
            dist = np.linalg.norm(self.nodes_info[leaf] - self.nodes_info[self.root_node])
            if dist < epsilon:
                self.G.add_edge(leaf, self.root_node, weight=dist)
                closure_edges.append((leaf, self.root_node))

        g = len(closure_edges)
        n = len(self.node_list)
        if n < 2:
            return np.zeros(top_k)

        if g == 0:
            # No cycles: shortest-path distance of the tree.
            D_Final = nx.floyd_warshall_numpy(self.G, nodelist=self.node_list)
        else:
            # Each undirected edge gets one index; both orientations map to it.
            edge_list = list(self.G.edges())
            edge_to_idx = {(u, v): i for i, (u, v) in enumerate(edge_list)}
            edge_to_idx.update({(v, u): i for i, (u, v) in enumerate(edge_list)})

            m = len(edge_list)
            L_diag = np.array([self.G.edges[e]['weight'] for e in edge_list])

            # Y_leaves (g, m): signed edge indicator of the tree path root -> leaf.
            Y_leaves = np.zeros((g, m))
            directed_tree = nx.bfs_tree(self.skeleton_tree, source=self.root_node)
            for i, (leaf, root) in enumerate(closure_edges):
                path = nx.shortest_path(directed_tree, source=self.root_node, target=leaf)
                for j in range(len(path) - 1):
                    u, v = path[j], path[j+1]
                    e_idx = edge_to_idx[(u, v)]
                    Y_leaves[i, e_idx] = 1.0 if (u, v) == edge_list[e_idx] else -1.0

            # C (g, m): cycle basis, tree path root -> leaf followed by the closure edge leaf -> root.
            C = np.zeros((g, m))
            for i, (leaf, root) in enumerate(closure_edges):
                C[i, :] = Y_leaves[i, :]
                closure_e_idx = edge_to_idx[(leaf, root)]
                C[i, closure_e_idx] = 1.0 if (leaf, root) == edge_list[closure_e_idx] else -1.0

            # V_universal (g, n): Abel-Jacobi coordinates of each node relative to the root,
            # accumulated along tree edges in BFS order.
            CL = C * L_diag
            V_universal = np.zeros((g, n))
            for u, v in nx.bfs_edges(self.skeleton_tree, source=self.root_node):
                u_idx = self.node_to_idx[u]
                v_idx = self.node_to_idx[v]
                e_idx = edge_to_idx[(u, v)]
                direction = 1.0 if (u, v) == edge_list[e_idx] else -1.0
                V_universal[:, v_idx] = V_universal[:, u_idx] + direction * CL[:, e_idx]

            # Period matrix Q = C diag(L) C^T (g, g); coordinates in the lattice basis are Q^+ V.
            Q = CL @ C.T
            Q_inv = pinvh(Q)

            V_tilde = Q_inv @ V_universal
            C_diff = V_tilde[:, :, np.newaxis] - V_tilde[:, np.newaxis, :]

            # Babai rounding: subtract the nearest integer vector from each pairwise difference.
            N_round = np.round(C_diff)
            Delta = C_diff - N_round

            # Torus distance (n, n): sqrt(Delta^T Q Delta) for each pair.
            D_Torus_sq = np.einsum('kij, kl, lij -> ij', Delta, Q, Delta)
            D_Torus = np.sqrt(np.maximum(D_Torus_sq, 0.0))

            # Backbone distance: path length over bridges of weight >= tau; cycle edges and
            # shorter bridges get weight 0.
            bridges = set(nx.bridges(self.G))
            F_backbone = nx.Graph()
            F_backbone.add_nodes_from(self.node_list)

            for u, v, data in self.G.edges(data=True):
                is_bridge = (u, v) in bridges or (v, u) in bridges
                weight = data['weight'] if (is_bridge and data['weight'] >= tau) else 0.0
                F_backbone.add_edge(u, v, weight=weight)

            D_Backbone = nx.floyd_warshall_numpy(F_backbone, nodelist=self.node_list)

            D_Final = D_Torus + D_Backbone

        D_Final = (D_Final + D_Final.T) / 2.0  # Remove floating-point asymmetry

        # Double centering: Gram = -1/2 J (D o D) J with J = I - 11^T / n, D o D the entrywise square.
        D_sq = D_Final ** 2
        J = np.eye(n) - np.ones((n, n)) / n
        Gram = -0.5 * J @ D_sq @ J

        # Absolute eigenvalues in decreasing order, compressed with log1p.
        eigenvalues = np.abs(eigh(Gram, eigvals_only=True))
        eigenvalues = np.sort(eigenvalues)[::-1]

        eigenvalues = np.log1p(eigenvalues)

        sig = np.zeros(top_k)
        actual_k = min(top_k, len(eigenvalues))
        sig[:actual_k] = eigenvalues[:actual_k]

        return sig
