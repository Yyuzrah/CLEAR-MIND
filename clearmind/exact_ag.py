# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Exact quotient AG signature (Algorithm 1 of the revised paper).

Bridges shorter than tau are contracted combinatorially, parallel conductances are summed, M_AG is
solved exactly on the quotient graph with a positive-definite linear solve, and the matrix is pulled
back to the core vertices. The signature is the 64 largest eigenvalue magnitudes, zero-padded.
"""
import numpy as np
import networkx as nx
from scipy.linalg import solve, eigh


def exact_signature(e, tau=5.0, K=64):
    """e: a SpectralTopologyExtractor after get_spectral_signature() has augmented e.G with chords."""
    graph = e.G
    short = [(a, b) for a, b in nx.bridges(graph) if graph[a][b]["weight"] < tau]
    forest = nx.Graph()
    forest.add_nodes_from(e.node_list)
    forest.add_edges_from(short)
    comp = {v: i for i, cc in enumerate(nx.connected_components(forest)) for v in cc}
    index = np.array([comp[v] for v in e.node_list])
    m = len(set(index))
    W = np.zeros((m, m))
    for u, v, attr in graph.edges(data=True):
        a, b = comp[u], comp[v]
        if a != b:
            w = 1 / max(attr["weight"], 1e-5)
            W[a, b] += w
            W[b, a] += w
    L = np.diag(W.sum(1)) - W
    J = np.full((m, m), 1 / m)
    P = solve(L + J, np.eye(m), assume_a="pos") - J
    M = np.diag(P)[:, None] + np.diag(P)[None, :] - 2 * P
    # Pulling M back to the core vertices repeats rows and columns; the nonzero spectrum of the
    # pulled-back matrix equals that of D^(1/2) M D^(1/2), D = class sizes. Missing eigenvalues are zeros.
    mult = np.sqrt(np.bincount(index, minlength=m))
    vals = np.sort(np.abs(eigh(mult[:, None] * M * mult[None, :], eigvals_only=True)))[::-1]
    sig = np.zeros(K)
    sig[:min(K, m)] = vals[:K]
    return sig
