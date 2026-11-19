# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Prop. A.11(iv) on random graphs (canonical measure, equilibrium identity, Green function, tree case,
extremal characterization) and the resistance formula for distance-regular graphs (Prop. A.44).
Usage (from the package root):  python 7_appendix_numerical_checks/verify_canonical_measure_and_drg.py
"""
import numpy as np, networkx as nx
random_tree = nx.random_labeled_tree if hasattr(nx, 'random_labeled_tree') else nx.random_tree
rng = np.random.default_rng(7)
worst = {"mucan_edge": 0, "equil": 0, "green": 0, "tree_mucan": 0, "max_char": 0}
for trial in range(200):
    n = int(rng.integers(3, 12)); T = random_tree(n, seed=int(rng.integers(1e9)))
    E = [(u, v, float(rng.uniform(0.2, 3))) for u, v in T.edges]
    tree_only = trial % 3 == 0
    if not tree_only:
        for _ in range(int(rng.integers(1, 4))):
            u, v = rng.choice(n, 2, replace=False); E.append((int(u), int(v), float(rng.uniform(0.2, 3))))
    W = np.zeros((n, n))
    for u, v, l in E: W[u, v] += 1 / l; W[v, u] += 1 / l
    L = np.diag(W.sum(1)) - W; Lp = np.linalg.pinv(L); d = np.diag(Lp); M = d[:, None] + d[None, :] - 2 * Lp
    mu = np.ones(n) / n + 0.5 * L @ d
    # edge formula mu(v) = 1 - 1/2 sum_{e at v} M(e-,e+)/l_e
    mu2 = np.ones(n)
    for u, v, l in E: mu2[u] -= 0.5 * M[u, v] / l; mu2[v] -= 0.5 * M[u, v] / l
    worst["mucan_edge"] = max(worst["mucan_edge"], np.abs(mu - mu2).max())
    Mm = M @ mu; worst["equil"] = max(worst["equil"], Mm.max() - Mm.min())
    tau = 0.5 * mu @ M @ mu
    P = np.eye(n) - np.outer(np.ones(n), mu)   # g_mu = (I - 1 mu^T) L^+ (I - mu 1^T)
    g = P @ Lp @ P.T
    worst["green"] = max(worst["green"], np.abs(g - (tau - 0.5 * M)).max())
    if tree_only:
        deg = np.array([T.degree(v) for v in range(n)])
        worst["tree_mucan"] = max(worst["tree_mucan"], np.abs(mu - (1 - 0.5 * deg)).max())
    # max characterization: mu^T M mu <= 2 tau for 1^T mu = 1
    for _ in range(20):
        nu = rng.normal(size=n); nu -= nu.mean(); m2 = mu + nu
        worst["max_char"] = max(worst["max_char"], m2 @ M @ m2 - 2 * tau)
print({k: float(v) for k, v in worst.items()})

def drg_r(G):
    n = G.number_of_nodes(); D = nx.floyd_warshall_numpy(G); diam = int(D.max()); x = 0
    k = [int((D[x] == j).sum()) for j in range(diam + 1)]
    b = []
    for j in range(diam):
        y = int(np.where(D[x] == j)[0][0]); b.append(int(sum(1 for z in G[y] if D[x, z] == j + 1)))
    r = [0.0]
    for j in range(1, diam + 1):
        r.append(2.0 / n * sum(sum(k[i + 1:]) / (k[i] * b[i]) for i in range(j)))
    return D, r
for name, G in [("petersen", nx.petersen_graph()), ("cube", nx.hypercube_graph(3)), ("heawood", nx.heawood_graph()),
                ("rook4x4", nx.cartesian_product(nx.complete_graph(4), nx.complete_graph(4))), ("dodeca", nx.dodecahedral_graph()),
                ("C7", nx.cycle_graph(7)), ("K5", nx.complete_graph(5))]:
    G = nx.convert_node_labels_to_integers(G)
    A = nx.to_numpy_array(G); L = np.diag(A.sum(1)) - A; Lp = np.linalg.pinv(L); d = np.diag(Lp); M = d[:, None] + d[None, :] - 2 * Lp
    D, r = drg_r(G)
    pred = np.vectorize(lambda j: r[int(j)])(D)
    print(f"{name:9s} max|M - r_dist| = {np.abs(M - pred).max():.2e}  r = {np.round(r, 6)}")
