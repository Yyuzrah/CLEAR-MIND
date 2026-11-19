# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Worked 5-node example of Figure R1 (main text and Appendix C.6).

Path x-u-v-y with lengths 1, 2, 3 um, side tip v-b of 1 um, and a chord x-y of 2 um (tau = 0).
Checks that Kron reduction of u leaves M_AG on the core vertices (x, v, y, b) unchanged, that
M_AG(x, y) = 1.5 and d_Trop(x, y)^2 = d_Gamma(x, y) - M_AG(x, y) = 0.5, and prints M_AG and its spectrum.
Usage (from the package root):  python 7_appendix_numerical_checks/verify_fig_r1_example.py
"""
import networkx as nx
import numpy as np


def resistance(graph, order):
    """M_AG = diag(L+) 1^T + 1 diag(L+)^T - 2 L+ with conductances 1 / length."""
    L = nx.laplacian_matrix(graph, nodelist=order, weight="conductance").toarray().astype(float)
    P = np.linalg.pinv(L)
    d = np.diag(P)
    return d[:, None] + d[None, :] - 2 * P


def metric_graph(edges):
    g = nx.Graph()
    for a, b, length in edges:
        g.add_edge(a, b, length=length, conductance=1.0 / length)
    return g


core = ["x", "v", "y", "b"]
full = metric_graph([("x", "u", 1), ("u", "v", 2), ("v", "y", 3), ("v", "b", 1), ("x", "y", 2)])
kron = metric_graph([("x", "v", 3), ("v", "y", 3), ("v", "b", 1), ("x", "y", 2)])

M_full = resistance(full, core + ["u"])[:4, :4]
M = resistance(kron, core)
print("Kron reduction keeps M_AG on the core vertices: max difference %.1e" % np.abs(M_full - M).max())

d_gamma = nx.shortest_path_length(kron, "x", "y", weight="length")
# Real unit-current energy 2p^2 + 6(1 - p)^2 with current p on the chord (2 um) and 1 - p on the path via v (6 um).
p = np.linspace(0, 1, 400001)
energy = 2 * p ** 2 + 6 * (1 - p) ** 2
print("d_Gamma(x, y) = %g, M_AG(x, y) = %.4f (minimum of the energy at p = %.4f: %.4f)"
      % (d_gamma, M[0, 2], p[energy.argmin()], energy.min()))
print("d_Trop(x, y)^2 = d_Gamma - M_AG = %.4f" % (d_gamma - M[0, 2]))

np.set_printoptions(precision=3, suppress=True)
print("M_AG in the order (x, v, y, b), um:\n", M)
vals = np.sort(np.linalg.eigvalsh(M))[::-1]
print("eigenvalues:", " ".join("%.4f" % v for v in vals))
print("trace %.1e; positive eigenvalue equals the sum of the magnitudes of the others: %.4f = %.4f"
      % (vals.sum(), vals[0], -vals[1:].sum()))
print("signature (sorted magnitudes):", " ".join("%.3f" % v for v in np.sort(np.abs(vals))[::-1]))

expected = {("x", "v"): 1.875, ("x", "y"): 1.5, ("x", "b"): 2.875, ("v", "y"): 1.875, ("v", "b"): 1.0, ("y", "b"): 2.875}
err = max(abs(M[core.index(p), core.index(q)] - val) for (p, q), val in expected.items())
print("entries match Appendix C.6: max difference %.1e" % err)
assert err < 1e-12 and abs(d_gamma - M[0, 2] - 0.5) < 1e-12
