# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Training-free BREC evaluation of the AG spectrum and the Laplacian spectrum on all 32 relabelings.

BREC stores, for every pair, 32 relabelings of both graphs (64 graphs, interleaved G, H, G, H, ...),
followed by a second block of 400 x 64 graphs for the reliability check (relabelings of the same graph).
For each pair and each spectrum this script reports
  diff         relative difference of the sorted absolute spectra of G and H (first relabeling)
  pass         diff > 1e-8
  inconsistent whether the pass/fail decision differs between the 32 relabelings
  relabel      largest relative change of a graph's spectrum across its 32 relabelings
  reliability  largest relative difference in the reliability block (should be ~0)
AG spectrum: eigenvalue magnitudes of M_AG = diag(L+) 1^T + 1 diag(L+)^T - 2 L+, with L+ built from the
exact eigendecomposition of L (null space = number of components). Laplacian spectrum: eigenvalues of L.
Usage (from the package root; needs data/BREC/brec_v3.npy from the BREC release):
  python 2_brec/spectrum_all_relabelings.py [--workers 4]
Runtime: about 30 s with 4 CPU workers. Writes rerun/ by default; the shipped outputs are in outputs/.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
# BREC graphs; set CLEARMIND_BREC to use a copy elsewhere.
DATA = Path(os.environ.get("CLEARMIND_BREC", HERE.parent / "data" / "BREC" / "brec_v3.npy"))
THRESHOLD = 1e-8
CATEGORIES = [("Basic", 0, 60), ("Simple regular", 60, 110), ("Strongly regular", 110, 160), ("Extension", 160, 260),
              ("CFI", 260, 360), ("4-vertex condition", 360, 380), ("Distance regular", 380, 400)]


def initialize():
    global graphs
    graphs = np.load(DATA, allow_pickle=True)
    assert len(graphs) >= 800 * 64


def spectra(encoded):
    graph = nx.from_graph6_bytes(encoded.encode() if isinstance(encoded, str) else encoded)
    A = nx.to_numpy_array(graph)
    freq, U = eigh(np.diag(A.sum(1)) - A)
    zeros = nx.number_connected_components(graph)
    pos = U[:, zeros:]
    P = (pos / freq[zeros:]) @ pos.T
    M = P.diagonal()[:, None] + P.diagonal()[None, :] - 2 * P
    return np.sort(np.abs(eigh(M, eigvals_only=True)))[::-1], freq


def relative(a, b):
    if len(a) != len(b):
        return 1.0
    return np.max(np.abs(a - b)) / max(1e-30, np.max(np.abs(a)), np.max(np.abs(b)))


def score_pair(pid):
    cache, blocks = {}, []
    for start in (pid * 64, (pid + 400) * 64):
        block = []
        for val in graphs[start:start + 64]:
            if val not in cache:
                cache[val] = spectra(val)
            block.append(cache[val])
        blocks.append(block)
    test, rel = blocks
    row = dict(pair=pid)
    for axis, method in enumerate(["AG", "Laplacian"]):
        diffs = np.array([relative(test[j][axis], test[j + 1][axis]) for j in range(0, 64, 2)])
        row[method + "_diff"] = diffs[0]
        row[method + "_pass"] = bool(diffs[0] > THRESHOLD)
        row[method + "_inconsistent"] = bool((diffs > THRESHOLD).any() != (diffs > THRESHOLD).all())
        row[method + "_relabel"] = max(relative(test[j % 2][axis], test[j][axis]) for j in range(64))
        row[method + "_reliability"] = max(relative(rel[j][axis], rel[j + 1][axis]) for j in range(0, 64, 2))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(HERE / "rerun"))
    args = ap.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as ex:
        result = pd.DataFrame(list(ex.map(score_pair, range(400), chunksize=4)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "spectrum_pairs.csv", index=False)
    summary = {}
    for method in ("AG", "Laplacian"):
        passed = result[result[method + "_pass"]]
        summary[method] = dict(
            separated=int(len(passed)),
            by_category={name: int(passed.pair.between(lo, hi - 1).sum()) for name, lo, hi in CATEGORIES},
            pairs_with_decision_changing_across_relabelings=int(result[method + "_inconsistent"].sum()),
            max_relative_change_across_relabelings=float(result[method + "_relabel"].max()),
            max_relative_difference_in_reliability_block=float(result[method + "_reliability"].max()),
            min_relative_difference_of_separated_pairs=float(passed[method + "_diff"].min()))
    ag = set(result.pair[result.AG_pass])
    lap = set(result.pair[result.Laplacian_pass])
    summary["separated_by_Laplacian_but_not_AG"] = sorted(int(p) for p in lap - ag)
    summary["separated_by_AG_but_not_Laplacian"] = sorted(int(p) for p in ag - lap)
    (out / "spectrum_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
