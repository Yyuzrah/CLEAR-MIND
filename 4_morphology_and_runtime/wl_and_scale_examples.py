# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""1-WL refinement on BREC and the two small examples of Appendix C.4 (physical scale, K_{3,3} versus the prism).

1. 1-WL color refinement, run until the coloring is stable, on the first relabeling of each of the
   400 BREC pairs, next to the AG-spectrum decision stored in 2_brec/outputs/spectrum_pairs.csv.
2. A 4-vertex arbor dilated by a factor 2 at fixed thresholds: the leading AG eigenvalue doubles, while
   the unit-sphere normalized coordinates change by about 7e-8, from the small offset added in the scaling.
3. K_{3,3} and the triangular prism (both 3-regular on 6 vertices, 1-WL equivalent): leading AG eigenvalues.

    python 4_morphology_and_runtime/wl_and_scale_examples.py [--out DIR]

Needs data/BREC/brec_v3.npy from the BREC release (see data/README.md); a few seconds.
Results go to 4_morphology_and_runtime/rerun/ by default; the paper's are in outputs/.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import argparse
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
import networkx as nx
from descriptors import quotient

# BREC graphs; set CLEARMIND_BREC to use a copy elsewhere.
BREC = Path(os.environ.get("CLEARMIND_BREC", HERE.parent / "data" / "BREC" / "brec_v3.npy"))
AG_PAIRS = HERE.parent / "2_brec" / "outputs" / "spectrum_pairs.csv"


def refine(G, H):
    """Joint 1-WL refinement of two graphs: (separated, rounds until the answer)."""
    graphs = [G, H]
    color = [{v: 0 for v in graph} for graph in graphs]
    for step in range(max(len(G) + len(H), 1)):
        hist = [Counter(labels.values()) for labels in color]
        if hist[0] != hist[1]:
            return True, step
        codes = {}
        new = []
        for graph, labels in zip(graphs, color):
            nxt = {}
            for v in graph:
                key = (labels[v], tuple(sorted(labels[w] for w in graph[v])))
                nxt[v] = codes.setdefault(key, len(codes))
            new.append(nxt)
        if len(codes) == len(set(color[0].values()) | set(color[1].values())):
            return False, step + 1
        color = new
    raise AssertionError("refinement did not stabilize")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(HERE / "rerun"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spectral = pd.read_csv(AG_PAIRS).set_index("pair")
    graphs = np.load(BREC, allow_pickle=True)
    rows = []
    for pid in range(400):
        enc = graphs[pid * 64:pid * 64 + 2]
        G, H = [nx.from_graph6_bytes(v.encode() if isinstance(v, str) else v) for v in enc]
        separated, steps = refine(G, H)
        rows.append(dict(pair=pid, wl=separated, rounds=steps, ag=bool(spectral.loc[pid, "AG_pass"])))
    table = pd.DataFrame(rows)
    table.to_csv(out / "wl_on_brec.csv", index=False)

    # The same augmented skeleton before and after dilation: all tips lie outside
    # epsilon = 20 and every edge is longer than tau = 5.
    xyz = np.array([[0., 0., 0.], [30., 0., 0.], [60., 25., 0.], [35., 45., 12.]])
    spec = []
    norm = []
    for factor in [1., 2.]:
        x = xyz * factor
        tree = nx.Graph()
        for a, b in [(0, 1), (1, 2), (1, 3)]:
            length = np.linalg.norm(x[a] - x[b])
            assert length > 5
            tree.add_edge(a, b, weight=length)
        assert all(np.linalg.norm(x[v]) > 20 for v in [2, 3])
        val, _ = quotient(tree, list(range(4)), [], False)
        spec.append(val["spec"])
        norm.append(x / (np.linalg.norm(x, axis=1).max() + 1e-5))
    assert np.allclose(spec[1], 2 * spec[0])

    pair = [nx.complete_bipartite_graph(3, 3), nx.circular_ladder_graph(3)]
    assert not refine(*pair)[0]
    leading = []
    for graph in pair:
        nx.set_edge_attributes(graph, 1., "weight")
        val, _ = quotient(graph, list(graph), [], False)
        leading.append(val["spec"][0])

    summary = dict(wl_separated=int(table.wl.sum()), ag_separated=int(table.ag.sum()),
                   ag_only=int((table.ag & ~table.wl).sum()), wl_max_rounds=int(table.rounds.max()),
                   bipartite_prism_leading=leading, morphology_leading=[s[0] for s in spec],
                   normalized_coordinate_max_change=float(abs(norm[0] - norm[1]).max()))
    (out / "wl_and_scale_examples.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
