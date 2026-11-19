# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Build the per-neuron Tree-LSTM inputs for all three settings.

For every neuron in data/splits.csv this stores the tree (node features, parents, leaf-first levels),
the Submitted AG spectrum (B, the submission's extractor) and the Exact quotient AG spectrum (C).
Usage (from the package root):  python 1_validation_selected_backbones/treelstm/prepare.py [--workers 4]
"""
import os
for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import networkx as nx
from clearmind.paths import splits, swc_path
from clearmind.exact_ag import exact_signature

CACHE = HERE / "cache"


def worker_init():
    global parse, Extractor
    from clearmind.submission_treelstm import MorphologyDataset, SpectralTopologyExtractor
    parse = lambda path: MorphologyDataset._parse_swc(None, path)
    Extractor = SpectralTopologyExtractor


def prepare_one(row):
    dataset, name, fold, label = row
    dest = CACHE / dataset / (name + ".npz")
    if dest.exists():
        return
    G, coords, root = parse(str(swc_path(dataset, name)))
    e = Extractor(G, coords, root)
    B = e.get_spectral_signature()  # augments e.G with chords; e.skeleton_tree stays the tree
    C = exact_signature(e)
    if not np.isfinite(B).all() or not np.isfinite(C).all():
        raise ValueError(name)
    n = len(e.node_list)
    mp = e.node_to_idx
    parent = np.zeros(n, dtype=np.int64)
    parent[mp[root]] = n
    children = [[] for _ in range(n)]
    for u, v in nx.bfs_edges(e.skeleton_tree, source=root):
        parent[mp[v]] = mp[u]
        children[mp[u]].append(mp[v])
    if sum(map(len, children)) != n - 1:
        raise ValueError("disconnected skeleton: " + name)
    # Leaf-first levels in the same order as the submission's batching.
    di = nx.DiGraph()
    di.add_nodes_from(e.node_list)
    di.add_edges_from((v, u) for u, v in nx.bfs_edges(e.skeleton_tree, root))
    deg = dict(di.in_degree())
    queue = [v for v in di if deg[v] == 0]
    flat, counts = [], []
    while queue:
        counts.append(len(queue))
        flat.extend(mp[v] for v in queue)
        nxt = []
        for v in queue:
            for u in di.successors(v):
                deg[u] -= 1
                if deg[u] == 0:
                    nxt.append(u)
        queue = nxt
    X = np.zeros((n, 5))
    for v, i in mp.items():
        X[i, :3] = coords[v] - coords[root]
        X[i, 3] = e.skeleton_tree.nodes[v].get("type", 3)
        X[i, 4] = e.skeleton_tree.degree[v]
    X[:, :3] /= np.linalg.norm(X[:, :3], axis=1).max() + 1e-5
    X[:, 4] /= 10
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, x=X.astype("f4"), parent=parent, levels=np.array(flat), counts=np.array(counts),
                 root=mp[root], B=B.astype("f4"), C=C.astype("f4"), fold=fold, label=label)
    os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N neurons per dataset (quick check)")
    args = ap.parse_args()
    df = splits()
    if args.limit:
        df = df.groupby("dataset", sort=False).head(args.limit)
    for d in df.dataset.unique():
        (CACHE / d).mkdir(parents=True, exist_ok=True)
    rows = list(df[["dataset", "name", "fold", "label"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init) as ex:
        for i, _ in enumerate(ex.map(prepare_one, rows, chunksize=3), 1):
            if i % 200 == 0:
                print("prepared", i, "/", len(rows), flush=True)
    print("done:", len(rows), "neurons in", CACHE)


if __name__ == "__main__":
    main()
