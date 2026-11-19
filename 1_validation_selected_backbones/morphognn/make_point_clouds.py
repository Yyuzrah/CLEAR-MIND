# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Sample the MorphoGNN point clouds (1,024 points per neuron and seed) and the Submitted AG spectrum.

The sampling seed of every neuron and training seed is listed in data/point_cloud_seeds.csv, so the
clouds (and their jitter) are identical across the three settings and across reruns.
The Exact quotient AG spectrum for setting C is taken from the Tree-LSTM cache, so run
treelstm/prepare.py first.
Usage (from the package root):  python 1_validation_selected_backbones/morphognn/make_point_clouds.py
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import pandas as pd
from clearmind.paths import DATA, SEEDS, splits, swc_path

CACHE = HERE / "cache"


def initialize():
    global parse, Extractor, SAMPLE_SEED
    from clearmind.submission_morphognn import MorphoPointDataset, TropicalGeometricExtractor
    parse = lambda path: MorphoPointDataset._parse_swc(None, path)
    Extractor = TropicalGeometricExtractor
    table = pd.read_csv(DATA / "point_cloud_seeds.csv")
    SAMPLE_SEED = {(r.dataset, r.name, r.seed): int(r.sample_seed) for r in table.itertuples()}


def one(row):
    dataset, name, split = row
    dest = CACHE / dataset / (name + ".npz")
    if dest.exists():
        return
    graph, coords, root = parse(str(swc_path(dataset, name)))
    spec = Extractor(graph, coords, root).get_spectral_signature()
    assert np.isfinite(spec).all() and spec.sum() > 0
    x = np.array([coords[v] - coords[root] for v in graph])
    x /= np.linalg.norm(x, axis=1).max() + 1e-5
    arrays = {"spec": spec.astype("f4")}
    for seed in SEEDS:
        rng = np.random.RandomState(SAMPLE_SEED[dataset, name, seed])
        replace = len(x) < 1024
        pts = x[rng.choice(len(x), 1024, replace=replace)]
        if split == "train" or replace:
            pts += rng.normal(0, .001, size=pts.shape)
        arrays[str(seed)] = np.ascontiguousarray(pts.T, dtype="f4")
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as fh:
        np.savez(fh, **arrays)
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
    rows = list(df[["dataset", "name", "split"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        for i, _ in enumerate(pool.map(one, rows, chunksize=3), 1):
            if i % 200 == 0:
                print("sampled", i, "/", len(rows), flush=True)
    print("done:", len(rows), "neurons in", CACHE)


if __name__ == "__main__":
    main()
