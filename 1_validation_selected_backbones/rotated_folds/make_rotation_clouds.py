# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""MorphoGNN point clouds for the four rotations.

Same sampling as morphognn/make_point_clouds.py: for every neuron and seed the 1,024 points are drawn
with the seed listed in data/point_cloud_seeds.csv, and jitter is added to the clouds of training
neurons and to every oversampled cloud. Only the set of training neurons changes with the rotation.
Usage (from the package root):  python 1_validation_selected_backbones/rotated_folds/make_rotation_clouds.py
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
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from clearmind.paths import DATA, SEEDS, splits, swc_path
from rotations import ROTATIONS, split_of

CACHE = HERE / "cache"


def initialize():
    global parse, SAMPLE_SEED
    from clearmind.submission_morphognn import MorphoPointDataset
    parse = lambda path: MorphoPointDataset._parse_swc(None, path)
    table = pd.read_csv(DATA / "point_cloud_seeds.csv")
    SAMPLE_SEED = {(r.dataset, r.name, r.seed): int(r.sample_seed) for r in table.itertuples()}


def clouds(dataset, name):
    """Per seed: the sampled points and the jitter that the sampling code would add."""
    graph, coords, root = parse(str(swc_path(dataset, name)))
    x = np.array([coords[v] - coords[root] for v in graph])
    x /= np.linalg.norm(x, axis=1).max() + 1e-5
    out = {}
    for seed in SEEDS:
        rng = np.random.RandomState(SAMPLE_SEED[dataset, name, seed])
        replace = len(x) < 1024
        pts = x[rng.choice(len(x), 1024, replace=replace)]
        noise = rng.normal(0, .001, size=pts.shape)
        out[seed] = (pts, noise, replace)
    return out


def one(row):
    dataset, name, fold = row
    todo = [r for r in ROTATIONS if not (CACHE / r / dataset / (name + ".npz")).exists()]
    if not todo:
        return
    per_seed = clouds(dataset, name)
    for r in todo:
        train = split_of(int(fold), r) == "train"
        arrays = {}
        for seed, (pts, noise, replace) in per_seed.items():
            p = pts + noise if (train or replace) else pts
            arrays[str(seed)] = np.ascontiguousarray(p.T, dtype="f4")
        dest = CACHE / r / dataset / (name + ".npz")
        tmp = dest.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            np.savez(fh, **arrays)
        os.replace(tmp, dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    df = splits()
    for r in ROTATIONS:
        for d in df.dataset.unique():
            (CACHE / r / d).mkdir(parents=True, exist_ok=True)
    rows = list(df[["dataset", "name", "fold"]].itertuples(index=False, name=None))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize) as pool:
        for i, _ in enumerate(pool.map(one, rows, chunksize=4), 1):
            if i % 300 == 0:
                print("sampled", i, "/", len(rows), flush=True)
    print("done:", len(rows), "neurons x", len(ROTATIONS), "rotations in", CACHE)


if __name__ == "__main__":
    main()
