# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Uniform-vote cosine k-NN, the descriptors behind every k-NN row, and the features file reader.

Selection rule (fold 7 only): k in {1, 3, 5, 10, 20} and raw versus training-standardized ('z') inputs;
the highest validation accuracy wins, ties go to the smaller k, then to raw inputs. Neighbors with equal
similarity keep the order of data/splits.csv (stable sort) and equal votes go to the lower class index.
"""
import numpy as np
from clearmind.paths import ROOT, splits
from descriptors import EPS, TAU

FEATURES = ROOT / "3_knn_descriptors" / "features.npz"
DEFAULT = "exact_ag_20_5"   # Exact quotient AG at eps=20, tau=5
K_GRID = [1, 3, 5, 10, 20]

# Each method is a list of (feature key, number of leading coordinates); every part is normalized separately.
METHODS = {"Submitted AG": [("submitted_ag", 64)], "Exact quotient AG": [(DEFAULT, 64)],
           "Laplacian high": [("lap_high", 64)], "Laplacian low": [("lap_low", 64)], "Sholl": [("sholl", 64)],
           "Radial TMD": [("tmd", 64)], "Five statistics": [("stats", 5)]}
CLASSICAL = METHODS["Sholl"] + METHODS["Radial TMD"] + METHODS["Five statistics"]
METHODS["Classical combination"] = CLASSICAL
METHODS["Classical + Exact quotient AG"] = CLASSICAL + METHODS["Exact quotient AG"]
for e in EPS:
    for t in TAU:
        METHODS[f"eps={e},tau={t}"] = [(f"exact_ag_{e}_{t}", 64)]   # Tables 4 and R4
for k in (16, 32, 64, 128):
    METHODS[f"K={k}"] = [(DEFAULT, k)]                               # Table R5
METHODS["Full-tree H0"] = [("full_tree_h0", 64)]
METHODS["Lattice pipeline"] = [("lattice", 64)]                      # Table 3, eps=20, tau=5

WIDTHS = {}
for _parts in METHODS.values():
    for _key, _k in _parts:
        WIDTHS[_key] = max(_k, WIDTHS.get(_key, 0))


def load_features(path=FEATURES):
    """The split table and a dict of float arrays, one row per neuron in data/splits.csv order."""
    table = splits()
    with np.load(path) as file:
        assert file["dataset"].tolist() == table.dataset.tolist() and file["name"].tolist() == table.name.tolist()
        arrays = {key: np.asarray(file[key], dtype=float) for key in file.files if key not in ("dataset", "name")}
    for key, width in WIDTHS.items():
        assert arrays[key].shape == (len(table), width) and np.isfinite(arrays[key]).all(), key
    return table, arrays


def dataset_view(table, arrays, dataset):
    mask = (table.dataset == dataset).to_numpy()
    return table[mask].reset_index(drop=True), {key: value[mask] for key, value in arrays.items()}


def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-30)


def embedding(data, parts, train, scale):
    pieces = []
    for key, k in parts:
        x = data[key][:, :k]
        if scale == "z":
            x = (x - x[train].mean(0)) / np.maximum(x[train].std(0), 1e-8)
        pieces.append(unit(x))
    return unit(np.concatenate(pieces, axis=1))


def ranks(x, mask, train):
    return np.argsort(-(x[mask] @ x[train].T), axis=1, kind="stable")


def vote(neighbors, y, k, classes):
    top = y[neighbors[:, :k]]
    return np.column_stack([(top == c).sum(1) for c in range(classes)]).argmax(1)


def choose(data, parts, train, valid, y, classes):
    choices = []
    for scale in ("raw", "z"):
        neighbors = ranks(embedding(data, parts, train, scale), valid, train)
        for k in K_GRID:
            pred = vote(neighbors, y[train], k, classes)
            choices.append(dict(correct=int((pred == y[valid]).sum()), k=k, scale=scale))
    return min(choices, key=lambda c: (-c["correct"], c["k"], c["scale"] != "raw"))
