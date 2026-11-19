# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Shared pieces of the Tree-LSTM experiment: data loading, batching, the three settings, prediction."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch
from clearmind.paths import CLASSES, SEEDS
from clearmind.submission_treelstm import GraphBatch, TopologicalTreeLSTM, DEVICE

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

CACHE = HERE / "cache"
OUTPUTS = HERE / "outputs"
DATASETS = list(CLASSES)
ARMS = ["A", "B", "C"]  # A: no AG (spectral branch zeroed), B: Submitted AG, C: Exact quotient AG
PATIENCE, MAX_EPOCH, BATCH = 25, 150, 128


def tag(dataset, seed, arm):
    return f"{dataset}_{seed}_{arm}"


def read_set(dataset, df):
    samples = []
    for row in df.itertuples():
        with np.load(CACHE / dataset / (row.name + ".npz")) as a:
            assert int(a["label"]) == row.label and int(a["fold"]) == row.fold
            samples.append(dict(x=a["x"], p=a["parent"], level=a["levels"],
                                depth=np.repeat(np.arange(len(a["counts"])), a["counts"]), root=int(a["root"]),
                                B=a["B"], C=a["C"], label=row.label, name=row.name, fold=row.fold))
    return samples


def make_batch(samples, index, arm):
    rows = [samples[int(i)] for i in index]
    sizes = np.array([len(r["p"]) for r in rows], dtype=np.int64)
    n = int(sizes.sum())
    offset = sizes.cumsum() - sizes
    parents = np.concatenate([np.where(r["p"] < s, r["p"] + o, n) for r, s, o in zip(rows, sizes, offset)])
    order = np.concatenate([r["level"] + o for r, o in zip(rows, offset)])
    dep = np.concatenate([r["depth"] for r in rows])
    sorted_ids = order[np.argsort(dep, kind="stable")]
    levels = list(torch.from_numpy(sorted_ids).split(np.bincount(dep).tolist()))
    return GraphBatch(torch.from_numpy(np.concatenate([r["x"] for r in rows])),
                      torch.from_numpy(parents), levels,
                      torch.tensor([r["root"] + int(o) for r, o in zip(rows, offset)]),
                      torch.from_numpy(np.stack([r["C" if arm == "C" else "B"] for r in rows])),
                      torch.tensor([r["label"] for r in rows]), n).to(DEVICE)


class PairModel(TopologicalTreeLSTM):
    """The submission's Tree-LSTM. Setting A feeds zeros to the fusion layer in place of the spectral branch,
    so all settings have the same width and parameters."""

    def __init__(self, dataset, arm):
        super().__init__(len(CLASSES[dataset]))
        self.arm = arm

    def forward(self, b):
        if self.arm != "A":
            return super().forward(b)
        h = self.treelstm(b.x, b.parent_indices, b.levels, b.total_nodes)
        root = self.graph_proj(h[b.root_indices])
        return self.classifier(torch.cat((root, torch.zeros_like(root)), dim=1))


@torch.no_grad()
def predict(model, rows, arm):
    model.eval()
    return torch.cat([model(make_batch(rows, range(i, min(i + BATCH, len(rows))), arm)).argmax(1).cpu()
                      for i in range(0, len(rows), BATCH)]).numpy()
