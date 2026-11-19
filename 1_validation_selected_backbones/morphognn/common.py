# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Shared pieces of the MorphoGNN experiment: data loading, the three settings, prediction."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np
import torch
from clearmind.paths import CLASSES, SEEDS
from clearmind.submission_morphognn import TropicalMorphoGNN

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)
DEVICE = "cuda"

CACHE = HERE / "cache"
EXACT = HERE.parent / "treelstm" / "cache"  # Exact quotient AG spectra (key "C"), shared with Tree-LSTM
OUTPUTS = HERE / "outputs"
DATASETS = list(CLASSES)
ARMS = ["A", "B", "C"]  # A: no AG (spectral projection output zeroed), B: Submitted AG, C: Exact quotient AG
PATIENCE, MAX_EPOCH, BATCH = 25, 150, 32


def tag(dataset, seed, arm):
    return f"{dataset}_{seed}_{arm}"


def load_set(dataset, sub, seed, arm):
    """Point clouds for this seed, the spectrum of the setting (A and B read the Submitted AG spectrum), labels."""
    clouds, spec = [], []
    for name in sub.name:
        with np.load(CACHE / dataset / (name + ".npz")) as f:
            clouds.append(f[str(seed)])
            if arm in "AB":
                spec.append(f["spec"])
        if arm == "C":
            with np.load(EXACT / dataset / (name + ".npz")) as g:
                spec.append(g["C"])
    return tuple(torch.as_tensor(x, device=DEVICE) for x in (np.stack(clouds), np.stack(spec), sub.label.to_numpy()))


def build(dataset, arm):
    model = TropicalMorphoGNN(int(dataset[-1])).to(DEVICE)
    if arm == "A":
        model.spectral_proj.register_forward_hook(lambda module, inputs, output: torch.zeros_like(output))
    return model


@torch.no_grad()
def predict(net, values):
    net.eval()
    x, s, y = values
    return torch.cat([net(x[i:i + BATCH], s[i:i + BATCH]).argmax(1) for i in range(0, len(y), BATCH)])
