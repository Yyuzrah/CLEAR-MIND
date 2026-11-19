# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Locations and constants shared by all experiments."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
# Raw SWC files: data/raw/<dataset folder>/swc/<name>; set CLEARMIND_RAW to use another location.
RAW = Path(os.environ.get("CLEARMIND_RAW", DATA / "raw"))
DATASETS = {"ACT-4": "allen_cell_type", "JML-4": "janelia_mouselight", "BIL-6": "bil"}
CLASSES = {
    "ACT-4": ["Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6"],
    "JML-4": ["Isocortex_layer23", "Isocortex_layer5", "Isocortex_layer6", "VPM"],
    "BIL-6": ["CP", "Isocortex_layer23", "Isocortex_layer4", "Isocortex_layer5", "Isocortex_layer6", "VPM"],
}
SEEDS = [42, 114514, 1919810, 666, 1453]


def swc_path(dataset, name):
    return RAW / DATASETS[dataset] / "swc" / name


def splits():
    """One row per neuron: dataset, name, fold, label, split (train = folds 0-6, valid = fold 7, test = folds 8-9)."""
    return pd.read_csv(DATA / "splits.csv")
