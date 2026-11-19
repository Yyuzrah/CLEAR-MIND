# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Validation-selected spectrum-only MLP (the MLP rows of Table 5 and Table R1).

An MLP 64 -> 96 -> 48 -> C is trained on the 64-dimensional spectrum alone, once on Submitted AG (setting B)
and once on Exact quotient AG (setting C), for the five seeds. Folds 0-6 train, inputs are standardized with
training statistics, fold 7 selects the earliest epoch with the highest validation accuracy (patience 25,
at most 800 epochs), and folds 8-9 are scored once.

    python 5_spectrum_only_mlp/spectrum_mlp.py train   # 30 runs on CPU, about one minute; writes rerun/
    python 5_spectrum_only_mlp/spectrum_mlp.py test    # scores the shipped checkpoints of outputs/, writes rerun/
    python 5_spectrum_only_mlp/spectrum_mlp.py test --run-dir 5_spectrum_only_mlp/rerun   # scores your own run

`train` uses folds 0-7 only and writes selection.json when all 30 runs have finished; `test` reads
that file. The spectra come from 3_knn_descriptors/features.npz.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import f1_score
from clearmind.paths import CLASSES, SEEDS
import knn

torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

DATASETS = list(CLASSES)
ARMS = {"B": "Submitted AG", "C": "Exact quotient AG"}
PATIENCE, MAX_EPOCH, BATCH = 25, 800, 32


def network(classes):
    return nn.Sequential(nn.Linear(64, 96), nn.LeakyReLU(.1), nn.Dropout(.05),
                         nn.Linear(96, 48), nn.LeakyReLU(.1), nn.Dropout(.05), nn.Linear(48, classes))


def tag(dataset, seed, arm):
    return f"{dataset}_{seed}_{arm}"


def spectra(arrays, arm):
    """The 64-dimensional spectrum in single precision, as given to the backbones."""
    key = "submitted_ag" if arm == "B" else knn.DEFAULT
    return arrays[key][:, :64].astype(np.float32).astype(float)


def standardized(x, train):
    mean = x[train].mean(0)
    std = x[train].std(0).clip(1e-8)
    return (x - mean) / std


def fit(name, seed, classes, x, y, v, vy, out):
    torch.manual_seed(seed)
    net = network(classes)
    optim = torch.optim.Adam(net.parameters(), lr=.003, weight_decay=1e-6)
    order = torch.Generator().manual_seed(seed)
    best = -1
    history = []
    for epoch in range(1, MAX_EPOCH + 1):
        net.train()
        for ix in torch.randperm(len(y), generator=order).split(BATCH):
            optim.zero_grad()
            nn.functional.cross_entropy(net(x[ix]), y[ix], label_smoothing=.05).backward()
            optim.step()
        net.eval()
        with torch.no_grad():
            correct = int((net(v).argmax(1) == vy).sum())
        history.append(correct)
        if correct > best:
            chosen, best = epoch, correct
            torch.save(net.state_dict(), out / "checkpoints" / (name + ".pt"))
        if epoch - chosen >= PATIENCE:
            break
    pd.DataFrame(dict(epoch=range(1, len(history) + 1), val_correct=history)
                 ).to_csv(out / "epoch_logs" / (name + ".csv"), index=False)
    print("trained", name, "best epoch", chosen, "stopped", epoch, flush=True)
    return dict(best_epoch=chosen, stop_epoch=epoch, best_val_correct=best, val_n=len(vy))


def train(table, arrays, out):
    for sub in ("checkpoints", "epoch_logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    selection = {}
    for ds in DATASETS:
        rows = ((table.dataset == ds) & (table.split != "test")).to_numpy()   # folds 0-7 only
        sub = table[rows]
        tr = (sub.split == "train").to_numpy()
        va = (sub.split == "valid").to_numpy()
        y = torch.LongTensor(sub.label.values[tr])
        vy = torch.LongTensor(sub.label.values[va])
        for arm in ARMS:
            z = standardized(spectra(arrays, arm)[rows], tr)
            x, v = torch.FloatTensor(z[tr]), torch.FloatTensor(z[va])
            for seed in SEEDS:
                selection[tag(ds, seed, arm)] = fit(tag(ds, seed, arm), seed, len(CLASSES[ds]), x, y, v, vy, out)
    (out / "selection.json").write_text(json.dumps(selection, indent=1))
    print("fold-7 selection written for", len(selection), "runs", flush=True)


def test(table, arrays, run, out):
    stored = run / "selection.json"
    runs = [tag(d, s, a) for d in DATASETS for a in ARMS for s in SEEDS]
    if not stored.exists() or any(r not in json.loads(stored.read_text()) for r in runs):
        raise SystemExit(f"{stored} is missing or incomplete; finish `spectrum_mlp.py train` first.")
    (out / "test_predictions").mkdir(parents=True, exist_ok=True)
    scores = []
    for ds in DATASETS:
        rows = (table.dataset == ds).to_numpy()
        sub = table[rows]
        tr = (sub.split == "train").to_numpy()
        te = (sub.split == "test").to_numpy()
        test_rows = sub[te]
        classes = len(CLASSES[ds])
        for arm in ARMS:
            x = torch.FloatTensor(standardized(spectra(arrays, arm)[rows], tr)[te])
            for seed in SEEDS:
                name = tag(ds, seed, arm)
                net = network(classes)
                net.load_state_dict(torch.load(run / "checkpoints" / (name + ".pt"), map_location="cpu"))
                net.eval()
                with torch.no_grad():
                    pred = net(x).argmax(1).numpy()
                pd.DataFrame(dict(dataset=ds, seed=seed, arm=arm, name=test_rows.name.values, fold=test_rows.fold.values,
                                  y=test_rows.label.values, pred=pred)).to_csv(out / "test_predictions" / (name + ".csv"), index=False)
                scores.append(dict(dataset=ds, arm=arm, seed=seed, test_n=len(test_rows),
                                   accuracy=100 * (test_rows.label.values == pred).mean(),
                                   macro_f1=100 * f1_score(test_rows.label.values, pred, average="macro", labels=range(classes),
                                                           zero_division=0)))
    per_seed = pd.DataFrame(scores)
    per_seed.to_csv(out / "per_seed.csv", index=False)
    # Five-seed mean and sample SD per dataset and spectrum.
    rows = []
    for arm, title in ARMS.items():
        for ds in DATASETS:
            g = per_seed[(per_seed.dataset == ds) & (per_seed.arm == arm)]
            rows.append(dict(model="MLP, " + title, dataset=ds, test_n=int(g.test_n.iloc[0]),
                             accuracy=g.accuracy.mean(), accuracy_sd=g.accuracy.std(),
                             macro_f1=g.macro_f1.mean(), macro_f1_sd=g.macro_f1.std()))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "table5_mlp_rows.csv", index=False, float_format="%.6f")
    print(summary.round(2).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "test"])
    parser.add_argument("--features", default=str(knn.FEATURES))
    parser.add_argument("--run-dir", default=str(HERE / "outputs"), help="test: folder with selection.json and checkpoints/")
    parser.add_argument("--out", default=str(HERE / "rerun"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, arrays = knn.load_features(args.features)
    if args.step == "train":
        train(table, arrays, out)
    else:
        test(table, arrays, Path(args.run_dir), out)


if __name__ == "__main__":
    main()
