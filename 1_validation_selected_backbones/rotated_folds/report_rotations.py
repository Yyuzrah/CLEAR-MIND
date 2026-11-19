# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Results of the rotated-fold evaluation (no GPU, a few seconds).

For each backbone, dataset and setting: test accuracy and macro-F1 per rotation, and pooled over
rotations R1-R4 (every neuron of folds 0-7 tested exactly once) and over all five splits (R1-R4 and the
Table 5 split with test folds 8-9: every neuron tested exactly once). Pooled metrics are computed per seed
on the pooled predictions, then averaged over the five seeds. Paired differences between settings come
with 95% paired neuron bootstrap intervals (10,000 draws; all five seed predictions of a neuron are kept
together; one generator seeded 20261106 per pooling, drawn in the order Tree-LSTM, MorphoGNN x ACT-4,
JML-4, BIL-6 x C-A, B-A, C-B).
Usage (from the package root):  python 1_validation_selected_backbones/rotated_folds/report_rotations.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from clearmind.paths import CLASSES, SEEDS
from rotations import ROTATIONS

BACKBONES = {"Tree-LSTM": "treelstm", "MorphoGNN": "morphognn"}
DATASETS = list(CLASSES)
ARMS = ["A", "B", "C"]
CONTRASTS = [("C", "A"), ("B", "A"), ("C", "B")]
OUT = HERE / "outputs"


def read(backbone, split, ds, seed, arm):
    if split == "T5":  # the Table 5 split (test folds 8-9)
        f = HERE.parent / BACKBONES[backbone] / "outputs" / "test_predictions" / f"{ds}_{seed}_{arm}.csv"
    else:
        f = OUT / split / BACKBONES[backbone] / ds / "test_predictions" / f"{ds}_{seed}_{arm}.csv"
    p = pd.read_csv(f)
    y = p["y"] if "y" in p else p["label"]
    return p.name.to_numpy(), y.to_numpy(), p.pred.to_numpy()


def interval(diff, groups, rng):
    code = pd.factorize(groups)[0]
    sizes = np.bincount(code)
    sums = np.bincount(code, weights=diff)
    draw = rng.integers(len(sizes), size=(10000, len(sizes)))
    means = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def pooled(splits_used, label):
    rng = np.random.default_rng(20261106)
    rows, pairs = [], []
    for backbone in BACKBONES:
        for ds in DATASETS:
            nclass = len(CLASSES[ds])
            hits, f1 = {}, {}
            for arm in ARMS:
                for seed in SEEDS:
                    names, ys, preds = [], [], []
                    for s in splits_used:
                        n, y, p = read(backbone, s, ds, seed, arm)
                        names.append(n), ys.append(y), preds.append(p)
                    n, y, p = np.concatenate(names), np.concatenate(ys), np.concatenate(preds)
                    if arm == "A" and seed == SEEDS[0]:
                        ref = n
                        assert len(set(n)) == len(n), "a neuron is tested twice"
                    assert np.array_equal(n, ref)
                    hits[arm, seed] = (y == p).astype(float) * 100
                    f1[arm, seed] = 100 * f1_score(y, p, labels=list(range(nclass)), average="macro", zero_division=0)
                    rows.append(dict(pooling=label, backbone=backbone, dataset=ds, arm=arm, seed=seed, test_n=len(y),
                                     accuracy=hits[arm, seed].mean(), macro_f1=f1[arm, seed]))
            for a, b in CONTRASTS:
                v = np.array([hits[a, s] - hits[b, s] for s in SEEDS])
                lo, hi = interval(v.mean(axis=0), np.arange(v.shape[1]), rng)
                dF = np.mean([f1[a, s] for s in SEEDS]) - np.mean([f1[b, s] for s in SEEDS])
                pairs.append(dict(pooling=label, backbone=backbone, dataset=ds, contrast=f"{a}-{b}", test_n=v.shape[1],
                                  delta_acc=v.mean(), ci_low=lo, ci_high=hi, seed_wins_acc=int((v.mean(axis=1) > 0).sum()),
                                  delta_macro_f1=dF,
                                  seed_wins_f1=int(sum(f1[a, s] > f1[b, s] for s in SEEDS))))
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def main():
    per_split = []
    for s in list(ROTATIONS) + ["T5"]:
        for backbone in BACKBONES:
            for ds in DATASETS:
                nclass = len(CLASSES[ds])
                for arm in ARMS:
                    for seed in SEEDS:
                        _, y, p = read(backbone, s, ds, seed, arm)
                        per_split.append(dict(split=s, backbone=backbone, dataset=ds, arm=arm, seed=seed, test_n=len(y),
                                              accuracy=100 * (y == p).mean(),
                                              macro_f1=100 * f1_score(y, p, labels=list(range(nclass)), average="macro", zero_division=0)))
    per_split = pd.DataFrame(per_split)
    per_split.to_csv(OUT / "per_split_runs.csv", index=False, float_format="%.6f")
    split_summary = per_split.groupby(["split", "backbone", "dataset", "arm"], sort=False).agg(
        test_n=("test_n", "first"), accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std")).reset_index()
    split_summary.to_csv(OUT / "per_split_summary.csv", index=False, float_format="%.6f")

    runs_r, pairs_r = pooled(list(ROTATIONS), "R1-R4")
    runs_a, pairs_a = pooled(list(ROTATIONS) + ["T5"], "all five splits")
    runs = pd.concat([runs_r, runs_a])
    table = runs.groupby(["pooling", "backbone", "dataset", "arm"], sort=False).agg(
        test_n=("test_n", "first"), accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std")).reset_index()
    table.to_csv(OUT / "pooled_summary.csv", index=False, float_format="%.6f")
    pairs = pd.concat([pairs_r, pairs_a])
    pairs.to_csv(OUT / "pooled_paired.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 220)
    for label in ("R1-R4", "all five splits"):
        print(f"\nPooled over {label}: test accuracy / macro-F1 (%, mean +- SD over 5 seeds)")
        for r in table[table.pooling == label].itertuples():
            print(f"  {r.backbone:9s} {r.dataset} {r.arm} n={r.test_n:4d}  {r.accuracy:6.2f} +- {r.accuracy_sd:4.2f}   {r.macro_f1:6.2f} +- {r.macro_f1_sd:4.2f}")
        print(f"Paired differences ({label}): delta accuracy [95% CI], delta macro-F1, seeds improved (acc, F1)")
        for r in pairs[pairs.pooling == label].itertuples():
            print(f"  {r.backbone:9s} {r.dataset} {r.contrast}  {r.delta_acc:+6.2f} [{r.ci_low:6.2f}, {r.ci_high:6.2f}]  "
                  f"{r.delta_macro_f1:+6.2f}   {r.seed_wins_acc}/5, {r.seed_wins_f1}/5")
    print("\nPer split, Exact quotient AG minus no AG (macro-F1 points):")
    s = split_summary.pivot_table(index=["split", "backbone", "dataset"], columns="arm", values="macro_f1")
    print((s["C"] - s["A"]).round(2).unstack("dataset").to_string())


if __name__ == "__main__":
    main()
