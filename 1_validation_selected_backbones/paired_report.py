# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Tables 5, R1 and R2 from the saved fold-7 selections and test predictions (no GPU, a few seconds).

1. Checks the selection rule for all 90 runs: the chosen epoch is the first epoch with the highest
   fold-7 accuracy, and training stopped 25 epochs later (or at epoch 150).
2. Test accuracy and macro-F1 per run, and 5-seed means and SDs per setting (Table 5, Table R1).
3. Paired differences between settings on the same neurons with 95% paired neuron-bootstrap intervals
   (Table R2): 10,000 draws over test neurons, each draw keeping all five seed predictions of a neuron.
   The draws use numpy generators seeded with 20261106: one for Tree-LSTM, one for the MorphoGNN contrast B-A,
   and one for the MorphoGNN contrasts C-A and C-B.
Usage (from the package root):  python 1_validation_selected_backbones/paired_report.py [--tree-dir D] [--gnn-dir D]
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from clearmind.paths import CLASSES, SEEDS, splits

DATASETS = list(CLASSES)
ARMS = ["A", "B", "C"]
NAMES = {"A": "no AG", "B": "Submitted AG", "C": "Exact quotient AG"}


def interval(diff, groups, rng):
    """95% bootstrap interval of the mean paired difference, resampling groups (neurons or donors)."""
    code = pd.factorize(groups)[0]
    sizes = np.bincount(code)
    sums = np.bincount(code, weights=diff)
    draw = rng.integers(len(sizes), size=(10000, len(sizes)))
    means = sums[draw].sum(axis=1) / sizes[draw].sum(axis=1)
    return np.percentile(means, [2.5, 97.5]).tolist()


def check_selection(folder):
    sel = json.loads((folder / "selection.json").read_text())
    for name, rec in sel.items():
        log = pd.read_csv(folder / "epoch_logs" / (name + ".csv"))
        assert int(log.loc[log.val_correct.idxmax(), "epoch"]) == rec["best_epoch"], name
        assert log.val_correct.max() == rec["best_val_correct"], name
        assert rec["stop_epoch"] == len(log) == min(150, rec["best_epoch"] + 25), name
    return sel


def hits_and_scores(backbone, folder, ds, test):
    """Per-run correctness vectors on the test neurons (in split order) and per-run metrics."""
    nclass = len(CLASSES[ds])
    hits, rows = {}, []
    for seed in SEEDS:
        for arm in ARMS:
            pred = pd.read_csv(folder / "test_predictions" / f"{ds}_{seed}_{arm}.csv")
            y = pred["y"] if "y" in pred else pred["label"]
            assert pred.name.tolist() == test.name.tolist() and np.array_equal(y, test.label)
            hits[seed, arm] = (y == pred.pred).to_numpy(dtype=float) * 100
            rows.append(dict(backbone=backbone, dataset=ds, arm=arm, seed=seed, test_n=len(test),
                             accuracy=hits[seed, arm].mean(),
                             macro_f1=100 * f1_score(y, pred.pred, labels=list(range(nclass)), average="macro", zero_division=0)))
    return hits, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree-dir", default=str(HERE / "treelstm" / "outputs"))
    ap.add_argument("--gnn-dir", default=str(HERE / "morphognn" / "outputs"))
    ap.add_argument("--out", default=str(HERE / "rerun"), help="output folder (default: rerun/; the tables used in the paper are in results/)")
    args = ap.parse_args()
    tree, gnn, out = Path(args.tree_dir), Path(args.gnn_dir), Path(args.out)
    out.mkdir(exist_ok=True)
    for folder in (tree, gnn):
        n = len(check_selection(folder))
        print(f"selection rule holds for all {n} runs in {folder.parent.name}")

    df = splits()
    runs, pairs = [], []
    f1 = {}

    # Tree-LSTM: contrasts B-A, C-A, C-B; neuron-level and (where donor ids exist) donor-level intervals.
    rng = np.random.default_rng(20261106)
    for ds in DATASETS:
        test = df[(df.dataset == ds) & (df.split == "test")]
        hits, rows = hits_and_scores("Tree-LSTM", tree, ds, test)
        runs += rows
        for a, b in (("B", "A"), ("C", "A"), ("C", "B")):
            v = np.array([hits[s, a] - hits[s, b] for s in SEEDS])
            diff = v.mean(axis=0)
            lo, hi = interval(diff, np.arange(len(test)), rng)
            donor = [np.nan, np.nan]
            if test.donor_id.notna().all():
                donor = interval(diff, test.donor_id.to_numpy(), rng)
            pairs.append(dict(backbone="Tree-LSTM", dataset=ds, contrast=f"{a}-{b}", delta_acc=diff.mean(),
                              ci_low=lo, ci_high=hi, seed_wins=int((v.mean(axis=1) > 0).sum()),
                              donor_ci_low=donor[0], donor_ci_high=donor[1]))

    # MorphoGNN: B-A first (its own generator), then C-A and C-B (a second generator with the same seed).
    gnn_hits = {}
    for ds in DATASETS:
        test = df[(df.dataset == ds) & (df.split == "test")]
        gnn_hits[ds], rows = hits_and_scores("MorphoGNN", gnn, ds, test)
        runs += rows
    for contrasts in ((("B", "A"),), (("C", "A"), ("C", "B"))):
        rng = np.random.default_rng(20261106)
        for ds in DATASETS:
            n = int((df[(df.dataset == ds) & (df.split == "test")]).shape[0])
            for a, b in contrasts:
                v = np.array([gnn_hits[ds][s, a] - gnn_hits[ds][s, b] for s in SEEDS])
                lo, hi = interval(v.mean(axis=0), np.arange(n), rng)
                pairs.append(dict(backbone="MorphoGNN", dataset=ds, contrast=f"{a}-{b}", delta_acc=v.mean(),
                                  ci_low=lo, ci_high=hi, seed_wins=int((v.mean(axis=1) > 0).sum()),
                                  donor_ci_low=np.nan, donor_ci_high=np.nan))

    per_run = pd.DataFrame(runs)
    table = per_run.groupby(["backbone", "dataset", "arm"], sort=False).agg(
        test_n=("test_n", "first"), accuracy=("accuracy", "mean"), accuracy_sd=("accuracy", "std"),
        macro_f1=("macro_f1", "mean"), macro_f1_sd=("macro_f1", "std")).reset_index()
    mean_f1 = table.set_index(["backbone", "dataset", "arm"]).macro_f1
    paired = pd.DataFrame(pairs)
    paired["delta_macro_f1"] = [mean_f1[bb, ds, c[0]] - mean_f1[bb, ds, c[2]]
                                for bb, ds, c in paired[["backbone", "dataset", "contrast"]].itertuples(index=False, name=None)]
    per_run.to_csv(out / "per_seed.csv", index=False, float_format="%.6f")
    table.to_csv(out / "table5_R1.csv", index=False, float_format="%.6f")
    paired.to_csv(out / "paired_R2.csv", index=False, float_format="%.6f")

    pd.set_option("display.width", 200)
    show = table.assign(arm=table.arm.map(NAMES))
    print("\nTest accuracy / macro-F1 (%, mean +- SD over 5 seeds)")
    for r in show.itertuples():
        print(f"  {r.backbone:9s} {r.dataset}  {r.arm:18s} {r.accuracy:6.2f} +- {r.accuracy_sd:4.2f}   {r.macro_f1:6.2f} +- {r.macro_f1_sd:4.2f}")
    print("\nPaired differences: delta accuracy [95% neuron-bootstrap CI] / delta macro-F1 (percentage points)")
    for r in paired.itertuples():
        print(f"  {r.backbone:9s} {r.dataset}  {r.contrast}  {r.delta_acc:+6.2f} [{r.ci_low:6.2f}, {r.ci_high:6.2f}]  "
              f"{r.delta_macro_f1:+6.2f}   seeds improved {r.seed_wins}/5")
    print("\nwritten:", ", ".join(p.name for p in sorted(out.glob("*.csv"))))


if __name__ == "__main__":
    main()
