# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Train, select and test one backbone on one dataset for one rotation (15 runs: 5 seeds x settings A/B/C).

The training code is the one of Table 5 (treelstm/train_select.py or morphognn/train_select.py, function
fit); only the fold assignment changes (see rotations.py). All 15 checkpoints are chosen on the
rotation's validation fold and written to selection.json before the test folds are scored, once.
Usage (from the package root):
  python 1_validation_selected_backbones/rotated_folds/train_rotation.py --backbone treelstm --rotation R1 --dataset ACT-4
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--backbone", choices=["treelstm", "morphognn"], required=True)
ap.add_argument("--rotation", required=True)
ap.add_argument("--dataset", required=True)
ap.add_argument("--out", default=str(HERE / "outputs"))
ap.add_argument("--max-epoch", type=int, default=150, help="for a quick check only")
args = ap.parse_args()

sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / args.backbone))

import numpy as np
import pandas as pd
import torch
from clearmind.paths import splits
from rotations import ROTATIONS, with_rotation
import common
from train_select import fit

ARMS, SEEDS, DEVICE = common.ARMS, common.SEEDS, common.DEVICE


def tree_sets(df):
    return {s: common.read_set(args.dataset, df[df.split == s]) for s in ("train", "valid", "test")}


def gnn_load(sub, seed, arm):
    """Point clouds of this rotation, the spectrum of the setting, labels (as morphognn/common.load_set)."""
    clouds, spec = [], []
    for name in sub.name:
        with np.load(HERE / "cache" / args.rotation / args.dataset / (name + ".npz")) as f:
            clouds.append(f[str(seed)])
        if arm in "AB":
            with np.load(common.CACHE / args.dataset / (name + ".npz")) as g:
                spec.append(g["spec"])
        else:
            with np.load(common.EXACT / args.dataset / (name + ".npz")) as g:
                spec.append(g["C"])
    return tuple(torch.as_tensor(x, device=DEVICE) for x in (np.stack(clouds), np.stack(spec), sub.label.to_numpy()))


def main():
    if args.rotation not in ROTATIONS:
        raise SystemExit(f"unknown rotation {args.rotation}")
    out = Path(args.out) / args.rotation / args.backbone / args.dataset
    for sub in ("runs", "epoch_logs", "checkpoints", "test_predictions"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df = with_rotation(splits(), args.rotation)
    df = df[df.dataset == args.dataset]
    print(args.backbone, args.rotation, args.dataset, "train/valid/test =",
          *(int((df.split == s).sum()) for s in ("train", "valid", "test")), "device", DEVICE, flush=True)

    # 1. Train on the training folds, select on the validation fold.
    if args.backbone == "treelstm":
        sets = tree_sets(df)
        for seed in SEEDS:
            for arm in ARMS:
                fit(args.dataset, seed, arm, sets["train"], sets["valid"], out, args.max_epoch)
    else:
        for seed in SEEDS:
            for arm in ARMS:
                fit(args.dataset, seed, arm, gnn_load(df[df.split == "train"], seed, arm),
                    gnn_load(df[df.split == "valid"], seed, arm), out, args.max_epoch)

    # 2. Fix all 15 checkpoints before any test prediction.
    names = [common.tag(args.dataset, s, a) for s in SEEDS for a in ARMS]
    selection = {n: json.loads((out / "runs" / (n + ".json")).read_text()) for n in names}
    (out / "selection.json").write_text(json.dumps(selection, indent=1))

    # 3. Score the test folds once.
    test = df[df.split == "test"]
    if args.backbone == "treelstm":
        rows = common.read_set(args.dataset, test)
    for seed in SEEDS:
        for arm in ARMS:
            name = common.tag(args.dataset, seed, arm)
            if args.backbone == "treelstm":
                model = common.PairModel(args.dataset, arm).to(DEVICE)
                model.load_state_dict(torch.load(out / "checkpoints" / (name + ".pt"), map_location=DEVICE))
                yhat = common.predict(model, rows, arm)
            else:
                net = common.build(args.dataset, arm)
                net.load_state_dict(torch.load(out / "checkpoints" / (name + ".pt"), map_location=DEVICE))
                yhat = common.predict(net, gnn_load(test, seed, arm)).cpu().numpy()
            pd.DataFrame(dict(name=test.name.to_numpy(), fold=test.fold.to_numpy(), label=test.label.to_numpy(),
                              pred=yhat)).to_csv(out / "test_predictions" / (name + ".csv"), index=False)
    print("finished", args.backbone, args.rotation, args.dataset, flush=True)


if __name__ == "__main__":
    main()
