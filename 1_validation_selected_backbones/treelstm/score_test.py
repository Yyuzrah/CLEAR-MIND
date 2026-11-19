# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Score folds 8-9 once with the checkpoints chosen on fold 7.

Needs <run-dir>/selection.json with all 45 runs, which train_select.py writes at the end of training.
Usage (from the package root):
  python 1_validation_selected_backbones/treelstm/score_test.py                 # shipped checkpoints, writes rerun/test_predictions
  python 1_validation_selected_backbones/treelstm/score_test.py --run-dir <dir> # your own rerun
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from common import ARMS, DATASETS, DEVICE, OUTPUTS, SEEDS, PairModel, predict, read_set, tag
from clearmind.paths import splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(OUTPUTS))
    ap.add_argument("--pred-dir", default=None, help="where to write predictions (default: <run-dir>/test_predictions, or rerun/test_predictions for the shipped run)")
    args = ap.parse_args()
    run = Path(args.run_dir)
    selection = json.loads((run / "selection.json").read_text())
    runs = [tag(d, s, a) for d in DATASETS for s in SEEDS for a in ARMS]
    missing = [r for r in runs if r not in selection]
    if missing:
        raise SystemExit(f"selection.json is incomplete ({len(missing)} runs missing); finish train_select.py first.")
    default = OUTPUTS.parent / "rerun" / "test_predictions" if run.resolve() == OUTPUTS.resolve() else run / "test_predictions"
    pred_dir = Path(args.pred_dir) if args.pred_dir else default
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = splits()
    for d in DATASETS:
        test = read_set(d, df[(df.dataset == d) & (df.split == "test")])
        for seed in SEEDS:
            for arm in ARMS:
                name = tag(d, seed, arm)
                model = PairModel(d, arm).to(DEVICE)
                model.load_state_dict(torch.load(run / "checkpoints" / (name + ".pt"), map_location=DEVICE))
                yhat = predict(model, test, arm)
                pd.DataFrame(dict(dataset=d, seed=seed, arm=arm, name=[r["name"] for r in test],
                                  fold=[r["fold"] for r in test], y=[r["label"] for r in test], pred=yhat)
                             ).to_csv(pred_dir / (name + ".csv"), index=False)
        print("scored", d, flush=True)


if __name__ == "__main__":
    main()
