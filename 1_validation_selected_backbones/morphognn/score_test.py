# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Score folds 8-9 once with the MorphoGNN checkpoints chosen on fold 7.

Needs <run-dir>/selection.json with all 45 runs, which train_select.py writes at the end of training. The shipped predictions are in
outputs/test_predictions; the MorphoGNN checkpoints (about 290 MB) are not shipped, so run
train_select.py first and pass its folder with --run-dir.
Usage (from the package root):  python 1_validation_selected_backbones/morphognn/score_test.py --run-dir <dir>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
from pathlib import Path

import torch
from common import ARMS, DATASETS, DEVICE, SEEDS, build, load_set, predict, tag
from clearmind.paths import splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run = Path(args.run_dir)
    selection = json.loads((run / "selection.json").read_text())
    runs = [tag(d, s, a) for d in DATASETS for s in SEEDS for a in ARMS]
    if any(r not in selection for r in runs):
        raise SystemExit("selection.json is incomplete; finish train_select.py first.")
    pred_dir = run / "test_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    df = splits()
    for d in DATASETS:
        test = df[(df.dataset == d) & (df.split == "test")]
        for seed in SEEDS:
            for arm in ARMS:
                name = tag(d, seed, arm)
                net = build(d, arm)
                net.load_state_dict(torch.load(run / "checkpoints" / (name + ".pt"), map_location=DEVICE))
                yhat = predict(net, load_set(d, test, seed, arm)).cpu().numpy()
                test.assign(pred=yhat)[["name", "fold", "label", "pred"]].to_csv(pred_dir / (name + ".csv"), index=False)
        print("scored", d, flush=True)


if __name__ == "__main__":
    main()
