# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Train the 45 MorphoGNN runs (3 datasets x 5 seeds x settings A/B/C) and select checkpoints on fold 7 only.

Same rule as for Tree-LSTM: folds 0-6 train, the earliest epoch with the highest fold-7 accuracy is kept
(patience 25, at most 150 epochs), and folds 8-9 are not read here. The settings of a seed share the seed,
initial weights, point clouds, jitter, minibatch order, optimizer and schedule; only the spectral input
differs. When all runs are done, the chosen epochs are written to <out>/selection.json.
Usage (from the package root):  python 1_validation_selected_backbones/morphognn/train_select.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import time
from pathlib import Path

import torch
from common import ARMS, BATCH, DATASETS, DEVICE, MAX_EPOCH, OUTPUTS, PATIENCE, SEEDS, build, load_set, predict, tag
from clearmind.paths import splits


def fit(dataset, seed, arm, train, valid, out, max_epoch=MAX_EPOCH):
    name = tag(dataset, seed, arm)
    record = out / "runs" / (name + ".json")
    if record.exists():
        return json.loads(record.read_text())
    weights = out / "checkpoints" / (name + ".pt")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = build(dataset, arm)
    optim = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=150, eta_min=1e-6)
    order = torch.Generator().manual_seed(seed)
    x, s, y = train
    hist, best, chosen = [], -1, 0
    for epoch in range(1, max_epoch + 1):
        net.train()
        for ix in torch.randperm(len(y), generator=order).split(BATCH):
            if len(ix) == 1:  # BatchNorm needs at least two samples per mini-batch
                continue
            ix = ix.to(DEVICE)
            optim.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(x[ix], s[ix]), y[ix])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
            optim.step()
        scheduler.step()
        score = int((predict(net, valid) == valid[2]).sum())
        hist.append(score)
        if score > best:
            best, chosen = score, epoch
            torch.save(net.state_dict(), weights)
        if epoch == 1 or epoch % 10 == 0:
            print(name, "epoch", epoch, "best fold-7 correct", best, flush=True)
        if epoch - chosen >= PATIENCE:
            break
    import pandas as pd
    pd.DataFrame(dict(epoch=range(1, len(hist) + 1), val_correct=hist)).to_csv(out / "epoch_logs" / (name + ".csv"), index=False)
    rec = dict(best_epoch=chosen, stop_epoch=epoch, best_val_correct=best, val_n=len(valid[2]))
    record.write_text(json.dumps(rec, indent=1))
    print("done", name, "best epoch", chosen, "stopped", epoch, flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS.parent / "rerun"),
                    help="output folder (default: rerun/; the shipped results are in outputs/)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH, help="for a quick check only")
    args = ap.parse_args()
    out = Path(args.out)
    for sub in ("runs", "epoch_logs", "checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df = splits()
    for d in args.datasets:
        sub = df[df.dataset == d]
        for seed in args.seeds:
            for arm in args.arms:
                fit(d, seed, arm, load_set(d, sub[sub.split == "train"], seed, arm),
                    load_set(d, sub[sub.split == "valid"], seed, arm), out, args.max_epoch)
    runs = [tag(d, s, a) for d in DATASETS for s in SEEDS for a in ARMS]
    if all((out / "runs" / (r + ".json")).exists() for r in runs):
        selection = {r: json.loads((out / "runs" / (r + ".json")).read_text()) for r in runs}
        (out / "selection.json").write_text(json.dumps(selection, indent=1))
        print("fold-7 selection written for", len(selection), "runs", flush=True)


if __name__ == "__main__":
    main()
