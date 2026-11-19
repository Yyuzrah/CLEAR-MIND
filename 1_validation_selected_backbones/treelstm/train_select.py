# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Train the 45 Tree-LSTM runs (3 datasets x 5 seeds x settings A/B/C) and select checkpoints on fold 7 only.

Folds 0-6 train. After every epoch the model is scored on fold 7; the earliest epoch with the highest
fold-7 accuracy is kept (patience 25, at most 150 epochs). The three settings of a seed share the seed,
initial weights, minibatch order, optimizer and schedule. Folds 8-9 are not read here.
When all runs are done, the chosen epochs are written to <out>/selection.json; score_test.py
only runs once that file exists. The results used in the paper are in outputs/.
Usage (from the package root):  python 1_validation_selected_backbones/treelstm/train_select.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from common import (ARMS, BATCH, DATASETS, DEVICE, MAX_EPOCH, OUTPUTS, PATIENCE, SEEDS, PairModel,
                    make_batch, predict, read_set, tag)
from clearmind.paths import splits


def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit(dataset, seed, arm, train, valid, out, max_epoch=MAX_EPOCH):
    name = tag(dataset, seed, arm)
    record = out / "runs" / (name + ".json")
    if record.exists():
        return json.loads(record.read_text())
    weights = out / "checkpoints" / (name + ".pt")
    init_seed(seed)
    model = PairModel(dataset, arm).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=150, eta_min=1e-6)
    best, best_ep, history = -1, 0, []
    val_y = np.array([r["label"] for r in valid])
    sampler = torch.Generator().manual_seed(seed)
    start = time.monotonic()
    for ep in range(1, max_epoch + 1):
        order = torch.randperm(len(train), generator=sampler).tolist()
        model.train()
        loss_sum = 0.0
        for i in range(0, len(order), BATCH):
            b = make_batch(train, order[i:i + BATCH], arm)
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(b), b.labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach()) * len(b.labels)
        sched.step()
        correct = int((predict(model, valid, arm) == val_y).sum())
        if correct > best:
            best, best_ep = correct, ep
            torch.save(model.state_dict(), weights)
        history.append(dict(epoch=ep, train_loss=loss_sum / len(train), val_correct=correct))
        if ep in (1, 10) or ep % 25 == 0:
            print(name, "epoch", ep, "best fold-7 correct", best, flush=True)
        if ep - best_ep >= PATIENCE:
            break
    pd.DataFrame(history).to_csv(out / "epoch_logs" / (name + ".csv"), index=False)
    rec = dict(best_epoch=best_ep, stop_epoch=ep, best_val_correct=best, val_n=len(valid))
    record.write_text(json.dumps(rec, indent=1))
    print("done", name, "best epoch", best_ep, "stopped", ep, flush=True)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUTPUTS.parent / "rerun"), help="output folder (default: rerun/; the shipped results are in outputs/)")
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--arms", nargs="*", default=ARMS)
    ap.add_argument("--max-epoch", type=int, default=MAX_EPOCH, help="for a quick check only")
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.out)
    for sub in ("runs", "epoch_logs", "checkpoints"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    df = splits()
    print("device:", DEVICE, flush=True)
    for d in args.datasets:
        tr = read_set(d, df[(df.dataset == d) & (df.split == "train")])
        va = read_set(d, df[(df.dataset == d) & (df.split == "valid")])
        for seed in args.seeds:
            for arm in args.arms:
                fit(d, seed, arm, tr, va, out, args.max_epoch)
    # Write the selection only when every run of the full design has finished.
    runs = [tag(d, s, a) for d in DATASETS for s in SEEDS for a in ARMS]
    if all((out / "runs" / (r + ".json")).exists() for r in runs):
        selection = {r: json.loads((out / "runs" / (r + ".json")).read_text()) for r in runs}
        (out / "selection.json").write_text(json.dumps(selection, indent=1))
        print("fold-7 selection written for", len(selection), "runs", flush=True)


if __name__ == "__main__":
    main()
