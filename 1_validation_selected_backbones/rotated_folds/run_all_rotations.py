# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Run all 24 rotation jobs (4 rotations x 3 datasets x 2 backbones), several at a time on one GPU.

Each job is train_rotation.py for one backbone, rotation and dataset. Finished runs are skipped when a
job is restarted. Logs go to outputs/logs/.
Usage (from the package root):  python 1_validation_selected_backbones/rotated_folds/run_all_rotations.py [--jobs 8]
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def run(job):
    backbone, rotation, dataset = job
    log = HERE / "outputs" / "logs" / f"{rotation}_{backbone}_{dataset}.log"
    start = time.monotonic()
    with log.open("w") as fh:
        code = subprocess.call([sys.executable, str(HERE / "train_rotation.py"), "--backbone", backbone,
                                "--rotation", rotation, "--dataset", dataset], stdout=fh, stderr=subprocess.STDOUT, cwd=ROOT)
    print(f"{rotation} {backbone} {dataset}: exit {code}, {round((time.monotonic() - start) / 60, 1)} min", flush=True)
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()
    (HERE / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    # Longest jobs first: BIL-6 is the largest dataset, Tree-LSTM the slower backbone there.
    jobs = [(b, r, d) for d in ("BIL-6", "JML-4", "ACT-4") for b in ("treelstm", "morphognn") for r in ("R1", "R2", "R3", "R4")]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        codes = list(pool.map(run, jobs))
    print("all jobs finished; failures:", sum(c != 0 for c in codes), flush=True)


if __name__ == "__main__":
    main()
