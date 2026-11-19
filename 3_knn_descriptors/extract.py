# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Compute per-neuron descriptors from the SWC files listed in data/splits.csv.

    python 3_knn_descriptors/extract.py features      # AG spectra, Laplacian, Sholl, TMD, statistics, stage timings
    python 3_knn_descriptors/extract.py persistence   # full-tree radial H0 histogram (Table R3)
    python 3_knn_descriptors/extract.py lattice       # lattice pipeline and Exact quotient AG at eps=20, tau=5 (Table 3)

One .npz per neuron is written to cache/<kind>/<dataset>/; existing files are kept, so an interrupted
run resumes. Each process uses one BLAS thread. The paper's timings used 4 parallel SWCs for
`features` (Table R6) and 10 for `lattice` (Table 3), the defaults here.
Options: --workers N, --per-dataset N (only the first N neurons of each dataset), --cache DIR.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np
from clearmind.paths import splits, swc_path
import descriptors

KINDS = {"features": ("neuron_features", 4, 2), "persistence": ("persistence_histogram", 4, 4),
         "lattice": ("lattice_features", 10, 1)}   # function, default workers, chunk size


def compute(job):
    function, dataset, name, cache = job
    dest = Path(cache) / dataset / (name + ".npz")
    if dest.exists():
        return
    output = getattr(descriptors, function)(swc_path(dataset, name))
    tmp = dest.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **output)
    os.replace(tmp, dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=list(KINDS))
    parser.add_argument("--cache", default=None, help="output folder (default: 3_knn_descriptors/cache/<kind>)")
    parser.add_argument("--per-dataset", type=int, default=None, help="only the first N neurons of each dataset")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    function, workers, chunk = KINDS[args.kind]
    cache = Path(args.cache) if args.cache else HERE / "cache" / args.kind
    table = splits()
    if args.per_dataset:
        table = table.groupby("dataset", sort=False).head(args.per_dataset)
    for dataset in table.dataset.unique():
        (cache / dataset).mkdir(parents=True, exist_ok=True)
    jobs = [(function, d, n, str(cache)) for d, n in zip(table.dataset, table.name)]
    with ProcessPoolExecutor(max_workers=args.workers or workers, initializer=descriptors.load_submission_code) as pool:
        for i, _ in enumerate(pool.map(compute, jobs, chunksize=chunk), 1):
            if i % 200 == 0:
                print(args.kind, i, "/", len(jobs), flush=True)
    print(args.kind, "done:", len(jobs), "neurons in", cache, flush=True)


if __name__ == "__main__":
    main()
