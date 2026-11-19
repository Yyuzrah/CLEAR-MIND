# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Collect the per-neuron caches of extract.py into one features file (rows in data/splits.csv order).

Each descriptor is truncated or zero-padded to the width the k-NN rows use (128 for the default Exact
quotient AG spectrum because of Table R5, 5 for the statistics, 64 otherwise). The file also holds the
per-neuron sizes, chord and bridge counts and the stage times behind Table R6, and the feature times of
Table 3 (lattice_s, exact_s). All times are in seconds.

    python 3_knn_descriptors/pack_features.py [--cache DIR] [--out FILE]

The default output is rerun/features.npz; the file used in the paper is features.npz in this folder.
"""
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np
from clearmind.paths import splits
import knn

DESCRIPTORS = [key for key in knn.WIDTHS if key not in ("full_tree_h0", "lattice")]
COUNTS = ["n_raw", "n_core", "m"] + [f"{name}{e}" for e in knn.EPS for name in ("chords", "overwrite")] \
    + [f"short{e}_{t}" for e in knn.EPS for t in knn.TAU]
STAGE_TIMES = ["parse_s", "kron_s", "augmentation_s", "solve_s", "spectrum_s"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(HERE / "cache"), help="folder with features/, persistence/ and lattice/")
    parser.add_argument("--out", default=str(HERE / "rerun" / "features.npz"))
    args = parser.parse_args()
    cache = Path(args.cache)
    table = splits()
    n = len(table)
    arrays = {key: np.zeros((n, width)) for key, width in knn.WIDTHS.items()}
    counts = {key: np.zeros(n, dtype=np.int64) for key in COUNTS}
    times = {key: np.zeros(n) for key in STAGE_TIMES + ["lattice_s", "exact_s"]}
    exact = np.zeros((n, 64))
    for i, (dataset, name) in enumerate(zip(table.dataset, table.name)):
        with np.load(cache / "features" / dataset / (name + ".npz")) as file:
            for key in DESCRIPTORS:
                val = file[key][:knn.WIDTHS[key]]
                arrays[key][i, :len(val)] = val
            for key in COUNTS:
                counts[key][i] = file[key]
            for key in STAGE_TIMES:
                times[key][i] = file[key]
        with np.load(cache / "persistence" / dataset / (name + ".npz")) as file:
            arrays["full_tree_h0"][i] = file["hist"]
        with np.load(cache / "lattice" / dataset / (name + ".npz")) as file:
            arrays["lattice"][i] = file["lattice"]
            exact[i] = file["exact"]
            times["lattice_s"][i] = file["lattice_s"]
            times["exact_s"][i] = file["exact_s"]
    for key, value in arrays.items():
        assert np.isfinite(value).all(), key
    # The Exact quotient AG spectrum timed next to the lattice pipeline is the default one used everywhere else.
    for dataset in table.dataset.unique():
        rows = (table.dataset == dataset).to_numpy()
        ref = arrays[knn.DEFAULT][rows, :64]
        assert np.allclose(exact[rows], ref, rtol=1e-9, atol=1e-9 * exact[rows].max()), dataset
    arrays["submitted_ag"] = arrays["submitted_ag"].astype(np.float32)   # single precision, as in the model inputs
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, dataset=table.dataset.to_numpy(dtype=str), name=table.name.to_numpy(dtype=str),
                        **arrays, **counts, **times)
    print("wrote", out, "with", n, "neurons")


if __name__ == "__main__":
    main()
