# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Extraction time by size (Table R6) and the chord, bridge and quotient-order statistics of Appendix C.

Reads the per-neuron sizes, counts and stage times stored in 3_knn_descriptors/features.npz and writes
  per_neuron.csv               one row per reconstruction (2,103 rows)
  tableR6_runtime_by_size.csv  stage medians, total median and total 90th percentile (ms) by core size
  geometry_summary.csv         per dataset, at eps = 20 um and tau = 5 um: share of neurons that gain a
                               cycle, that have no tip within eps of the soma, that have a contracted
                               bridge, and whose quotient has fewer than K vertices
  cycles_by_class.csv          training neurons that gain a cycle or have a contracted bridge, by class

    python 4_morphology_and_runtime/runtime_and_geometry.py [--features FILE] [--out DIR]

Results go to 4_morphology_and_runtime/rerun/ by default; the tables of the paper are in outputs/.
The times in features.npz were measured with 4 parallel SWCs and one BLAS thread each.
"""
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
from clearmind.paths import CLASSES
import knn

COUNTS = ["n_raw", "n_core", "m"] + [f"{name}{e}" for e in knn.EPS for name in ("chords", "overwrite")] \
    + [f"short{e}_{t}" for e in knn.EPS for t in knn.TAU]
STAGES = ["parse_s", "kron_s", "augmentation_s", "solve_s", "spectrum_s"]   # read, Kron, chords and bridges, solve, eigh


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(knn.FEATURES))
    parser.add_argument("--out", default=str(HERE / "rerun"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table, arrays = knn.load_features(args.features)
    meta = table[["dataset", "name", "split", "label"]].copy()
    for key in COUNTS[:3] + STAGES + COUNTS[3:]:
        meta[key] = arrays[key].astype(int) if key in COUNTS else arrays[key]
    meta.to_csv(out / "per_neuron.csv", index=False)

    # Table R6: each column is a separate median, so the total is not the sum of the stages.
    meta["size"] = pd.cut(meta.n_core, [0, 64, 128, 256, np.inf], labels=["<=64", "65-128", "129-256", ">256"])
    meta["total_s"] = meta[STAGES].sum(axis=1)
    rows = []
    for size, grp in meta.groupby("size", observed=True):
        median = grp[STAGES + ["total_s"]].quantile(.5)
        rows.append(dict(core=size, neurons=len(grp), read_ms=1000 * median.parse_s, kron_ms=1000 * median.kron_s,
                         augmentation_ms=1000 * median.augmentation_s, solve_ms=1000 * median.solve_s,
                         spectrum_ms=1000 * median.spectrum_s, total_ms=1000 * median.total_s,
                         total_p90_ms=1000 * grp.total_s.quantile(.9),
                         solve_plus_spectrum_ms=1000 * (grp.solve_s + grp.spectrum_s).median()))
    runtime = pd.DataFrame(rows)
    runtime.round(4).to_csv(out / "tableR6_runtime_by_size.csv", index=False)

    rows = []
    for ds, grp in meta.groupby("dataset", sort=False):
        rows.append(dict(dataset=ds, neurons=len(grp),
                         new_cycle_pct=100 * (grp.chords20 > 0).mean(),
                         length_update_pct=100 * (grp.overwrite20 > 0).mean(),
                         no_tip_within_eps_pct=100 * ((grp.chords20 + grp.overwrite20) == 0).mean(),
                         contracted_bridge_pct=100 * (grp.short20_5 > 0).mean(),
                         **{f"order_below_{k}_pct": 100 * (grp.m < k).mean() for k in (16, 32, 64, 128)}))
    geometry = pd.DataFrame(rows)
    geometry.round(2).to_csv(out / "geometry_summary.csv", index=False)

    rows = []
    train = meta[meta.split == "train"]
    for ds in CLASSES:
        for label, cell_class in enumerate(CLASSES[ds]):
            grp = train[(train.dataset == ds) & (train.label == label)]
            rows.append(dict(dataset=ds, cell_class=cell_class, neurons=len(grp),
                             with_new_cycle=int((grp.chords20 > 0).sum()),
                             with_contracted_bridge=int((grp.short20_5 > 0).sum()),
                             mean_new_cycles=grp.chords20.mean(), mean_contracted_bridges=grp.short20_5.mean()))
    pd.DataFrame(rows).to_csv(out / "cycles_by_class.csv", index=False)

    pd.set_option("display.width", 200)
    print(runtime.round(1).to_string(index=False))
    print(geometry.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
