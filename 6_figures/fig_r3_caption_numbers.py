# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Numbers quoted in the caption of Fig. R3, from 3_knn_descriptors/features.npz and outputs/test_coordinates.csv.

  python 6_figures/fig_r3_caption_numbers.py      # a few seconds

1. The lambda_1 entry of each unit-normalized Exact quotient AG spectrum (64 magnitudes), over all 2,103 neurons,
   against the same entry for the highest weighted Laplacian modes.
2. Per dataset, the Spearman correlation between PC1 of Fig. R3 (test neurons) and the second signature entry
   divided by the first (|lambda_N| / lambda_1).
3. BIL-6 test neurons: AUC of VPM against L4 and against L6 along PC1 (VPM to the right).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from clearmind.paths import CLASSES
import knn


def main():
    table, arrays = knn.load_features(str(knn.FEATURES))
    ag = knn.unit(np.asarray(arrays[knn.DEFAULT], float)[:, :64])[:, 0]
    lap = knn.unit(np.asarray(arrays["lap_high"], float)[:, :64])[:, 0]
    print("1. lambda_1 entry of the unit-normalized spectrum, all %d neurons:" % len(ag))
    print("   Exact quotient AG: minimum %.5f, median %.5f (lower bound 1/sqrt(2) = %.5f)" % (ag.min(), np.median(ag), 2 ** -0.5))
    print("   highest Laplacian modes: median %.5f" % np.median(lap))
    coords = pd.read_csv(HERE / "outputs" / "test_coordinates.csv")
    print("2. Spearman correlation of PC1 (Fig. R3, test neurons) with |lambda_N| / lambda_1:")
    for ds in CLASSES:
        sub, data = knn.dataset_view(table, arrays, ds)
        te = (sub.split == "test").to_numpy()
        spec = np.asarray(data[knn.DEFAULT], float)[te]
        c = coords[(coords.dataset == ds) & (coords.descriptor == "Exact quotient AG")].set_index("name")
        pc1 = c.loc[sub.name[te].values, "PC1"].to_numpy()
        print("   %s: %.4f" % (ds, spearmanr(pc1, spec[:, 1] / spec[:, 0]).correlation))
        if ds == "BIL-6":
            names = [s.replace("Isocortex_layer", "L").replace("L23", "L2/3") for s in CLASSES[ds]]
            y = sub.label.to_numpy()[te]
            vpm = names.index("VPM")
            for other in ("L4", "L6"):
                k = names.index(other)
                sel = (y == vpm) | (y == k)
                auc = roc_auc_score((y[sel] == vpm).astype(int), pc1[sel])
                print("3. BIL-6 test neurons, VPM against %s along PC1: AUC %.4f (VPM %d, %s %d)" % (other, auc, (y == vpm).sum(), other, (y == k).sum()))


if __name__ == "__main__":
    main()
