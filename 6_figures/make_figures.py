# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Figures R2-R4 on the test neurons (folds 8-9), from 3_knn_descriptors/features.npz.

  spectrum_by_class              Fig. R2: class-wise median spectra with interquartile bands, for Exact
                                 quotient AG (um) and the highest weighted Laplacian modes (1/um)
  test_pca                       Fig. R3: PCA fitted on unit-normalized training spectra (folds 0-6),
                                 test neurons projected, same axis limits within a row
  leading_eigenvalue_histograms  Fig. R4: leading AG eigenvalue per class, shared logarithmic bins
Each figure is saved as .pdf and .png in the output folder. The plotted numbers are saved next to them:
  test_coordinates.csv     the PCA coordinates of every test neuron
  spectrum_by_class.csv    quartiles of every eigenvalue rank per class (the curves and bands of Fig. R2)
  leading_eigenvalue.csv   per class: test neurons with the leading eigenvalue below 0.5, in 0.5-1, in
                           1-2 and above 2 (x 10^6 um), and the quartiles of the rank-16 eigenvalue

    python 6_figures/make_figures.py [--features FILE] [--out DIR]

Results go to 6_figures/rerun/ by default; the figures of the paper are in outputs/.
"""
import sys
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from clearmind.paths import CLASSES
import knn

COLORS = ["#17659b", "#c84e22", "#00816d", "#7d4e99", "#b07800", "#444444"]
SYMBOLS = ["o", "s", "^", "D", "v", "P"]
STYLE = {label: (COLORS[n], SYMBOLS[n]) for n, label in enumerate(["L2/3", "L4", "L5", "L6", "VPM", "CP"])}
# feature key, panel title as in the paper figures, unit, name in the CSV files
KEYS = [(knn.DEFAULT, "Exact AG", "µm", "Exact quotient AG"), ("lap_high", "Weighted Laplacian", "µm⁻¹", "Laplacian high")]
LEADING_BINS = [0, .5e6, 1e6, 2e6, np.inf]   # um


def save(fig, path):
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=.08, metadata={"CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=165, bbox_inches="tight", pad_inches=.08)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(knn.FEATURES))
    parser.add_argument("--out", default=str(HERE / "rerun"))
    args = parser.parse_args()
    folder = Path(args.out)
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    spec, axes = plt.subplots(3, 2, figsize=(8.2, 8.8), constrained_layout=True)
    scatter, pcaxes = plt.subplots(3, 2, figsize=(8.2, 8.8), constrained_layout=True)
    hist, haxes = plt.subplots(3, 1, figsize=(6.2, 7.4), constrained_layout=True)
    table, arrays = knn.load_features(args.features)
    coords, bands, summary = [], [], []
    for i, ds in enumerate(CLASSES):
        sub, data = knn.dataset_view(table, arrays, ds)
        tr = (sub.split == "train").to_numpy()
        te = (sub.split == "test").to_numpy()
        y = sub.label.to_numpy()[te]
        names = [s.replace("Isocortex_layer", "L").replace("L23", "L2/3") for s in CLASSES[ds]]
        positions = []
        for j, (key, name, units, long_name) in enumerate(KEYS):
            raw = data[key][:, :64]
            embed = knn.unit(raw)
            pca = PCA(n_components=2, svd_solver="full").fit(embed[tr])
            xy = pca.transform(embed[te])
            positions.append(xy)
            coords.append(pd.DataFrame(dict(dataset=ds, name=sub.name[te].values, label=y, descriptor=long_name,
                                            PC1=xy[:, 0], PC2=xy[:, 1])))
            a = axes[i, j]
            b = pcaxes[i, j]
            for c, label in enumerate(names):
                color, symbol = STYLE[label]
                mask = (y == c)
                q = np.percentile(raw[te][mask], [25, 50, 75], axis=0)
                ranks = np.arange(1, 65)
                a.fill_between(ranks, q[0], q[2], color=color, alpha=.10)
                a.plot(ranks, q[1], color=color, lw=1.3, label=label)
                b.scatter(*xy[mask].T, s=18, color=color, marker=symbol, alpha=.73,
                          linewidths=.25, edgecolors="white", label=label)
                bands.append(pd.DataFrame(dict(dataset=ds, descriptor=long_name, cell_class=label, neurons=int(mask.sum()),
                                               rank=ranks, q25=q[0], median=q[1], q75=q[2])))
            a.set_yscale("symlog", linthresh=1 if j == 0 else .001)
            a.set(xlim=(1, 64), xlabel="Eigenvalue rank", ylabel=f"Magnitude ({units})", title=f"{ds}  ·  {name}")
            expl = 100 * pca.explained_variance_ratio_
            b.set(xlabel=f"PC1 ({expl[0]:.0f}%)", ylabel=f"PC2 ({expl[1]:.0f}%)", title=f"{ds}  ·  {name}")
        lims = np.concatenate(positions)
        for dim, method in enumerate(["set_xlim", "set_ylim"]):
            low, high = lims[:, dim].min(), lims[:, dim].max()
            pad = (high - low) * .07
            for b in pcaxes[i]:
                getattr(b, method)(low - pad, high + pad)
        for a in (axes[i, 1], pcaxes[i, 1]):
            a.legend(loc="best", ncol=2, fontsize=7, framealpha=.9)
        leading = data[knn.DEFAULT][te, 0]
        bins = np.geomspace(leading.min() * .99, leading.max() * 1.01, 15)
        for c, label in enumerate(names):
            z = leading[y == c]
            haxes[i].stairs(np.histogram(z, bins=bins)[0] * 100 / len(z), bins, label=label, color=STYLE[label][0], lw=1.5)
            counts = np.histogram(z, bins=LEADING_BINS)[0]
            q = np.percentile(data[knn.DEFAULT][te][y == c, 15], [25, 50, 75])
            summary.append(dict(dataset=ds, cell_class=label, neurons=len(z), leading_below_0p5e6=counts[0],
                                leading_0p5e6_to_1e6=counts[1], leading_1e6_to_2e6=counts[2], leading_above_2e6=counts[3],
                                rank16_q25=q[0], rank16_median=q[1], rank16_q75=q[2]))
        haxes[i].set(xscale="log", title=ds, xlabel="Leading AG eigenvalue (µm)", ylabel="Neurons in bin (%)")
        haxes[i].legend(ncol=3, fontsize=7)
    save(spec, folder / "spectrum_by_class")
    save(scatter, folder / "test_pca")
    save(hist, folder / "leading_eigenvalue_histograms")
    pd.concat(coords, ignore_index=True).to_csv(folder / "test_coordinates.csv", index=False)
    pd.concat(bands, ignore_index=True).to_csv(folder / "spectrum_by_class.csv", index=False)
    pd.DataFrame(summary).to_csv(folder / "leading_eigenvalue.csv", index=False)
    print("figures and their data written to", folder)


if __name__ == "__main__":
    main()
