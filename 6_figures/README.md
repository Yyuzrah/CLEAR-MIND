> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Figures R2-R4

```
python 6_figures/make_figures.py      # about 10 s, writes rerun/
```

The script reads `3_knn_descriptors/features.npz` and draws, for the test neurons of folds 8-9:

| File | Figure |
|---|---|
| `spectrum_by_class.pdf` | Fig. R2: class-wise median spectra and interquartile bands of Exact quotient AG (left) and of the highest weighted Laplacian modes (right) |
| `test_pca.pdf` | Fig. R3: PCA fitted on the unit-normalized spectra of folds 0-6, test neurons projected |
| `leading_eigenvalue_histograms.pdf` | Fig. R4: leading AG eigenvalue per class, shared logarithmic bins |

The plotted numbers are saved next to the figures: `test_coordinates.csv` (PCA coordinates of every test neuron), `spectrum_by_class.csv` (quartiles of every eigenvalue rank per class) and `leading_eigenvalue.csv` (per class: counts of test neurons by leading eigenvalue, and the quartiles of the rank-16 eigenvalue quoted in Appendix C.6). Expected: the figures and CSV files of `outputs/` (PCA coordinates agree up to floating-point rounding across machines); for JML-4 the rank-16 medians are 7,327 (L2/3), 22,190 (L5), 7,613 (L6) and 9,307 um (VPM).

```
python 6_figures/fig_r3_caption_numbers.py
```

`fig_r3_caption_numbers.py` prints the numbers quoted in the Fig. R3 caption: the lambda_1 entry of the unit-normalized spectra over all 2,103 neurons (Exact quotient AG: minimum 0.75149, median 0.89912; highest Laplacian modes: median 0.28498), the Spearman correlation of PC1 with |lambda_N|/lambda_1 for each dataset (0.9965, 0.9989, 0.9986), and, for the BIL-6 test neurons, the AUC of VPM against L4 and against L6 along PC1 (0.9900 and 0.9459).
