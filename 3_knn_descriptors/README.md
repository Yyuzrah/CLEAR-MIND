> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Cosine k-NN on standalone descriptors (Tables 3, 4, R3, R4, R5)

Every row of these tables is a uniform-vote cosine k-NN on one descriptor or a concatenation of descriptors, with no training. Folds 0-6 are the reference set, fold 7 alone picks k from {1, 3, 5, 10, 20} and raw or training-standardized inputs (ties go to the smaller k, then to raw inputs), and folds 8-9 are scored once (95 / 77 / 242 test neurons).

## Check the tables from the saved descriptors (seconds, no GPU)

```
python 3_knn_descriptors/evaluate.py select
python 3_knn_descriptors/evaluate.py test
```

`select` uses folds 0-7 only and writes `rerun/selection.json`. `test` reads that file and writes `rerun/all_results.csv` (24 k-NN rows x 3 datasets), `rerun/predictions/` (one prediction per test neuron and descriptor) and one CSV per table, and prints the comparison with `outputs/all_results.csv`. Expected: the same selections and a largest metric difference of 0.

| File in `outputs/` | Table |
|---|---|
| `table3_lattice_vs_exact_ag.csv` | Table 3: explicit lattice pipeline (period matrix and Babai rounding) and Exact quotient AG at eps = 20, tau = 5 |
| `table4_ablation.csv` | Table 4: Exact quotient AG with and without augmentation and bridge contraction |
| `tableR3_descriptors.csv` | Table R3: Submitted AG, Exact quotient AG, Laplacian readouts, Sholl, radial TMD, full-tree H0, five statistics and their combinations |
| `tableR4_threshold_grid.csv` | Table R4: Exact quotient AG on the grid eps in {0, 20, 50}, tau in {0, 5, 10}; the setting chosen by fold 7 for each dataset is in `selection.json` under `grid_by_validation` |
| `tableR5_readout_length.csv` | Table R5: readout length K in {16, 32, 64, 128} |

`features.npz` (14 MB) holds the descriptors of the 2,103 neurons in the order of `data/splits.csv`, together with the per-neuron sizes, chord and bridge counts and computation times used in `4_morphology_and_runtime/`. `descriptors.py` defines every descriptor and `knn.py` the classifier and the list of rows.

## Recompute the descriptors from the SWC files

```
python 3_knn_descriptors/extract.py features      # 4 workers, about 15 min
python 3_knn_descriptors/extract.py persistence   # 4 workers, about 10 min
python 3_knn_descriptors/extract.py lattice       # 10 workers, about 15 min
python 3_knn_descriptors/pack_features.py         # writes rerun/features.npz
python 3_knn_descriptors/evaluate.py select --features 3_knn_descriptors/rerun/features.npz --out 3_knn_descriptors/rerun_extracted
python 3_knn_descriptors/evaluate.py test --features 3_knn_descriptors/rerun/features.npz --out 3_knn_descriptors/rerun_extracted
```

The SWC files are read from `data/raw/` (see `data/README.md`); `--per-dataset N` restricts a run to the first N neurons of each dataset. The descriptors are deterministic. The time columns (Table 3 and Table R6) are measured run times, so a rerun gives different absolute times; `features.npz` stores the times measured for the paper.
