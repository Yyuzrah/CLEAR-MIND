> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Spectrum-only MLP, validation-selected (MLP rows of Table 5 and Table R1)

An MLP 64 -> 96 -> 48 -> C is trained on the 64-dimensional spectrum alone: setting **B** uses Submitted AG and setting **C** uses Exact quotient AG. Folds 0-6 train, inputs are standardized with training statistics, fold 7 alone selects the earliest epoch with the highest validation accuracy (patience 25, at most 800 epochs), and folds 8-9 are scored once. Seeds 42, 114514, 1919810, 666, 1453; Adam (lr 3e-3, weight decay 1e-6), label smoothing 0.05, batch 32, CPU.

## Check the rows from the saved checkpoints (seconds)

```
python 5_spectrum_only_mlp/spectrum_mlp.py test
```

This scores the 30 checkpoints of `outputs/checkpoints/` (1.4 MB) on folds 8-9 and writes `rerun/test_predictions/`, `rerun/per_seed.csv` and `rerun/table5_mlp_rows.csv` (5-seed mean and sample SD of accuracy and macro-F1). Expected: the same files as in `outputs/`, with accuracies 50.74 / 49.61 / 68.43 (Submitted AG) and 42.74 / 50.39 / 76.28 (Exact quotient AG) on ACT-4 / JML-4 / BIL-6.

## Retrain (CPU, about one minute)

```
python 5_spectrum_only_mlp/spectrum_mlp.py train
python 5_spectrum_only_mlp/spectrum_mlp.py test --run-dir 5_spectrum_only_mlp/rerun
```

`train` uses folds 0-7 only. It writes `rerun/checkpoints/`, `rerun/epoch_logs/` (fold-7 correct count after every epoch) and, once all 30 runs have finished, `rerun/selection.json` (chosen epoch, stopping epoch, fold-7 count); `test` reads that file. With the package versions of `requirements.txt`, training on CPU is deterministic and a rerun gives the epoch logs, weights and predictions of `outputs/` again.

The spectra are read from `3_knn_descriptors/features.npz`, in single precision as given to the backbones of `1_validation_selected_backbones/`.
