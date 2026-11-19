> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Validation-selected backbone comparisons (Table 5, Tables R1-R2)

Folds 0-6 train, fold 7 alone selects the checkpoint, folds 8-9 are scored once. For each backbone (Tree-LSTM, MorphoGNN), dataset and seed there are three paired runs: **A** no AG (the spectral branch output is replaced by zeros, so the network keeps the same width and parameters), **B** Submitted AG, **C** Exact quotient AG. The runs of a seed share the seed, initial weights, inputs, minibatch order, optimizer and schedule; only the spectral input differs. Seeds 42, 114514, 1919810, 666, 1453.

## Check the tables from the saved runs (seconds, no GPU)

```
python 1_validation_selected_backbones/paired_report.py
```

This reads `treelstm/outputs/` and `morphognn/outputs/` and
1. checks the selection rule for all 90 runs against the per-epoch fold-7 logs (chosen epoch = first epoch with the highest fold-7 accuracy; training stops 25 epochs later or at epoch 150);
2. computes test accuracy and macro-F1 per run and the 5-seed means and SDs (Table 5, Table R1);
3. computes the paired differences between settings with 95% paired neuron-bootstrap intervals (Table R2): 10,000 draws over test neurons, each draw keeping all five seed predictions of a neuron.

Results are written to `rerun/` (`table5_R1.csv`, `paired_R2.csv`, `per_seed.csv`); the copies used in the paper are in `results/`. For Tree-LSTM on ACT-4 and BIL-6, where donor ids are available, `paired_R2.csv` also gives the interval with donors resampled in place of neurons (`donor_ci_low`, `donor_ci_high`).

What is saved per run in `<backbone>/outputs/`: `epoch_logs/` (fold-7 correct count after every epoch), `selection.json` (chosen epoch, stopping epoch, fold-7 count), `test_predictions/` (one prediction per test neuron). The Tree-LSTM checkpoints are in `treelstm/outputs/checkpoints/` (18 MB). The MorphoGNN checkpoints (about 290 MB) are not included; `morphognn/train_select.py` regenerates them.

## Rerun everything (one GPU)

```
python 1_validation_selected_backbones/treelstm/prepare.py           # per-neuron trees and both spectra (CPU, a few minutes)
python 1_validation_selected_backbones/treelstm/train_select.py
python 1_validation_selected_backbones/treelstm/score_test.py --run-dir 1_validation_selected_backbones/treelstm/rerun
python 1_validation_selected_backbones/morphognn/make_point_clouds.py
python 1_validation_selected_backbones/morphognn/train_select.py
python 1_validation_selected_backbones/morphognn/score_test.py --run-dir 1_validation_selected_backbones/morphognn/rerun
python 1_validation_selected_backbones/paired_report.py --tree-dir 1_validation_selected_backbones/treelstm/rerun --gnn-dir 1_validation_selected_backbones/morphognn/rerun
```

`train_select.py` uses folds 0-7 only and writes `selection.json` once all 45 runs of a backbone have finished; `score_test.py` reads that file and then scores folds 8-9. After `prepare.py`, `score_test.py` without arguments rescores the shipped Tree-LSTM checkpoints into `treelstm/rerun/test_predictions/` (expected: the files of `treelstm/outputs/test_predictions/`).

Settings: AdamW (lr 1e-3, weight decay 1e-4), cosine schedule over 150 epochs, gradient clipping 1.0, batch 128 (Tree-LSTM) and 32 (MorphoGNN), deterministic cuDNN. Spectra use eps = 20 um, tau = 5 um and K = 64, the defaults of the submitted implementation.

Code: `clearmind/submission_treelstm.py` and `clearmind/submission_morphognn.py` contain the models and the extractor of the submission; `clearmind/exact_ag.py` computes the Exact quotient AG spectrum (Algorithm 1).

## Rotated folds (App. C.7, Table R9)

The original study evaluated only folds 8-9 of Table 5. `rotated_folds/` repeats the protocol of Table 5 on four rotated splits that test on folds 0-7, never used for testing before; over R1-R4 every neuron of folds 0-7 is tested exactly once.

| Rotation | Test folds | Validation fold | Training folds |
|---|---|---|---|
| R1 | 0, 1 | 2 | 3-9 |
| R2 | 2, 3 | 4 | 0, 1, 5-9 |
| R3 | 4, 5 | 6 | 0-3, 7-9 |
| R4 | 6, 7 | 8 | 0-5, 9 |

Everything else is as above: the same `fit` functions, settings A/B/C, seeds, optimizer, schedule, selection rule and spectra. `train_rotation.py` writes `selection.json` for a job before it scores the test folds, once. One case needs a guard: for MorphoGNN on R4, JML-4 has 289 training neurons, so the last minibatch of 32 holds a single neuron, which batch normalization cannot train on; `morphognn/train_select.py` skips a one-neuron minibatch. No other run, including those of Table 5, has such a minibatch, so their results do not change.

### Check Table R9 from the saved runs (seconds, no GPU)

```
python 1_validation_selected_backbones/rotated_folds/report_rotations.py
```

This reads `rotated_folds/outputs/` and the Table 5 predictions and writes `per_split_runs.csv`, `per_split_summary.csv`, `pooled_summary.csv` and `pooled_paired.csv` in `rotated_folds/outputs/`: test accuracy and macro-F1 per run, per rotation, pooled over R1-R4 and pooled over all five splits, the 5-seed means and SDs, and the paired differences with 95% paired neuron-bootstrap intervals (10,000 draws, generator seeded 20261106). Table R9 gives the C-A rows of `pooled_paired.csv` (Exact quotient AG minus no AG), pooled over R1-R4 and over all five splits.

Saved per job in `rotated_folds/outputs/<rotation>/<backbone>/<dataset>/`: `epoch_logs/` (validation correct count after every epoch), `runs/` (chosen epoch, stopping epoch, validation count), `selection.json` and `test_predictions/`. Checkpoints are not included.

### Rerun (one GPU per job)

```
python 1_validation_selected_backbones/treelstm/prepare.py                    # if not done already
python 1_validation_selected_backbones/rotated_folds/make_rotation_clouds.py   # MorphoGNN point clouds for the rotations
python 1_validation_selected_backbones/rotated_folds/train_rotation.py --backbone treelstm --rotation R1 --dataset ACT-4
```

`run_all_rotations.py --jobs N` runs all 24 jobs (4 rotations x 3 datasets x 2 backbones), N at a time; finished runs are skipped when a job is restarted. Each job's 15 runs ran on one GPU, so the three settings of a seed are compared on the same hardware. Training is deterministic on a given GPU model; on a different GPU model the epoch logs can drift after some epochs, so a rerun may select slightly different epochs.
