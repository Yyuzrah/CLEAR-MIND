# Dataset provenance

Version one uses one manifest per dataset. Every downstream model reads these
same rows; models are not allowed to rescan raw directories or choose their own
folds.

## Canonical source

The locked manifests are generated from the exact-loader data copy used by the
formal fixed-configuration MLP notebook:

```text
external/treemoco/data/
├── info/
└── raw/
```

Only selected columns and hashes are committed in `manifests/`; the full
metadata tables and raw SWC files are not copied into this repository. All
selected raw SWC files were byte-compared with `EM/data/raw` and are identical,
so either raw directory can satisfy an already-built manifest.

The selection rule is:

1. normalize `2/3` to `23` in `structure_merge__acronym`;
2. retain only the configured ACT-4, JML-4, or BIL-6 classes;
3. require the exact `swc__fname` to exist, with no fuzzy or prefix matching;
4. assign folds 0–7 to train and folds 8–9 to test;
5. require every selected BIL filename to end in `_reg.swc`.

## Locked cohort sizes

| Dataset | Train | Test | Total |
|---|---:|---:|---:|
| ACT-4 | 400 | 95 | 495 |
| JML-4 | 332 | 76 | 408 |
| BIL-6 | 958 | 242 | 1,200 |

## JML legacy split discrepancy

The old workspace contains two different JML fold assignments:

- `EM/data/info/JML_info_swc_10folds.csv`: 331 train / 77 test;
- `external/treemoco/data/info/JML_info_swc_10folds.csv`: 332 train / 76 test.

They select the same 408 SWC files with identical labels and byte-identical
content, but 119 samples cross the train/test boundary and only 61 samples keep
the exact same fold number. The saved TreeLSTM-arch3 JML result uses the older
77-sample test split (its percentage increments prove that denominator); the
formal MLP result notebook uses the newer 76-sample test split.

Version one chooses the newer fixed exact-loader split as the single
canonical default. Consequently, the historical TreeLSTM JML number is
provenance evidence, not a valid exact regression target for the canonical
JML manifest. The package must establish a new TreeLSTM JML five-seed baseline
on the locked manifest. ACT and BIL do not have this split discrepancy.

This choice prevents model-specific datasets while making the historical
difference explicit rather than silently claiming incompatible results are
directly comparable.
