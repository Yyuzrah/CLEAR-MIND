> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Data

## Neuron reconstructions (ACT-4, JML-4, BIL-6)

The SWC reconstructions are public: ACT-4 from the Allen Cell Types Database, JML-4 from the Janelia MouseLight project, and BIL-6 from the Brain Image Library, with the 10-fold assignment of the benchmark of Chen et al. (2022). Place them as

```
data/raw/allen_cell_type/swc/<name>
data/raw/janelia_mouselight/swc/<name>
data/raw/bil/swc/<name>
```

where `<name>` is the `name` column of `splits.csv`. To keep the files elsewhere, set the environment variable `CLEARMIND_RAW` to the folder that contains `allen_cell_type/`, `janelia_mouselight/` and `bil/`.

## Files in this folder

- `splits.csv`: one row per neuron used in the paper (dataset, file name, fold, class label, split, specimen and donor ids). Split: train = folds 0-6, valid = fold 7, test = folds 8-9. Test sizes 95 / 77 / 242 (ACT-4 / JML-4 / BIL-6).
- `excluded.csv`: the 79 BIL-6 rows of the benchmark table whose SWC file name is not among the released files; matching by file name, as in the submission, leaves the 1,200 BIL-6 neurons of the paper.
- `point_cloud_seeds.csv`: the random seed used to sample the 1,024-point cloud of each neuron for each training seed (MorphoGNN). Using these seeds gives the same clouds and the same jitter (on subsampled training clouds and on every oversampled cloud, App. B.6) as in the paper.

## BREC

`2_brec/` and `4_morphology_and_runtime/wl_and_scale_examples.py` read `data/BREC/brec_v3.npy` from the BREC release (Wang & Zhang, 2024); to use a copy elsewhere, set `CLEARMIND_BREC` to the path of that file.
