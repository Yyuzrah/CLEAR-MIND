# EM Connectome Reproduction

Minimal, installable reproduction package for the version-one experiment:

```text
SWC + metadata
→ locked ACT-4 / JML-4 / BIL-6 manifests
→ Arakelov–Green effective-resistance spectrum
→ model-specific adapter
→ GNN-alpha / TreeLSTM-arch3 / MLP
→ local metrics, predictions, logs, and checkpoints
```

Version one contains exactly one upstream method, three datasets, and three
downstream models. Tropical Jacobian, CVP/Babai, Graph Transformer, TreeMoCo,
MorphVAE, hyperparameter sweeps, and unrelated legacy experiments are out of
scope.

## Install

Python 3.10 or newer is required. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[train,test]"
```

For a conda-based installation:

```powershell
conda env create -f environment.yml
conda activate em-connectome
```

See [`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md) for the validated CUDA
environment and the exact reference package list.

## Prepare the local data root

Raw data are not committed. Point `EM_CONNECTOME_DATA_ROOT` at a directory with
the following layout:

```text
<data-root>/
├── info/
│   ├── ACT_info_swc_10folds.csv
│   ├── JML_info_swc_10folds.csv
│   └── BIL_info_swc_10folds.csv
└── raw/
    ├── allen_cell_type/swc/
    ├── janelia_mouselight/swc/
    └── bil/swc/
```

```powershell
$env:EM_CONNECTOME_DATA_ROOT = "D:\path\to\data"
em-connectome verify data
```

Committed manifests lock every sample ID, fold, label, relative path, and SWC
SHA-256. The three models always consume the same manifest for a dataset.
Dataset counts and the historical JML split difference are documented in
[`docs/DATA_PROVENANCE.md`](docs/DATA_PROVENANCE.md).

## Run the pipeline

Audit data and build the canonical index:

```powershell
em-connectome data audit --dataset bil6
em-connectome data build --dataset bil6
```

Build or validate Arakelov–Green features:

```powershell
em-connectome spectral build --dataset bil6
em-connectome verify features
em-connectome verify adapters
em-connectome verify results
```

Train one configured experiment:

```powershell
em-connectome train --config configs/experiments/bil6_gnn_alpha.yaml
```

Run the complete nine-experiment smoke matrix:

```powershell
em-connectome reproduce --experiment all --mode smoke
```

Run all five formal seeds for one experiment:

```powershell
em-connectome reproduce --experiment bil6_gnn_alpha --mode formal
```

Run the complete `3 datasets × 3 models × 5 seeds` formal matrix:

```powershell
em-connectome reproduce --experiment all --mode formal
```

Every run creates an isolated directory under `runs/` containing:

```text
checkpoint.pt
config.resolved.yaml
environment.json
history.csv
manifest_hash.json
metrics.csv
predictions.csv
train.log
```

`runs/`, spectra caches, raw data, and Python/tool caches are intentionally
Git-ignored.

## Design

```text
em_connectome.data
  SWC parser → manifest loader → NeuronRecord
                         │
em_connectome.spectral   │
  ArakelovGreenSpectrum ─┘
            │
            ├── MLPAdapter       → MLP
            ├── GNNAdapter       → GNNAlpha
            └── TreeLSTMAdapter  → TreeLSTMArch3
                                      │
                         shared trainer + artifact writer
```

Dataset parsing, spectrum generation, input adaptation, model definition, and
training are separate modules. A future upstream method can register a new
spectrum extractor; a future downstream model can add its own adapter, model,
and config without changing the canonical data layer.

## Reproduction gates

```powershell
em-connectome verify all
python -m pytest -q
python -m ruff check .
python -m ruff format --check .
```

The gates cover:

1. manifest counts, folds, labels, paths, and file hashes;
2. real-sample spectrum vectors extracted from the untouched MLP notebook;
3. GNN and TreeLSTM adapter fixtures;
4. notebook-equivalent parameter shapes, logits, and loss for all models;
5. an end-to-end CLI smoke matrix and five-seed formal regressions.

The three notebooks in `notebooks/reference/` remain byte-identical to their
result-producing sources, including saved outputs. They are provenance records,
not the supported runtime. See [`docs/SOURCE_MAPPING.md`](docs/SOURCE_MAPPING.md)
for their hashes and precise execution status.

The completed 45-run matrix is in
[`reports/v1_formal_results.csv`](reports/v1_formal_results.csv), with its
machine-checkable tolerances in
[`reports/v1_regression_targets.yaml`](reports/v1_regression_targets.yaml).
See the Chinese
[`migration report`](docs/MIGRATION_REPORT.zh-CN.md) for the aggregate table,
historical comparisons, exclusions, and known interpretation boundaries.

## Repository contents

```text
configs/                validated dataset, spectrum, model, and experiment YAML
docs/                   environment, data, source, and migration documentation
manifests/              locked ACT/JML/BIL sample indexes and hashes
notebooks/reference/    three untouched result-source notebooks
reports/                normalized formal results and regression tolerances
scripts/                provenance/golden-vector maintainer utilities
src/em_connectome/      installable package
tests/                  unit, golden, integration, and CLI tests
```
