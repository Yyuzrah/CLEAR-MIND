# Version-One Reproduction Repository Migration Report

## 1. Outcome

The migration does not publish the original workspace or preserve old experiment
directories under a new name. The public repository implements one version-one
pipeline:

```text
SWC/metadata
→ locked dataset manifests
→ Arakelov–Green spectrum
→ three model-specific adapters
→ GNN-alpha / TreeLSTM-arch3 / MLP
→ training, evaluation, and local artifacts
```

The implementation is an installable Python package driven by YAML configuration
and one command-line interface. The reference notebooks retain their saved outputs
and execution state as provenance records; they are not the supported runtime. The
MLP notebook has an English reader-facing edition in which only Markdown cells were
rewritten. Its code cells, execution counts, and outputs remain identical to the
historical source.

## 2. Version-one scope

Included:

- upstream method: Arakelov–Green/effective-resistance spectrum;
- datasets: ACT-4, JML-4, and BIL-6;
- downstream models: `gnn_alpha`, `treelstm_arch3`, and `mlp`;
- folds 0–7 for training and folds 8–9 for testing;
- formal seeds: 42, 1453, 666, 114514, and 1919810;
- data audits, spectrum caching, training, evaluation, predictions, logs,
  checkpoints, and environment records.

Explicitly excluded:

- Tropical Jacobian;
- CVP/Babai;
- Graph Transformer;
- TreeMoCo and MorphVAE;
- tuning grids, ablation scripts, and unrelated notebooks;
- legacy caches, checkpoints, logs, and generated results;
- raw SWC data.

Excluded material is not part of the package and is not tracked by Git. The complete
migration workspace remains outside the public repository.

## 3. Implementation provenance

| Version-one component | Authoritative source | Packaged result |
|---|---|---|
| Exact data loader, spectrum, and MLP | Fixed MLP result notebook | Split into data, spectral, MLP adapter, model, and trainer modules |
| TreeLSTM-arch3 | TreeLSTM reference notebook | Split into tree adapter, batch collation, model, and trainer modules |
| GNN-alpha architecture | GNN reference notebook | Isolated in `models/gnn_alpha.py` |
| Deterministic GNN point-cloud implementation | `EM/vbeta_gnn.py` | Only the baseline point-cloud construction was migrated; other V.beta methods were excluded |
| Strict BIL GNN protocol | `EM/run_valpha_bil_reg_5seeds.py` | Locked the `_reg.swc` cohort and five-seed training settings |

The GNN and TreeLSTM public notebook copies remain byte-identical to their historical
sources. The MLP public copy contains English Markdown while preserving every
non-Markdown cell exactly. Hashes and preservation policies are recorded in
[`SOURCE_MAPPING.md`](SOURCE_MAPPING.md) and `notebooks/reference/SHA256SUMS`.

## 4. Unified datasets

The three models no longer scan directories independently or load model-specific
CSV files. They consume the same `manifests/<dataset>.csv` rows. Each row fixes:

- `sample_id`;
- exact relative path and filename;
- SWC SHA-256;
- label name and integer ID;
- fold and train/test phase;
- whether a BIL file is a registered `_reg.swc` sample.

Locked cohort sizes:

| Dataset | Train | Test | Total |
|---|---:|---:|---:|
| ACT-4 | 400 | 95 | 495 |
| JML-4 | 332 | 76 | 408 |
| BIL-6 | 958 | 242 | 1,200 |

All three BIL downstream models use the same 1,200 `_reg.swc` files. The legacy
workspace contains incompatible 331/77 and 332/76 JML fold assignments. Version one
uses the 332/76 split consumed by the fixed MLP result notebook. The older 77-test
TreeLSTM JML number is retained only as historical provenance and is not treated as
an exact regression target for the canonical manifest. Evidence is documented in
[`DATA_PROVENANCE.md`](DATA_PROVENANCE.md).

## 5. Modular design

```text
data/
  registry → manifest → SWC parser → NeuronRecord
                                      │
spectral/
  ArakelovGreenSpectrum ─────────────┘
                 │
                 ├── MLPAdapter      → MLP
                 ├── GNNAdapter      → GNNAlpha
                 └── TreeLSTMAdapter → TreeLSTMArch3
                                            │
                                  shared trainer/artifact writer
```

- `data/` is the only data entry point;
- `spectral/` owns spectrum extraction and cache keys;
- `adapters/` converts one `NeuronRecord` into model-specific inputs;
- each file in `models/` defines one downstream model and does not load data or
  calculate spectra;
- `training/` centralizes seeds, data loaders, optimizer, scheduler, early stopping,
  evaluation, and artifact writing;
- `configs/` separates dataset, spectral, model, and nine experiment definitions.

A future spectrum method can register a new extractor. A future downstream model can
add an adapter, model, and configuration without changing existing manifests or
other downstream models.

## 6. Paths, cache, and environment

The supported pipeline contains no legacy absolute data paths. Users provide the raw
data location with `--data-root` or `EM_CONNECTOME_DATA_ROOT`. Generated files are
written only to:

- `cache/`: local spectrum and data-index caches;
- `runs/`: one isolated directory per training run.

Both directories are ignored by Git. A spectrum cache key includes the SWC hash,
manifest hash, method name, all spectral parameters, and implementation version, so
a cache from another dataset or method cannot be reused silently.

Environment entry points:

- `pyproject.toml`: package and dependency definitions;
- `requirements.txt`: training/runtime pip entry point;
- `requirements-dev.txt`: development/validation pip entry point;
- `environment.yml`: conda entry point;
- `requirements/reference-environment.txt`: exact versions from the validation host;
- each run's `environment.json`: actual Python, PyTorch, CUDA, cuDNN, and device.

## 7. Reproduction evidence

Version one establishes five validation layers:

1. data: sample counts, IDs, folds, labels, paths, and SWC hashes;
2. spectrum: real-sample golden vectors generated from the preserved MLP code cell;
3. adapters: GNN point clouds and TreeLSTM structure, parent direction, root, and
   spectrum/label alignment;
4. models: notebook classes loaded dynamically and compared by parameter keys,
   shapes, fixed-batch logits, and loss;
5. experiments: five formal seeds for each of nine configurations, checked against
   published mean tolerances.

The full `3 × 3 × 5 = 45` run matrix was completed with Python 3.11.9,
PyTorch 2.11.0+cu128, CUDA 12.8, cuDNN 9.19, and an RTX 5080 Laptop GPU.

| Dataset | Model | Accuracy (%) | Mean Macro-F1 (%) | Interpretation |
|---|---|---:|---:|---|
| ACT-4 | GNN-alpha | 57.26 ± 1.95 | 55.92 | New v1 baseline; +0.21 pp from the 57.05% engineering baseline |
| ACT-4 | TreeLSTM-arch3 | 71.37 ± 1.55 | 70.70 | Historical 72.42%; −1.05 pp, within the ±2 pp tolerance |
| ACT-4 | MLP | 63.79 ± 1.07 | 62.67 | All five seed results match the result notebook exactly |
| JML-4 | GNN-alpha | 70.79 ± 2.55 | 74.71 | New canonical baseline |
| JML-4 | TreeLSTM-arch3 | 76.05 ± 2.81 | 80.19 | New canonical baseline |
| JML-4 | MLP | 60.26 ± 1.53 | 60.00 | All five seed results match the result notebook exactly |
| BIL-6 | GNN-alpha | 91.82 ± 0.48 | 84.42 | −0.91 pp from the strict 92.73% baseline, within the ±3 pp tolerance |
| BIL-6 | TreeLSTM-arch3 | 93.80 ± 0.26 | 87.83 | Matches the historical mean exactly |
| BIL-6 | MLP | 73.31 ± 0.77 | 63.13 | All five seed results match the result notebook exactly |

Following the historical protocol, folds 8–9 are used both for checkpoint selection
and result reporting. These values are therefore test-selected reproduction results,
not independent blind-test estimates.

The mean tolerances are ±0.5 percentage points for MLP, ±2 for TreeLSTM, and ±3
for GNN. Tolerances and the corresponding config, manifest, and feature-set hashes
are locked in `reports/v1_regression_targets.yaml`; the path-normalized 45-run result
table is committed as `reports/v1_formal_results.csv`.

The result gate:

```powershell
em-connectome verify results
```

requires five unique formal seeds for each experiment and checks:

- train/test counts;
- resolved-config SHA-256;
- manifest SHA-256;
- Arakelov–Green feature-set hash;
- finite accuracy in `[0, 1]`;
- five-seed means within the declared tolerances.

## 8. Interpretation boundaries

- The MLP notebook is a complete fixed-configuration result source and supports
  strict seed-by-seed comparison.
- TreeLSTM ACT/BIL use the same cohorts as their historical runs, but CUDA/PyTorch
  scatter and reduction behavior permits small sample-level differences.
- The historical TreeLSTM JML 77-test result cannot validate the canonical 76-test
  manifest; the version-one result is a new canonical baseline.
- The GNN notebook output stops at ACT epoch 1. Its architecture and adapter support
  code-equivalence tests, while ACT/JML statistics are version-one baselines.
- Strict BIL GNN results can be compared with the historical `_reg.swc` five-seed
  runner.

The report does not present incompatible splits, incomplete notebooks, or results
from different methods as strict reproductions.

## 9. Public repository boundary

GitHub contains only:

```text
configs/
docs/
manifests/
notebooks/reference/
reports/
requirements/
requirements.txt
requirements-dev.txt
scripts/
src/em_connectome/
tests/
.gitignore
environment.yml
pyproject.toml
README.md
```

`runs/`, `cache/`, Python/tool caches, the legacy workspace, and the complete local
migration archive are excluded from Git.

## 10. Final audit

| Check | Result |
|---|---|
| `pip install -e ".[train,test]"` | Installed `em-connectome==0.1.0` successfully |
| Wheel build | Successful; wheel contains only package source and metadata |
| `pytest -q` | 26 passed |
| `ruff check .` | Passed |
| `ruff format --check .` | Passed |
| `em-connectome verify all` | Data, features, adapters, and results passed |
| Final smoke matrix | 9/9 experiments completed |
| Formal regression matrix | 45/45 runs completed |
| Formal run artifacts | 45 unique directories, 360 required files, and 6,195 prediction rows with no missing artifacts |
| Reference notebooks | 3/3 valid nbformat documents; GNN/TreeLSTM byte-identical; MLP non-Markdown cells identical |
| Publication candidates | 77 tracked files; no candidate exceeds 1 MiB |
| Local-only artifacts | `cache/` and `runs/` are ignored |

Version one is therefore a minimal but complete installable, command-line driven,
verifiable, extensible reproduction repository. The remaining publication decisions
are the open-source license and final author metadata; `pyproject.toml` retains an
explicit pre-release license placeholder.
