# Source mapping

This document fixes the provenance boundary for version one. Only the three
result-source notebooks below are included in the public repository. Other files
in the original workspace were consulted during migration but were not copied
unless a packaged component required a documented implementation source.

## Reference notebooks

| Public copy | Historical source | Historical SHA-256 | Public SHA-256 | Preservation policy | Saved execution status |
|---|---|---|---|---|---|
| `notebooks/reference/V.alpha_GNN.ipynb` | `EM/V.alpha_GNN.ipynb` | `7aa8cdc7c68cba1548028c264aef5ec74a133fab86fcdafdbd9e1d8ed41045c8` | same as source | Byte-identical | Setup and feature preparation were run; saved output stops at ACT-4 epoch 1 and is not a completed result |
| `notebooks/reference/V.alpha_TreeLSTM_arch3.ipynb` | `EM/V.alpha_TreeLSTM_arch3.ipynb` | `8642222098205f93d3c348ad5e7c3df968e39890c7245f97f11f4ff8160d20a7` | same as source | Byte-identical | Result-producing cells are saved; the notebook was not rerun top-to-bottom in the clean repository |
| `notebooks/reference/V.alpha_MLP_clean_champion_reproduction.ipynb` | `notebooks/mlp_champion_reproduction/V.alpha_MLP_clean_champion_reproduction.ipynb` | `48c6f729efcb8aa4a9096712e5e8c3a8057801adfcdbad8c274cfa9556bd0907` | `50d0cd40e851750ce1258a465303c062f722570fa738e0dbd0bafe829043603b` | English Markdown edition; all non-Markdown cells are identical to the historical source | All six code cells retain their saved execution counts and outputs; the notebook was not rerun in the clean repository |

`notebooks/reference/SHA256SUMS` audits the current public copies. For the MLP
notebook, the canonical digest of notebook metadata, Markdown cell positions, and
all non-Markdown cell contents is
`12c49567f9fb0d5f085742743fa51a45b663e3b696f59662fd3b7ea74d159ad7` for both
the historical source and the English public copy. The regression test recomputes
this digest, so changes to code, execution counts, outputs, or code-cell metadata
cannot be hidden by updating the public-file SHA-256.

## Package migration sources

| Version-one component | Authoritative source | Migration rule |
|---|---|---|
| Exact dataset cohort and MLP | Fixed MLP result notebook | Preserve exact-filename matching, class filters, fold split, feature normalization, model parameters, training settings, and five seeds |
| Arakelov–Green spectrum | Fixed MLP result notebook | Preserve SWC parsing, degree-two skeletonization, leaf-to-root closure, bridge-threshold behavior, eigenvalue ordering, and zero padding |
| TreeLSTM-arch3 | TreeLSTM reference notebook | Separate the tree adapter, model definition, and training loop without changing tensor conventions |
| GNN-alpha | GNN reference notebook; `EM/vbeta_gnn.py` used only as an engineering transcription aid | Prove architecture and preprocessing equivalence before accepting fixed-batch model tests; do not import CVP or other V.beta methods |
| Strict GNN BIL protocol | `EM/run_valpha_bil_reg_5seeds.py` | Use only to recover the exact `_reg.swc` cohort and formal GNN run settings |

Historical identifiers such as `CHAMPION` and `ChampionMLP` remain inside the MLP
code cell so its executed computational record stays unchanged. The public Markdown
describes one fixed protocol and does not present a parameter search or
candidate-model comparison. Configuration source references retain the historical
source SHA-256 so the resolved hashes attached to the 45 completed runs remain
auditable; the current public-copy SHA-256 is recorded separately above.

## Explicit exclusions

Version one does not migrate Tropical Jacobian, CVP/Babai, Graph Transformer,
TreeMoCo, MorphVAE, parameter-search grids, ablation notebooks, cached features,
checkpoints, logs, or generated result directories.

## Notebook validation gap

The notebooks are historical experiment records, not the supported runtime. They
were not rerun top-to-bottom: the GNN notebook has no complete saved run, the
TreeLSTM notebook contains unexecuted exploratory cells, and the MLP notebook embeds
the legacy exact loader. All three are valid nbformat documents and retain their
historical code execution counts and outputs.

The clean package is the supported rerunnable path. It validates each migrated layer
with data hashes, real-sample spectrum golden vectors, adapter fixtures, fixed-batch
model checks, a nine-experiment smoke matrix, and the complete 45-run formal
regression recorded in `reports/v1_formal_results.csv`.
