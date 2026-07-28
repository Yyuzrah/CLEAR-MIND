# Source mapping

This document fixes the provenance boundary for version one. Only the three
result-source notebooks below are preserved in the public repository. Other
files in the original workspace are consulted during migration but are not
copied unless the resulting package code needs a documented implementation
source.

## Reference notebooks

| Package copy | Original workspace path | SHA-256 | Role | Saved execution status |
|---|---|---|---|---|
| `notebooks/reference/V.alpha_GNN.ipynb` | `EM/V.alpha_GNN.ipynb` | `7aa8cdc7c68cba1548028c264aef5ec74a133fab86fcdafdbd9e1d8ed41045c8` | GNN-alpha architecture and preprocessing provenance | Partial: setup and feature preparation were run; the saved output stops at ACT-4 epoch 1 and is not a completed result |
| `notebooks/reference/V.alpha_TreeLSTM_arch3.ipynb` | `EM/V.alpha_TreeLSTM_arch3.ipynb` | `8642222098205f93d3c348ad5e7c3df968e39890c7245f97f11f4ff8160d20a7` | TreeLSTM-arch3 architecture, adapter, training protocol, and saved five-seed results | Saved result-producing cells are present; the notebook as a whole has not been rerun top-to-bottom in the clean repository |
| `notebooks/reference/V.alpha_MLP_clean_champion_reproduction.ipynb` | `notebooks/mlp_champion_reproduction/V.alpha_MLP_clean_champion_reproduction.ipynb` | `48c6f729efcb8aa4a9096712e5e8c3a8057801adfcdbad8c274cfa9556bd0907` | Exact loader, Arakelov–Green spectrum, champion MLP, and five-seed protocol | All six code cells have saved execution counts and outputs; it was not rerun in the clean repository |

The copies are byte-identical to their sources. Run
`Get-FileHash -Algorithm SHA256 notebooks/reference/*.ipynb` on Windows or
`sha256sum -c notebooks/reference/SHA256SUMS` on a POSIX system to audit them.

## Package migration sources

| Version-one component | Authoritative source | Migration rule |
|---|---|---|
| Exact dataset cohort and MLP | Champion MLP reference notebook | Preserve exact-filename matching, class filters, fold split, feature normalization, model hyperparameters, and five seeds |
| Arakelov–Green spectrum | Champion MLP reference notebook | Preserve SWC parsing, degree-two skeletonization, leaf-to-root closure, bridge threshold behavior, eigenvalue ordering, and zero padding |
| TreeLSTM-arch3 | TreeLSTM reference notebook | Separate tree adapter, model definition, and training loop without changing tensor conventions |
| GNN-alpha | GNN reference notebook; `EM/vbeta_gnn.py` may be used only as an engineering transcription aid | Prove architecture and preprocessing equivalence before accepting fixed-batch model tests; do not import CVP or other V.beta methods |
| Strict GNN BIL protocol | `EM/run_valpha_bil_reg_5seeds.py` | Use only to recover the exact `_reg.swc` cohort and formal GNN run settings |

## Explicit exclusions

Version one does not migrate Tropical Jacobian, CVP/Babai, Graph Transformer,
TreeMoCo, MorphVAE, tuning grids, ablation notebooks, cached features,
checkpoints, logs, or generated result directories.

## Notebook validation gap

The notebooks are historical experiment records, not the supported runtime.
They are deliberately not edited or cleared. They were not rerun top-to-bottom:
the GNN notebook has no complete saved run, the TreeLSTM notebook contains many
unexecuted exploratory cells, and the MLP notebook embeds the legacy exact
loader. All three are valid nbformat documents and retain their original
execution counts and outputs.

The clean package is the supported rerunnable path. It validates each migrated
layer with data hashes, real-sample spectrum golden vectors, adapter fixtures,
fixed-batch model checks, a nine-experiment smoke matrix, and the complete
45-run formal regression recorded in `reports/v1_formal_results.csv`.
