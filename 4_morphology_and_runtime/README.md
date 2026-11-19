> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Extraction time and morphology statistics (Table R6, Appendix C.3 and C.4)

Each script writes to `rerun/` by default; the tables of the paper are in `outputs/`.

| Command | What it computes | Files |
|---|---|---|
| `python 4_morphology_and_runtime/runtime_and_geometry.py` | Table R6 (stage medians, total median and 90th percentile by core size). Per dataset at eps = 20 um, tau = 5 um: share of neurons that gain a cycle, that have no tip within eps of the soma, that have a contracted bridge, and whose quotient has fewer than K vertices. Training neurons that gain a cycle, by class. Seconds. | `tableR6_runtime_by_size.csv`, `geometry_summary.csv`, `cycles_by_class.csv`, `per_neuron.csv` |
| `python 4_morphology_and_runtime/kron_preservation.py` | Kron reduction on the 1,475 training neurons: branch-vertex and tip IDs, cable length, share of samples removed in the 25% densest neurons, Spearman correlation of the leading eigenvalue with cable length; the bend and side-branch example. Reads the SWC files, about 5 min. | `kron_summary.csv`, `kron_training_neurons.csv`, `bend_example.json` |
| `python 4_morphology_and_runtime/wl_and_scale_examples.py` | 1-WL refinement on the 400 BREC pairs next to the AG-spectrum decision of `2_brec/`; the 4-vertex arbor dilated by 2; K_{3,3} versus the triangular prism. Needs `data/BREC/brec_v3.npy`, seconds. | `wl_on_brec.csv`, `wl_and_scale_examples.json` |

`runtime_and_geometry.py` reads the per-neuron sizes, counts and stage times stored in `3_knn_descriptors/features.npz`. These times were measured by `3_knn_descriptors/extract.py features` with 4 parallel SWCs and one BLAS thread each. They are measured run times: after a new extraction, `--features 3_knn_descriptors/rerun/features.npz` gives the same counts and percentages and different absolute times.

Expected values: Table R6 total medians 75.8, 134.7, 284.9 and 563.6 ms; new cycles in 41.2%, 3.7% and 16.8% of ACT-4, JML-4 and BIL-6 neurons; 0 branch-vertex and tip mismatches; Spearman correlations 0.98, 0.90 and 0.79; 1-WL separates 0 of 400 BREC pairs and the AG spectrum 217.
