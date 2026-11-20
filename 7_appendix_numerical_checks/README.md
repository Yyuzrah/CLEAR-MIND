> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# Numerical checks

The scripts check results of Appendix A numerically, work through the example of Figure R1, and compute the numbers of Figure 1 (numpy, scipy, networkx, matplotlib; no GPU). Run them from the package root, e.g. `python 7_appendix_numerical_checks/verify_propositions.py`. Result numbers refer to the revised paper.

| Script | What it checks | Expected output |
|---|---|---|
| `verify_exact_tropical_distance.py` | Theorem A.30(ii), d_Trop² = d_Γ − M_AG, against brute-force CVP; Theorem A.26 on the spanning-tree lift of every pair; how often Babai rounding of a spanning-tree lift is suboptimal (Remark A.34) | max discrepancy of order 1e-14 (the last digits depend on the BLAS library and the CPU: 4.5e-14, the value in the paper, with numpy 1.23.5 and MKL; 3.6e-14 with the pinned packages and OpenBLAS); Babai suboptimal on 877 of the 7,726 pairs (11.4%); runs in about 1.5 minutes |
| `verify_propositions.py` | Period matrix = LCA depths (Prop. A.18), morphological closed form (Cor. A.28), rank-one update (Prop. A.21), bridge contraction (Prop. A.23), Lipschitz bound (Prop. A.41), EDM sign pattern (Thm. A.39), subdivision invariance (Lemma A.25), unit-current bound (Lemma A.9), strongly regular example (Cor. A.45) | all errors ≤ 1e-9 |
| `verify_half_circuit_criterion.py` | Prop. A.36: closest lift ⇔ half-circuit condition | agreement on all instances |
| `verify_canonical_measure_and_drg.py` | Prop. A.11(iv) (canonical measure, equilibrium identity); resistance formula for distance-regular graphs (Prop. A.44) | errors ≤ 1e-13 |
| `verify_worked_examples.py` | A two-chord instance of the loop interaction of Remark A.22 and Cor. A.28 (M_AG(w, v1) = 2/3 with one chord, 5/8 with two) and the theta-graph example in the proof of Prop. A.35(iii) | 2/3 → 5/8; 0.54 vs 0.74 |
| `verify_fig_r1_example.py` | The worked 5-node example of Figure R1 and Appendix C.6: Kron reduction, M_AG(x, y) = 1.5, d_Trop(x, y)² = 0.5, the matrix M_AG and its spectrum | eigenvalues 6.0814, −0.8949, −1.5000, −3.6865 |
| `make_figure1.py` | The numbers of Figure 1 for its ACT-4 reconstruction, with a plot of each step | prints |V| = 2867, |V_core| = 74, g = 2, two contracted bridges of 1.7 and 2.5 um, largest M_AG entry 662 um, lambda1 = 18113.7; writes `rerun/figure1.pdf` |

`make_figure1.py` reads SWC files from `data/raw/` or `CLEARMIND_RAW` (see `data/README.md`).
