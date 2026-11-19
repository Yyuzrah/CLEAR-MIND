> Part of the CLEARMIND code release. Copyright (c) 2026 the CLEARMIND authors. All rights reserved; see `LICENSE` at the repository root.

# BREC (Table 1, App. B.3, App. C.5)

Needs `data/BREC/brec_v3.npy` from the BREC release (see `data/README.md`).

## Training-free AG spectrum and Laplacian spectrum, all 32 relabelings

```
python 2_brec/spectrum_all_relabelings.py      # about 30 s with 4 CPU workers
```

For every pair, BREC provides 32 relabelings of both graphs and a reliability block of relabelings of the same graph. The script computes the sorted absolute AG spectrum and the Laplacian spectrum of every graph and writes `rerun/spectrum_pairs.csv` (one row per pair) and `rerun/spectrum_summary.json`; the shipped copies are in `outputs/`. Expected summary:

| | AG spectrum | Laplacian spectrum |
|---|---:|---:|
| pairs separated (relative tolerance 1e-8) | 217 / 400 (60 Basic, 50 simple regular, 100 Extension, 7 CFI) | 213 / 400 (60, 50, 100, 3 CFI) |
| pairs whose decision changes across the 32 relabelings | 0 | 0 |
| largest relative change of a spectrum across relabelings | 5.4e-14 | 4.9e-15 |
| smallest relative difference among separated pairs | 3.0e-8 | 1.1e-2 |

Every pair separated by the Laplacian spectrum is separated by the AG spectrum; the AG spectrum also separates the CFI pairs 273, 289, 307 and 317 (0-based). Any threshold between 1e-12 and 1e-8 gives the same 217 pairs.

## AG-GIN, ten seeds (Table 1, Tables R7 and R8)

```
python 2_brec/aggin_seed_replication/summarize.py      # instant, reads the saved per-seed counts
```

`aggin_seed_replication/outputs/per_seed.csv` holds the ten-seed AG-GIN run reported in the paper (protocol of App. B.3): seeds 100, 200, ..., 1000, each trained and scored on all 400 pairs at threshold 72.34. There is one row per statistic and seed with the number of separated pairs per category, the total, and the number of failed reliability checks; `regular` is the sum of the simple regular, strongly regular, 4-vertex condition and distance regular pairs. Both statistics are computed from the same trained models: `eq4` is Eq. (4) of BREC (q = 32, ridge 1e-7) and `brec_code` is the statistic of the BREC release code. A pair is separated when its statistic exceeds the threshold and differs from that of its reliability block, and a reliability check fails when the statistic of the reliability block reaches the threshold. Under `eq4` a separated pair is counted if its reliability check holds; under `brec_code` the count is that of the release code, and the reliability checks are listed in their own column.

`summarize.py` prints Table R8 (per seed: total, CFI pairs and failed reliability checks under both statistics), Table R7 and the AG-GIN row of Table 1; `--write DIR` also saves them as CSV files. Expected summary:

| | Eq. (4), q = 32 | BREC release code |
|---|---:|---:|
| mean over the ten seeds | 221.2 / 400 | 202.9 / 400 |
| best seed | 224 / 400 | 206 / 400 (seed 300) |
| CFI pairs, single-seed range | 6 to 12 | 1 to 3 |
| failed reliability checks | 95 / 4,000 | 0 / 4,000 |

AG-GIN row of Table 1 (seed 300, statistic of the BREC release code): Basic 59/60 (98.3%), Regular 49/140 (35.0%), Extension 96/100 (96.0%), CFI 2/100 (2.0%), total 206/400 (51.5%).
