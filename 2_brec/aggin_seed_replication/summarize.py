# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Tables R7 and R8 and the AG-GIN row of Table 1 from the per-seed BREC counts of AG-GIN (instant, no GPU).

outputs/per_seed.csv has one row per RPC statistic and seed (seeds 100, 200, ..., 1000, each on all 400 pairs,
threshold 72.34): separated pairs per category, their total, and failed reliability checks.
  eq4        Eq. (4) of BREC (q = 32, ridge 1e-7)
  brec_code  the statistic of the BREC release code
Printed:
  Table R8  per seed: total, CFI pairs and failed reliability checks under both statistics
  Table R7  per statistic: mean and best single-seed total, single-seed CFI range, failed reliability checks
  Table 1   AG-GIN row: the best seed under brec_code, in percent of the pairs of each category
Usage (from the package root):
  python 2_brec/aggin_seed_replication/summarize.py [--per-seed FILE] [--write DIR]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import argparse

import pandas as pd

STATISTICS = ["eq4", "brec_code"]
# categories of Table 1 and their number of pairs
TABLE1 = [("basic", 60), ("regular", 140), ("extension", 100), ("cfi", 100), ("total", 400)]


def table_r8(per_seed):
    parts = [per_seed[per_seed.statistic == stat].set_index("seed")[["total", "cfi", "reliability_failures"]]
             .add_suffix("_" + stat) for stat in STATISTICS]
    return pd.concat(parts, axis=1).reset_index()


def table_r7(per_seed):
    rows = []
    for stat in STATISTICS:
        g = per_seed[per_seed.statistic == stat]
        rows.append({"statistic": stat, "seeds": len(g), "mean_total": round(g.total.mean(), 1),
                     "best_total": int(g.total.max()),
                     "best_seeds": " ".join(str(s) for s in g.seed[g.total == g.total.max()]),
                     "cfi_min": int(g.cfi.min()), "cfi_max": int(g.cfi.max()),
                     "reliability_failures": int(g.reliability_failures.sum()),
                     "reliability_checks": 400 * len(g)})
    return pd.DataFrame(rows)


def table1_row(per_seed):
    g = per_seed[per_seed.statistic == "brec_code"]
    best = g.loc[g.total.idxmax()]
    row = {"seed": int(best.seed)}
    for name, size in TABLE1:
        row[name] = int(best[name])
        row[name + "_percent"] = round(100 * best[name] / size, 1)
    return pd.DataFrame([row])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-seed", default=str(HERE / "outputs" / "per_seed.csv"))
    ap.add_argument("--write", default=None, help="folder for table_R8.csv, table_R7.csv and table1_aggin_row.csv")
    args = ap.parse_args()
    per_seed = pd.read_csv(args.per_seed)
    r8, r7, t1 = table_r8(per_seed), table_r7(per_seed), table1_row(per_seed)

    pd.set_option("display.width", 200)
    print("Table R8: separated pairs, CFI pairs and failed reliability checks by seed")
    print(r8.to_string(index=False))
    print("\nTable R7: summary over seeds")
    print(r7.to_string(index=False))
    best = t1.iloc[0]
    print(f"\nTable 1, AG-GIN row (seed {int(best.seed)}, statistic of the BREC release code):")
    print("  " + "   ".join(f"{name} {int(best[name])}/{size} = {best[name + '_percent']:.1f}%" for name, size in TABLE1))

    if args.write:
        out = Path(args.write)
        out.mkdir(parents=True, exist_ok=True)
        r8.to_csv(out / "table_R8.csv", index=False)
        r7.to_csv(out / "table_R7.csv", index=False)
        t1.to_csv(out / "table1_aggin_row.csv", index=False)


if __name__ == "__main__":
    main()
