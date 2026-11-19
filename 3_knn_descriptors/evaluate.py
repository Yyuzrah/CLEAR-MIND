# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Cosine k-NN rows of Tables 3, 4, R3, R4 and R5: fold-7 selection first, then one scoring of folds 8-9.

    python 3_knn_descriptors/evaluate.py select   # folds 0-6 are the references, fold 7 picks k and scaling
    python 3_knn_descriptors/evaluate.py test     # scores folds 8-9 once with the stored selection

`select` uses folds 0-7 only and writes <out>/selection.json. `test` needs that file
and only uses the choices stored there. Results go to 3_knn_descriptors/rerun/ by default; the results
of the paper are in outputs/. Options: --features FILE (default features.npz in this folder), --out DIR.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import argparse
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from clearmind.paths import CLASSES
import knn

DATASETS = list(CLASSES)


def select(table, arrays):
    choices, nominate = {}, {}
    for ds in DATASETS:
        sub, data = knn.dataset_view(table, arrays, ds)
        seen = (sub.split != "test").to_numpy()   # folds 0-7
        sub, data = sub[seen].reset_index(drop=True), {key: value[seen] for key, value in data.items()}
        train = (sub.split == "train").to_numpy()
        valid = (sub.split == "valid").to_numpy()
        y = sub.label.to_numpy()
        for method, parts in knn.METHODS.items():
            choices[ds + "|" + method] = knn.choose(data, parts, train, valid, y, len(CLASSES[ds]))
        # At most one threshold setting per dataset; ties go to the default (20, 5), then to grid order.
        grid = [name for name in knn.METHODS if name.startswith("eps=")]
        nominate[ds] = min(grid, key=lambda name: (-choices[ds + "|" + name]["correct"], name != "eps=20,tau=5",
                                                    grid.index(name)))
    return dict(choices=choices, grid_by_validation=nominate)


def score(table, arrays, selection, out):
    choices = selection["choices"]
    rows = []
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    for ds in DATASETS:
        sub, data = knn.dataset_view(table, arrays, ds)
        train = (sub.split == "train").to_numpy()
        test = (sub.split == "test").to_numpy()
        y = sub.label.to_numpy()
        classes = len(CLASSES[ds])
        columns = {}
        for method, parts in knn.METHODS.items():
            chosen = choices[ds + "|" + method]
            x = knn.embedding(data, parts, train, chosen["scale"])
            columns[method] = knn.vote(knn.ranks(x, test, train), y[train], chosen["k"], classes)
        preds = pd.concat([sub.loc[test, ["name", "fold", "label"]].reset_index(drop=True), pd.DataFrame(columns)], axis=1)
        preds.to_csv(out / "predictions" / (ds + ".csv"), index=False)
        for method in knn.METHODS:
            chosen = choices[ds + "|" + method]
            rows.append(dict(dataset=ds, method=method, k=chosen["k"], scale=chosen["scale"],
                             val_accuracy=100 * chosen["correct"] / (sub.split == "valid").sum(), test_n=len(preds),
                             accuracy=100 * (preds[method] == preds.label).mean(),
                             macro_f1=100 * f1_score(preds.label, preds[method], average="macro",
                                                     labels=range(classes), zero_division=0)))
    return pd.DataFrame(rows)


def paper_tables(metrics, table, arrays, out):
    m = metrics.set_index(["dataset", "method"])

    def cells(method, columns=("accuracy", "macro_f1")):
        return {f"{ds} {col}": round(m.loc[(ds, method), col], 2) for ds in DATASETS for col in columns}

    # Table 3: lattice pipeline versus Exact quotient AG at (20, 5); time = feature computation after Kron
    # reduction, summed over the neurons of the dataset.
    rows = []
    for ds in DATASETS:
        mask = (table.dataset == ds).to_numpy()
        rows.append({"dataset": ds,
                     "lattice accuracy": round(m.loc[(ds, "Lattice pipeline"), "accuracy"], 2),
                     "lattice macro_f1": round(m.loc[(ds, "Lattice pipeline"), "macro_f1"], 2),
                     "lattice time_s": round(arrays["lattice_s"][mask].sum(), 1),
                     "exact quotient AG accuracy": round(m.loc[(ds, "Exact quotient AG"), "accuracy"], 2),
                     "exact quotient AG macro_f1": round(m.loc[(ds, "Exact quotient AG"), "macro_f1"], 2),
                     "exact quotient AG time_s": round(arrays["exact_s"][mask].sum(), 1)})
    pd.DataFrame(rows).to_csv(out / "table3_lattice_vs_exact_ag.csv", index=False)
    # Table 4: augmentation and contraction ablation.
    pd.DataFrame([dict(eps=e, tau=t, **cells(f"eps={e},tau={t}")) for e, t in [(0, 0), (50, 0), (0, 10), (50, 10)]]
                 ).to_csv(out / "table4_ablation.csv", index=False)
    # Table R3: standalone and combined descriptors.
    names = ["Submitted AG", "Exact quotient AG", "Laplacian high", "Laplacian low", "Sholl", "Radial TMD",
             "Full-tree H0", "Five statistics", "Classical combination", "Classical + Exact quotient AG"]
    pd.DataFrame([dict(descriptor=name, **cells(name)) for name in names]
                 ).to_csv(out / "tableR3_descriptors.csv", index=False)
    # Table R4: threshold grid; Table R5: readout length K (accuracy only).
    pd.DataFrame([dict(eps=e, tau=t, **cells(f"eps={e},tau={t}", ("accuracy",))) for e in knn.EPS for t in knn.TAU]
                 ).to_csv(out / "tableR4_threshold_grid.csv", index=False)
    pd.DataFrame([dict(K=k, **cells(f"K={k}", ("accuracy",))) for k in (16, 32, 64, 128)]
                 ).to_csv(out / "tableR5_readout_length.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["select", "test"])
    parser.add_argument("--features", default=str(knn.FEATURES))
    parser.add_argument("--out", default=str(HERE / "rerun"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stored = out / "selection.json"
    table, arrays = knn.load_features(args.features)
    if args.step == "select":
        selection = select(table, arrays)
        if stored.exists() and json.loads(stored.read_text()) != selection:
            raise SystemExit(f"{stored} holds different choices; remove it to select again.")
        stored.write_text(json.dumps(selection, indent=2))
        print("fold-7 selection written to", stored)
        return
    if not stored.exists():
        raise SystemExit(f"{stored} not found: run `evaluate.py select` before scoring the test folds.")
    selection = json.loads(stored.read_text())
    assert set(selection["choices"]) == {ds + "|" + m for ds in DATASETS for m in knn.METHODS}
    metrics = score(table, arrays, selection, out)
    metrics.to_csv(out / "all_results.csv", index=False)
    paper_tables(metrics, table, arrays, out)
    pd.set_option("display.width", 200)
    print(metrics.pivot_table(index="method", columns="dataset", values="accuracy", sort=False)[DATASETS].round(2))
    reference = HERE / "outputs" / "all_results.csv"
    if reference.exists() and reference.resolve() != (out / "all_results.csv").resolve():
        ref = pd.read_csv(reference, float_precision="round_trip")
        same = ref[["dataset", "method", "k", "scale"]].equals(metrics[["dataset", "method", "k", "scale"]])
        gap = np.abs(ref[["accuracy", "macro_f1"]].to_numpy() - metrics[["accuracy", "macro_f1"]].to_numpy()).max()
        print(f"compared with outputs/all_results.csv: same selections = {same}, largest metric difference = {gap:.2e}")


if __name__ == "__main__":
    main()
