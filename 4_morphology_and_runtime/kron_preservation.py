# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""What Kron reduction keeps on the training neurons (Appendix C.4).

For every training reconstruction (folds 0-6) the raw SWC graph is compared with its Kron-reduced
skeleton: branch-vertex IDs (degree >= 3), tip IDs (degree 1) and total cable length. The summary gives,
per dataset, the number of mismatches, the largest relative change of cable length, the share of raw
samples removed in the 25% densest neurons (branch vertices per mm of cable), and the Spearman
correlation of the leading Exact quotient AG eigenvalue with total cable length and with core size.
The bend example compares two soma-rooted trees with the same branch lengths and different bends
inside a branch, and a third tree with a 1 um side branch at the bend (spectra at eps = tau = 0).

    python 4_morphology_and_runtime/kron_preservation.py [--features FILE] [--out DIR] [--workers 3]

Reads the SWC files (see data/README.md) and the leading eigenvalues in 3_knn_descriptors/features.npz;
about 5 minutes. Results go to 4_morphology_and_runtime/rerun/ by default; the paper's are in outputs/.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
import sys
import json
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "3_knn_descriptors")]

import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import spearmanr
from clearmind.paths import swc_path
import descriptors
import knn


def topology(job):
    ds, name, leading, cable_ref, core_ref = job
    raw, coords, root = descriptors.parse_swc(swc_path(ds, name))
    tree = descriptors.Extractor(raw, coords, root).skeleton_tree

    def partition(graph):
        return {v for v, n in graph.degree() if n >= 3}, {v for v, n in graph.degree() if n == 1}

    branch, tips = partition(raw)
    new_branch, new_tips = partition(tree)
    cable = sum(w for u, v, w in raw.edges(data="weight"))
    new_cable = sum(w for u, v, w in tree.edges(data="weight"))
    assert np.isclose(cable_ref, cable, rtol=1e-7) and len(tree) == core_ref
    return dict(dataset=ds, name=name, n_raw=len(raw), n_core=len(tree), branch_n=len(branch), tip_n=len(tips),
                changed_branch_ids=len(branch ^ new_branch), changed_tip_ids=len(tips ^ new_tips),
                cable_um=cable, cable_relative_error=abs(cable - new_cable) / cable, density=1000 * len(branch) / cable,
                vertices_removed=1 - len(tree) / len(raw), leading=leading)


def bend_example(mid, extra=False):
    """Soma 0, a bend at `mid` on the way to the branch point 2, two tips; `extra` adds a side branch at the bend."""
    coords = {0: np.zeros(3), 1: np.array(mid), 2: np.array([2., 0., 0.]), 3: np.array([3., 1., 0.]),
              4: np.array([3., 0., 2.])}
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    if extra:
        coords[5] = np.array([1., 2., 0.])
        edges.append((1, 5))
    g = nx.Graph()
    for u, v in edges:
        g.add_edge(u, v, weight=np.linalg.norm(coords[u] - coords[v]))
    tree = descriptors.Extractor(g, coords, 0).skeleton_tree
    val, _ = descriptors.quotient(tree, list(tree), [], False)
    return dict(core=len(tree), branch_points=sum(deg >= 3 for _, deg in tree.degree()), spectrum=val["spec"].tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=str(knn.FEATURES))
    parser.add_argument("--out", default=str(HERE / "rerun"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    descriptors.load_submission_code()
    examples = [bend_example([1., 1., 0.]), bend_example([1., 0., 1.]), bend_example([1., 1., 0.], extra=True)]
    assert np.allclose(examples[0]["spectrum"], examples[1]["spectrum"])
    (out / "bend_example.json").write_text(json.dumps(examples, indent=2))

    splits, arrays = knn.load_features(args.features)
    train = (splits.split == "train").to_numpy()
    jobs = list(zip(splits.dataset[train], splits.name[train], arrays[knn.DEFAULT][train, 0], arrays["stats"][train, 2],
                    arrays["n_core"][train]))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=descriptors.load_submission_code) as pool:
        table = pd.DataFrame(pool.map(topology, jobs, chunksize=3))
    table.to_csv(out / "kron_training_neurons.csv", index=False)
    rows = []
    for ds, data in table.groupby("dataset", sort=False):
        cut = data.density.quantile(.75)
        dense = data[data.density >= cut]
        rows.append(dict(dataset=ds, training_n=len(data), high_density_n=len(dense), cut_branches_per_mm=cut,
                         branch_id_mismatches=int(data.changed_branch_ids.sum()),
                         tip_id_mismatches=int(data.changed_tip_ids.sum()),
                         max_cable_relative_error=data.cable_relative_error.max(),
                         dense_median_removed=dense.vertices_removed.median(),
                         dense_median_branches=dense.branch_n.median(),
                         rho_leading_cable=spearmanr(data.leading, data.cable_um).statistic,
                         rho_leading_core=spearmanr(data.leading, data.n_core).statistic))
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "kron_summary.csv", index=False)
    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print("leading eigenvalue of the bend example:", [round(e["spectrum"][0], 2) for e in examples])


if __name__ == "__main__":
    main()
