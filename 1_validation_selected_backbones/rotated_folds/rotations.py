# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Fold rotations of App. C.7 (Table R9).

Table 5 trains on folds 0-6, selects on fold 7 and tests on folds 8-9. Each rotation below moves the
test pair to folds that were training folds in the original study, keeps one other fold for checkpoint
selection, and trains on the remaining seven folds. Together with the Table 5 split, every neuron is
tested exactly once.
"""
ROTATIONS = {
    "R1": dict(test=(0, 1), valid=2),
    "R2": dict(test=(2, 3), valid=4),
    "R3": dict(test=(4, 5), valid=6),
    "R4": dict(test=(6, 7), valid=8),
}


def split_of(fold, rotation):
    r = ROTATIONS[rotation]
    if fold in r["test"]:
        return "test"
    if fold == r["valid"]:
        return "valid"
    return "train"


def with_rotation(df, rotation):
    """Copy of the split table with the split column set by the rotation."""
    out = df.copy()
    out["split"] = [split_of(int(f), rotation) for f in out.fold]
    return out
