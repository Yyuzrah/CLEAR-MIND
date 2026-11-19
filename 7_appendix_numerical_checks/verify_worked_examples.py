# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Two worked examples: a two-chord instance of the loop interaction of Remark A.22 and Cor. A.28,
and the theta-graph example in the proof of Prop. A.35(iii) (Babai rounding against the closest lift).
Usage (from the package root):  python 7_appendix_numerical_checks/verify_worked_examples.py
"""
import numpy as np, itertools
def res(n, E):
    W = np.zeros((n, n))
    for u, v, l in E: W[u, v] += 1 / l; W[v, u] += 1 / l
    L = np.diag(W.sum(1)) - W; Lp = np.linalg.pinv(L); d = np.diag(Lp)
    return d[:, None] + d[None, :] - 2 * Lp
# interaction example: rho=0, w=1, v1=2, v2=3
tree = [(0, 1, 1.0), (1, 2, 1.0), (1, 3, 1.0)]
M1 = res(4, tree + [(2, 0, 1.0)]); M2 = res(4, tree + [(2, 0, 1.0), (3, 0, 1.0)])
print("M(w,v1) one chord:", M1[1, 2], " two chords:", M2[1, 2], "(expect 2/3 -> 5/8 =", 5 / 8, ")")
# theta example: a=0, b=1, x=2 on e1 (a->x 0.9, x->b 0.1), e2 a->b 1, e3 a->b 1
E = [(0, 2, 0.9), (2, 1, 0.1), (0, 1, 1.0), (0, 1, 1.0)]
M = res(3, E); print("M(x,a) =", M[2, 0], "(expect 0.36); d(x,a)=0.9 -> dTrop^2 =", 0.9 - M[2, 0])
# edges indexed: e1'=0 (a->x), e1''=1 (x->b), e2=2 (a->b), e3=3 (a->b); lengths
ell = np.array([0.9, 0.1, 1.0, 1.0]); D = np.diag(ell)
z1 = np.array([-1, -1, 1, 0.]); z2 = np.array([-1, -1, 0, 1.])
C = np.array([z1, z2]); Q = C @ D @ C.T; print("Q =", Q.tolist())
p1 = np.array([-1, 0, 0, 0.])             # x -> a along e1' reversed
t = C @ D @ p1; a = np.linalg.solve(Q, t); nh = np.round(a); dl = a - nh
print("T1: t", t, "alpha", a, "nhat", nh, "Babai^2", dl @ Q @ dl)
C2 = np.array([-z1, z2 - z1]); Q2 = C2 @ D @ C2.T; print("Q' =", Q2.tolist())
p2 = np.array([0, 1, -1, 0.])             # x -> b (e1''), b -> a (e2 reversed)
t2 = C2 @ D @ p2; a2 = np.linalg.solve(Q2, t2); nh2 = np.round(a2); dl2 = a2 - nh2
print("T2: t'", t2, "alpha'", a2, "nhat'", nh2, "Babai^2", dl2 @ Q2 @ dl2)
best = min(((t - Q @ np.array(n)) @ np.linalg.solve(Q, t - Q @ np.array(n))) for n in itertools.product(range(-3, 4), repeat=2))
print("brute-force dTrop^2 =", best)
