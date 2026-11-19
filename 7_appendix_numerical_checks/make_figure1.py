# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Figure 1: one ACT-4 reconstruction through Kron reduction, cycle augmentation and bridge contraction.

Read the SWC file, keep the largest component, Kron-reduce the degree-2 non-root vertices, add a
(tip, soma) edge of Euclidean length for every tip within eps of the soma, contract the bridges
shorter than tau, compute M_AG on the quotient, pull it back to the core vertices, and draw the matrix
and its spectrum. The script also prints the relative difference to the kernel with conductance 1e9
on the short bridges (Submitted AG).
Usage (from the package root):
  python 7_appendix_numerical_checks/make_figure1.py [SWC file] [eps] [tau] [output stem] [zoom]
Writes <output stem>.pdf and .png; the default stem is rerun/figure1 next to this script.
"""
import sys, os, numpy as np, networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, ConnectionPatch
from scipy.linalg import pinvh, eigh

RAW = os.environ.get("CLEARMIND_RAW", os.path.join("data", "raw"))
SWC = sys.argv[1] if len(sys.argv) > 1 else os.path.join(RAW, "allen_cell_type", "swc", "Scnn1a-Tg2-Cre_Ai14-173240.03.02.01_491770179_m.swc")
EPS = float(sys.argv[2]) if len(sys.argv) > 2 else 20.0
TAU = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
OUT = sys.argv[4] if len(sys.argv) > 4 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "rerun", "figure1")
ZOOM = float(sys.argv[5]) if len(sys.argv) > 5 else 2.6   # zoom half-width in units of eps

OI = dict(blue="#0072B2", orange="#E69F00", green="#009E73", verm="#D55E00",
          sky="#56B4E9", purple="#CC79A7", yellow="#F0E442", black="#000000")
CYC = [OI["blue"], OI["orange"], OI["purple"], OI["sky"], OI["yellow"]]

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 7, "axes.linewidth": 0.5, "mathtext.fontset": "dejavusans",
                     "pdf.fonttype": 42, "ps.fonttype": 42})

# ---------------------------------------------------------------- load
info, parent, types = {}, {}, {}
root = None
for line in open(SWC):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    p = line.split()
    if len(p) < 7:
        continue
    i = int(p[0]); info[i] = np.array(list(map(float, p[2:5]))); par = int(p[6]); types[i] = int(p[1])
    if par == -1:
        if root is None:
            root = i
        else:
            parent[i] = root
    else:
        parent[i] = par
G = nx.Graph()
G.add_nodes_from(info)
for c, p in parent.items():
    if c in info and p in info:
        G.add_edge(c, p, weight=max(np.linalg.norm(info[c] - info[p]), 1e-5))
cc = max(nx.connected_components(G), key=len)
G = G.subgraph(cc).copy()
nV = G.number_of_nodes()
full_edges = list(G.edges())

# ---------------------------------------------------------------- Kron reduction
T = G.copy()
chains = {}          # reduced edge -> list of original polyline vertices
for u, v in T.edges():
    chains[frozenset((u, v))] = [u, v]
deg2 = [n for n in T.nodes() if T.degree(n) == 2 and n != root]
for n in deg2:
    nb = list(T.neighbors(n))
    if len(nb) == 2:
        u, v = nb
        w = T[u][n]["weight"] + T[n][v]["weight"]
        cu = chains.pop(frozenset((u, n))); cv = chains.pop(frozenset((n, v)))
        if cu[0] != u: cu = cu[::-1]          # u ... n
        if cv[0] != n: cv = cv[::-1]          # n ... v
        poly = cu + cv[1:]
        if T.has_edge(u, v):
            T[u][v]["weight"] = min(T[u][v]["weight"], w)
        else:
            T.add_edge(u, v, weight=w); chains[frozenset((u, v))] = poly
        T.remove_node(n)
nC = T.number_of_nodes()
Tmin = T.copy()

# ---------------------------------------------------------------- augmentation (a tip already adjacent to the soma keeps one edge)
leaves = [n for n, d in T.degree() if d == 1 and n != root]
aug, overwritten = [], []
for lf in leaves:
    dist = np.linalg.norm(info[lf] - info[root])
    if dist < EPS:
        if T.has_edge(lf, root):
            overwritten.append(lf)
        else:
            aug.append(lf)
        T.add_edge(lf, root, weight=dist)
g = len(aug)
bridges = set(frozenset(e) for e in nx.bridges(T))
contract = [e for e in T.edges() if frozenset(e) in bridges and T[e[0]][e[1]]["weight"] < TAU]

# ---------------------------------------------------------------- Submitted AG and Exact quotient AG kernels, spectrum
nodes = list(T.nodes()); idx = {v: i for i, v in enumerate(nodes)}
W = np.zeros((len(nodes), len(nodes)))
for u, v, d in T.edges(data=True):
    isn = frozenset((u, v)) in bridges and d["weight"] < TAU
    W[idx[u], idx[v]] = W[idx[v], idx[u]] = 1e9 if isn else 1.0 / max(d["weight"], 1e-5)
L = np.diag(W.sum(1)) - W
Lp = pinvh(L); dg = np.diag(Lp); M_impl = dg[:, None] + dg[None, :] - 2 * Lp
Hc = nx.Graph(); Hc.add_nodes_from(nodes); Hc.add_edges_from(contract)
comp = {}
for k_, c_ in enumerate(nx.connected_components(Hc)):
    for v_ in c_: comp[v_] = k_
mC = max(comp.values()) + 1; Wc = np.zeros((mC, mC))
for u, v, d in T.edges(data=True):
    if frozenset((u, v)) in bridges and d["weight"] < TAU: continue
    a_, b_ = comp[u], comp[v]
    if a_ != b_:
        Wc[a_, b_] += 1.0 / max(d["weight"], 1e-5); Wc[b_, a_] += 1.0 / max(d["weight"], 1e-5)
Lc = np.diag(Wc.sum(1)) - Wc; Jc = np.ones((mC, mC)) / mC; Lcp = np.linalg.inv(Lc + Jc) - Jc
dc = np.diag(Lcp); Mc = dc[:, None] + dc[None, :] - 2 * Lcp
ci = np.array([comp[v_] for v_ in nodes]); M = Mc[np.ix_(ci, ci)]
print("Submitted AG vs Exact quotient AG, max relative difference:", float(np.abs(M_impl - M).max() / np.abs(M).max()))
print("contracted bridge lengths (um):", sorted(round(T[u][v]["weight"], 1) for u, v in contract), " largest M_AG entry (um):", round(float(M.max()), 1))
ev = eigh(M, eigvals_only=True)
absv = np.sort(np.abs(ev))[::-1]
print(f"|V|={nV} |V_core|={nC} g={g} overwritten={len(overwritten)} contracted_bridges={len(contract)} "
      f"lambda1={ev.max():.1f} n_pos={(ev>1e-6).sum()} top5={absv[:5].round(1)}")

# order vertices by DFS from root for the heatmap
order = [n for n in nx.dfs_preorder_nodes(Tmin, root)]
perm = [idx[v] for v in order]

# ---------------------------------------------------------------- drawing helpers
P = {v: info[v][:2] for v in info}
r0 = P[root]
allxy = np.array([P[v] for v in G.nodes()])

def draw_full(ax, color="0.55", lw=0.35):
    for u, v in full_edges:
        a, b = P[u], P[v]
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, solid_capstyle="round", zorder=1)

def draw_chain(ax, poly, **kw):
    xy = np.array([P[v] for v in poly]); ax.plot(xy[:, 0], xy[:, 1], **kw)

def tree_path_edges(a, b):
    path = nx.shortest_path(Tmin, a, b)
    return list(zip(path[:-1], path[1:]))

TITLES = []
def style(ax, title, lim=None):
    ax.set_aspect("equal"); ax.axis("off")
    if lim is not None:
        ax.set_xlim(lim[0], lim[1]); ax.set_ylim(lim[2], lim[3])
    TITLES.append((ax, title))

def scalebar(ax, length, label, where):
    x, y = where
    ax.plot([x, x + length], [y, y], color="k", lw=1.2, solid_capstyle="butt", zorder=10)
    ax.text(x + length / 2, y, label, ha="center", va="bottom", fontsize=6)

pad = 15
xmin, ymin = allxy.min(0) - pad; xmax, ymax = allxy.max(0) + pad
lim_full = (xmin, xmax, ymin, ymax)
h = ZOOM * EPS
lim_zoom = (r0[0] - h, r0[0] + h, r0[1] - h, r0[1] + h)

fig = plt.figure(figsize=(7.0, 1.72))
W_, H_, GAP, L0, B0 = 0.178, 0.80, 0.022, 0.0, 0.0
axA, axB, axC, axD, axE = [fig.add_axes([L0 + i * (W_ + GAP), B0, W_, H_]) for i in range(5)]
# (a) reconstruction
draw_full(axA)
axA.plot(*r0, marker="s", ms=3.5, color="k", zorder=5)
style(axA, "(a) Reconstruction", lim_full)
SUB = {}; SUB[id(axA)] = f"$|V|={nV:,}$ sample points"
scalebar(axA, 100, "100 µm", (xmax - 115, ymin + 5))

# (b) branching skeleton after Kron reduction
for e, poly in chains.items():
    draw_chain(axB, poly, color="0.6", lw=0.45, zorder=1)
bp = [v for v in Tmin if Tmin.degree(v) >= 3 and v != root]
tips = [v for v in Tmin if Tmin.degree(v) == 1 and v != root]
axB.scatter([P[v][0] for v in bp], [P[v][1] for v in bp], s=3.5, color="k", zorder=4, lw=0)
axB.scatter([P[v][0] for v in tips], [P[v][1] for v in tips], s=4, facecolor="white", edgecolor="k", lw=0.4, zorder=4)
axB.plot(*r0, marker="s", ms=3.5, color="k", zorder=5)
style(axB, "(b) Kron reduction", lim_full)
SUB[id(axB)] = rf"$|V_{{\mathrm{{core}}}}|={nC}$ core vertices"

# (c) augmentation, zoom on the soma
for e, poly in chains.items():
    draw_chain(axC, poly, color="0.7", lw=0.6, zorder=1)
for k, lf in enumerate(aug):
    col = CYC[k % len(CYC)]
    for a, b in tree_path_edges(root, lf):
        draw_chain(axC, chains[frozenset((a, b))], color=col, lw=1.3, zorder=3)
    axC.plot([P[lf][0], r0[0]], [P[lf][1], r0[1]], ls=(0, (2.2, 1.4)), color=col, lw=1.1, zorder=4)
    axC.plot(*P[lf], marker="o", ms=2.8, mfc="white", mec=col, mew=0.8, zorder=6)
axC.add_patch(Circle(r0, EPS, fill=False, ls="--", lw=0.8, ec=OI["green"], zorder=2))
axC.text(r0[0] - EPS * 0.75, r0[1] + EPS * 0.80, f"$\\epsilon={EPS:g}$ µm", color=OI["green"], fontsize=6, ha="right", va="bottom")
axC.plot(*r0, marker="s", ms=4, color="k", zorder=7)
style(axC, "(c) Cycle augmentation", lim_zoom)
SUB[id(axC)] = rf"$g=\beta_1={g}$ loops"
scalebar(axC, 10, "10 µm", (lim_zoom[1] - 14, lim_zoom[2] + 2))

# (d) quotient: short bridges, zoom on the soma
for e, poly in chains.items():
    draw_chain(axD, poly, color="0.7", lw=0.6, zorder=1)
for k, lf in enumerate(aug):
    col = CYC[k % len(CYC)]
    for a, b in tree_path_edges(root, lf):
        draw_chain(axD, chains[frozenset((a, b))], color=col, lw=1.3, zorder=3)
    axD.plot([P[lf][0], r0[0]], [P[lf][1], r0[1]], ls=(0, (2.2, 1.4)), color=col, lw=1.1, zorder=4)
for u, v in contract:
    key = frozenset((u, v))
    if key in chains:
        draw_chain(axD, chains[key], color=OI["verm"], lw=2.0, zorder=5)
    else:
        axD.plot([P[u][0], P[v][0]], [P[u][1], P[v][1]], color=OI["verm"], lw=2.0, zorder=5)
axD.plot(*r0, marker="s", ms=4, color="k", zorder=7)
style(axD, "(d) Quotient", lim_zoom)
SUB[id(axD)] = rf"{len(contract)} bridges $<\tau={TAU:g}$ µm contracted"

# (e) kernel and spectrum
Mo = M[np.ix_(perm, perm)]
im = axE.imshow(Mo, cmap="viridis", interpolation="nearest")
axE.set_xticks([]); axE.set_yticks([])
for s in axE.spines.values():
    s.set_linewidth(0.4)
TITLES.append((axE, "(e) AG matrix and spectrum"))
SUB[id(axE)] = "one positive eigenvalue"
ins = axE.inset_axes([0.50, 0.50, 0.47, 0.47])
kk = min(20, len(absv))
cols = [OI["verm"]] + [OI["blue"]] * (kk - 1)
ins.bar(np.arange(kk), absv[:kk], color=cols, width=0.8)
ins.set_yscale("log"); ins.tick_params(labelsize=5, length=1.5, width=0.4, pad=1)
ins.set_xticks([0, kk - 1]); ins.set_xticklabels(["1", str(kk)])
ins.set_facecolor("white")
for s in ins.spines.values():
    s.set_linewidth(0.4)
ins.text(0.96, 0.94, "$|\\lambda_i|$", transform=ins.transAxes, fontsize=5.5, ha="right", va="top")

for ax_, t_ in TITLES:
    x0 = ax_.get_position().x0
    fig.text(x0 + 0.004, 0.965, t_, fontsize=7.2, fontweight="bold", ha="left", va="bottom")
    if id(ax_) in SUB:
        col_ = OI["verm"] if ax_ is axD else "k"
        fig.text(x0 + 0.004, 0.955, SUB[id(ax_)], fontsize=6.3, ha="left", va="top", color=col_)
os.makedirs(os.path.dirname(os.path.abspath(OUT)), exist_ok=True)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.02, metadata={"CreationDate": None})
fig.savefig(OUT + ".png", dpi=220, bbox_inches="tight", pad_inches=0.02)
print("saved", OUT)
