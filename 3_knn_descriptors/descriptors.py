# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Per-neuron descriptors behind the k-NN comparisons (Tables 3, 4 and R3-R5) and the runtime table (R6).

Everything here is computed from one SWC reconstruction; no labels are used.
  neuron_features        Submitted AG, Exact quotient AG on the threshold grid eps x tau, the weighted
                         Laplacian spectrum of the default quotient, Sholl curve, radial TMD histogram,
                         five morphometric statistics, sizes, chord and bridge counts, stage timings
  persistence_histogram  full-tree radial H0 persistence histogram
  lattice_features       the explicit lattice (Babai rounding) baseline of Table 3 and Exact quotient AG,
                         each with its computation time after Kron reduction
"""
import time
import numpy as np
import networkx as nx
from scipy.linalg import eigh, solve

EPS = (0, 20, 50)        # augmentation thresholds epsilon (um)
TAU = (0, 5, 10)         # bridge-contraction thresholds tau (um)
RADII = np.geomspace(5, 20000, 64)                         # Sholl radii (um)
BAR_EDGES = np.r_[0., np.geomspace(5, 10000, 7), np.inf]   # 8 x 8 (birth, death) bins (um); the last bin takes overflow


def load_submission_code():
    """Import the submission's SWC parser and extractors once per process (keeps imports out of the timings)."""
    global parse_swc, Extractor, LatticeExtractor, exact_signature
    from clearmind.submission_treelstm import MorphologyDataset, SpectralTopologyExtractor
    from clearmind.lattice_baseline import PureTropicalJacobianExtractor
    from clearmind.exact_ag import exact_signature
    parse_swc = lambda path: MorphologyDataset._parse_swc(None, str(path))
    Extractor = SpectralTopologyExtractor
    LatticeExtractor = PureTropicalJacobianExtractor


def morphology(raw, coords, root, tree):
    """Sholl curve on the full polyline, radial TMD histogram of the skeleton, five statistics."""
    center = coords[root]
    ends = np.array([(coords[u] - center, coords[v] - center) for u, v in raw.edges()])
    direction = ends[:, 1] - ends[:, 0]
    sq = (direction ** 2).sum(1)
    t = np.clip(-(ends[:, 0] * direction).sum(1) / np.maximum(sq, 1e-30), 0, 1)
    low = np.linalg.norm(ends[:, 0] + t[:, None] * direction, axis=1)
    high = np.linalg.norm(ends, axis=2)
    # Each straight segment has two monotone radial pieces. The half-open
    # convention [lower, upper) makes an ordinary through-vertex count once.
    sholl = np.zeros(len(RADII), dtype=int)
    for col in (0, 1):
        start = np.searchsorted(RADII, low)
        stop = np.searchsorted(RADII, high[:, col])
        nz = (stop > start) & (high[:, col] - low > 1e-9)
        sholl += np.cumsum(np.bincount(start[nz], minlength=65) - np.bincount(stop[nz], minlength=65))[:-1]
    # Radial TMD: elder rule on the rooted skeleton, one bar per tip.
    rooted = nx.bfs_tree(tree, root)
    radial = {v: np.linalg.norm(coords[v] - center) for v in tree}
    surviving = {}
    bars = []
    for v in reversed(list(rooted)):
        children = sorted((surviving[c] for c in rooted.successors(v)), reverse=True)
        if children:
            bars.extend((birth, radial[v]) for birth in children[1:])
            surviving[v] = children[0]
        else:
            surviving[v] = radial[v]
    bars.append((surviving[root], 0.))
    hist = np.histogram2d(*np.asarray(bars).T, bins=(BAR_EDGES, BAR_EDGES))[0]
    tips = sum(tree.degree(v) == 1 and v != root for v in tree)
    assert len(bars) == tips and hist.sum() == tips
    path = nx.single_source_dijkstra_path_length(tree, root, weight="weight")
    # tips, branch points, total cable, maximal radial reach, longest root-to-tip path
    stats = [tips, sum(n >= 3 for _, n in tree.degree()), np.sqrt(sq).sum(), high.max(), max(path.values())]
    return dict(sholl=sholl, tmd=hist.ravel(), stats=np.array(stats))


def quotient(graph, nodes, short, with_laplacian):
    """Exact AG spectrum of `graph` after contracting the edges in `short`, pulled back to `nodes`.

    Returns a dict with the sorted eigenvalue magnitudes (full length), the quotient order m and, if
    asked, the nonconstant weighted Laplacian eigenvalues of the quotient; and the eigendecomposition time.
    """
    group = nx.utils.UnionFind(nodes)
    for u, v in short:
        group.union(u, v)
    labels = {}
    index = np.array([labels.setdefault(group[v], len(labels)) for v in nodes])
    component = dict(zip(nodes, index))
    m = len(labels)
    W = np.zeros((m, m))
    for u, v, data in graph.edges(data=True):
        i, j = component[u], component[v]
        if i != j:
            c = 1 / max(data["weight"], 1e-5)
            W[i, j] += c
            W[j, i] += c
    L = np.diag(W.sum(1)) - W
    J = np.full_like(L, 1 / m)
    inv = solve(L + J, np.eye(m), assume_a="pos") - J
    M = inv.diagonal()[:, None] + inv.diagonal()[None, :] - 2 * inv
    counts = np.sqrt(np.bincount(index))
    tic = time.perf_counter()
    vals = np.sort(np.abs(eigh(M * counts[:, None] * counts[None, :], eigvals_only=True)))[::-1]
    spectrum_s = time.perf_counter() - tic
    out = {"spec": vals, "m": m, "lap_s": 0.}
    if with_laplacian:
        start_lap = time.perf_counter()
        freq = np.maximum(0, eigh(L, eigvals_only=True)[1:])
        out.update(lap_low=freq, lap_high=freq[::-1], lap_s=time.perf_counter() - start_lap)
    return out, spectrum_s


def neuron_features(path):
    """All spectral and morphological descriptors of one SWC, with stage timings in seconds."""
    start = time.perf_counter()
    raw, coords, root = parse_swc(path)
    t1 = time.perf_counter()
    ext = Extractor(raw, coords, root)      # Kron reduction
    tree = ext.skeleton_tree
    assert nx.is_tree(tree)
    t2 = time.perf_counter()
    nodes = ext.node_list
    output = morphology(raw, coords, root, tree)
    output.update(parse_s=t1 - start, kron_s=t2 - t1, n_raw=len(raw), n_core=len(tree))
    for eps in EPS:
        init = time.perf_counter()
        g = tree.copy()
        attempts = 0
        for v, n in tree.degree():
            if n == 1 and v != root:
                length = np.linalg.norm(coords[v] - coords[root])
                if length < eps:
                    attempts += 1
                    g.add_edge(v, root, weight=length)
        bridges = [(u, v, g[u][v]["weight"]) for u, v in nx.bridges(g)]
        elapsed = time.perf_counter() - init
        # A new chord creates a cycle; a tip already adjacent to the root only has its edge length updated.
        output[f"chords{eps}"] = g.number_of_edges() - tree.number_of_edges()
        output[f"overwrite{eps}"] = attempts - output[f"chords{eps}"]
        for tau in TAU:
            default = eps == 20 and tau == 5
            short = [(u, v) for u, v, w in bridges if w < tau]
            start = time.perf_counter()
            values, spectrum_s = quotient(g, nodes, short, default)
            total = time.perf_counter() - start - values["lap_s"]
            output[f"exact_ag_{eps}_{tau}"] = values["spec"]
            output[f"short{eps}_{tau}"] = len(short)
            if default:
                output.update(augmentation_s=elapsed, solve_s=total - spectrum_s, spectrum_s=spectrum_s, m=values["m"])
                output.update(lap_low=values["lap_low"], lap_high=values["lap_high"])
    # Submitted AG at the default eps=20, tau=5, in single precision as in the model inputs.
    submitted = ext.get_spectral_signature()   # augments ext.G with the chords; `tree` is a separate copy
    output["submitted_ag"] = submitted.astype(np.float32)
    # The default exact spectrum is the one clearmind.exact_ag gives to the backbones.
    padded = np.zeros(64)
    val = output["exact_ag_20_5"][:64]
    padded[:len(val)] = val
    assert np.linalg.norm(padded - exact_signature(ext)) <= 1e-9 * max(1, np.linalg.norm(padded)), path
    return output


def full_barcode(g, coords, root):
    """0-dimensional superlevel-set persistence of soma distance on the complete SWC polyline.

    Degree-two vertices are kept and each segment's interior radial minimum is inserted, so bends
    inside a branch contribute bars. Components merge by the elder rule; zero-length bars are dropped
    and the essential bar is closed at radius zero.
    """
    nodes = list(g)
    ids = {v: i for i, v in enumerate(nodes)}
    points = np.array([coords[v] - coords[root] for v in nodes])
    radii = list(np.linalg.norm(points, axis=1))
    adj = [[] for _ in nodes]
    for u, v in g.edges:
        i, j = ids[u], ids[v]
        delta = points[j] - points[i]
        norm = delta @ delta
        t = -float(points[i] @ delta) / norm if norm else 0.
        if 0 < t < 1:
            k = len(radii)
            radii.append(float(np.linalg.norm(points[i] + t * delta)))
            adj.append([i, j])
            adj[i].append(k)
            adj[j].append(k)
        else:
            adj[i].append(j)
            adj[j].append(i)
    rad = np.array(radii)
    parent = np.full(len(rad), -1)
    births = rad.copy()

    def find(v):
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    bars = []
    for v in np.argsort(-rad, kind="stable"):
        neighbors = {find(w) for w in adj[v] if parent[w] >= 0}
        if not neighbors:
            parent[v] = v
            continue
        elder = max(neighbors, key=lambda w: (births[w], -w))
        parent[v] = elder
        for w in neighbors:
            if w != elder:
                if births[w] > rad[v]:
                    bars.append((births[w], rad[v]))
                parent[w] = elder
    bars.append((births[find(ids[root])], 0.))
    return np.asarray(bars)


def persistence_histogram(path):
    """Full-tree H0 barcode of one SWC as an 8 x 8 count histogram (64 numbers), plus the bars."""
    g, coords, root = parse_swc(path)
    bars = full_barcode(g, coords, root)
    counts = np.histogram2d(*bars.T, bins=(BAR_EDGES, BAR_EDGES))[0].ravel()
    assert int(counts.sum()) == len(bars) and (bars[:, 0] >= bars[:, 1]).all()
    return dict(hist=counts, bars=bars)


def exact_timed(tree, nodes, coords, root, eps, tau):
    """Exact quotient AG at (eps, tau) from the Kron-reduced tree, with its computation time."""
    start = time.perf_counter()
    g = tree.copy()
    for v, n in tree.degree():
        if n == 1 and v != root:
            length = np.linalg.norm(coords[v] - coords[root])
            if length < eps:
                g.add_edge(v, root, weight=length)
    short = [(u, v) for u, v in nx.bridges(g) if g[u][v]["weight"] < tau]
    values, _ = quotient(g, nodes, short, False)
    elapsed = time.perf_counter() - start
    spec = np.zeros(64)
    spec[:min(64, len(values["spec"]))] = values["spec"][:64]
    return spec, elapsed


def lattice_features(path, eps=20, tau=5):
    """Lattice-pipeline spectrum and Exact quotient AG at (eps, tau), each timed after Kron reduction."""
    graph, coords, root = parse_swc(path)
    ext = LatticeExtractor(graph, coords, root)
    start = time.perf_counter()
    lattice = ext.get_spectral_signature(eps, tau)
    lattice_s = time.perf_counter() - start
    base = Extractor(graph, coords, root)
    exact, exact_s = exact_timed(base.skeleton_tree, base.node_list, coords, root, eps, tau)
    for value in (lattice, exact):
        assert np.isfinite(value).all(), path
    return dict(lattice=lattice, lattice_s=lattice_s, exact=exact, exact_s=exact_s)
