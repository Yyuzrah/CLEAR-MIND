# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Prop. A.36: a lift is closest exactly when it satisfies the half-circuit condition, checked against
brute force on random small graphs.
Usage (from the package root):  python 7_appendix_numerical_checks/verify_half_circuit_criterion.py
"""
import numpy as np, networkx as nx, itertools
random_tree = nx.random_labeled_tree if hasattr(nx, 'random_labeled_tree') else nx.random_tree
rng=np.random.default_rng(3); bad=0; tot=0; agree=0
for trial in range(120):
    n=int(rng.integers(5,9)); T=random_tree(n,seed=int(rng.integers(1e9))); G=nx.Graph(T)
    cand=[(u,v) for u in range(n) for v in range(u+1,n) if not G.has_edge(u,v)]
    for k in rng.choice(len(cand),size=int(rng.integers(1,4)),replace=False): G.add_edge(*cand[k])
    E=list(G.edges); m=len(E); ell=np.array([float(rng.uniform(0.2,3)) for _ in E])
    eid={}; 
    for i,(u,v) in enumerate(E): eid[(u,v)]=(i,1.); eid[(v,u)]=(i,-1.)
    def chain(path, closed=False):
        c=np.zeros(m); P=list(path)+([path[0]] if closed else [])
        for a,b in zip(P[:-1],P[1:]): i,sg=eid[(a,b)]; c[i]+=sg
        return c
    D=np.diag(ell)
    # all oriented circuits
    DG=G.to_directed(); circuits=[chain(cyc,closed=True) for cyc in nx.simple_cycles(DG) if len(cyc)>2]
    Tb=nx.bfs_tree(G,0).to_undirected(); non=[e for e in E if not Tb.has_edge(*e)]
    C=np.array([chain(nx.shortest_path(Tb,e[1],e[0]))+chain([e[0],e[1]]) for e in non]); Q=C@D@C.T; Qi=np.linalg.inv(Q); g=len(non)
    W=np.zeros((n,n))
    for (u,v),l in zip(E,ell): W[u,v]+=1/l; W[v,u]+=1/l
    L=np.diag(W.sum(1))-W; Lp=np.linalg.pinv(L); R=np.diag(Lp)[:,None]+np.diag(Lp)[None,:]-2*Lp
    for x in range(n):
        for y in range(n):
            if x==y: continue
            for path in itertools.islice(nx.all_simple_paths(G,x,y),6):
                p=chain(path); t=C@D@p
                best=min((t-Q@np.array(z))@Qi@(t-Q@np.array(z)) for z in itertools.product(range(-3,4),repeat=g))
                closest=abs(t@Qi@t-best)<1e-9
                crit=all(2*(p@D@c)<=c@D@c+1e-9 for c in circuits)
                tot+=1; agree+=(closest==crit)
print(f"criterion agrees with brute-force closest-lift status on {agree}/{tot} (path,pair) instances")
