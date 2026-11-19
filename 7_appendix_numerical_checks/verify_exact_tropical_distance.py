# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Theorem A.30(ii), d_Trop^2 = d_Gamma - M_AG, against a brute-force closest-vector search on random
metric graphs; Theorem A.26 on the spanning-tree lift of every pair; and how often Babai rounding of
that lift is not the closest lattice point (Remark A.34).
Usage (from the package root):  python 7_appendix_numerical_checks/verify_exact_tropical_distance.py
"""
import numpy as np, networkx as nx, itertools
rng=np.random.default_rng(0)
def make_graph(n,extra):
    while True:
        T=nx.random_labeled_tree(n,seed=int(rng.integers(1e9))) if hasattr(nx,'random_labeled_tree') else nx.random_tree(n,seed=int(rng.integers(1e9)))
        G=nx.Graph(T)
        cand=[(u,v) for u in range(n) for v in range(u+1,n) if not G.has_edge(u,v)]
        idx=rng.choice(len(cand),size=extra,replace=False)
        for k in idx: G.add_edge(*cand[k])
        for u,v in G.edges: G[u][v]['w']=float(rng.uniform(0.2,3.0))
        return G
worst=0; babai_bad=0; tot=0; worst_lift=0
for trial in range(300):
    n=int(rng.integers(5,11)); g=int(rng.integers(1,5))
    G=make_graph(n,g)
    E=list(G.edges); m=len(E); ell=np.array([G[u][v]['w'] for u,v in E])
    eidx={e:i for i,e in enumerate(E)}; eidx.update({(v,u):i for i,(u,v) in enumerate(E)})
    def chain(path):
        c=np.zeros(m)
        for a,b in zip(path[:-1],path[1:]):
            i=eidx[(a,b)]; c[i]+= 1.0 if E[i]==(a,b) else -1.0
        return c
    # spanning tree (BFS) and fundamental cycles
    T=nx.bfs_tree(G,0).to_undirected()
    non=[e for e in E if not T.has_edge(*e)]
    C=np.array([chain(nx.shortest_path(T,e[1],e[0]))+chain([e[0],e[1]]) for e in non])
    D=np.diag(ell); Q=C@D@C.T; Qi=np.linalg.inv(Q)
    # Laplacian pinv -> resistance
    W=np.zeros((n,n))
    for (u,v),l in zip(E,ell): W[u,v]+=1/l; W[v,u]+=1/l
    L=np.diag(W.sum(1))-W; Lp=np.linalg.pinv(L)
    R=np.diag(Lp)[:,None]+np.diag(Lp)[None,:]-2*Lp
    dG=nx.floyd_warshall_numpy(nx.Graph([(u,v,{'weight':G[u][v]['w']}) for u,v in E]),nodelist=range(n))
    # AJ period coords via tree paths from root 0
    V=np.array([C@D@chain(nx.shortest_path(T,0,v)) if v!=0 else np.zeros(g) for v in range(n)])
    rngbox=range(-4,5)
    for x in range(n):
        for y in range(x+1,n):
            t=V[y]-V[x]
            best=min(((t-Q@np.array(k))@Qi@(t-Q@np.array(k))) for k in itertools.product(rngbox,repeat=g))
            pred=dG[x,y]-R[x,y]
            worst=max(worst,abs(best-pred))
            a=Qi@t; delta=a-np.round(a); babai=delta@Q@delta
            tot+=1; babai_bad+= (babai-best)>1e-9
            # lift identity for tree path p (y<-x through tree) : ell(p) = R + ||pi p||^2
            p=chain(nx.shortest_path(T,x,y)); lp=p@D@p; tt=C@D@p
            worst_lift=max(worst_lift,abs(lp-(R[x,y]+tt@Qi@tt)))
print("max |bruteforce CVP - (d_Gamma - r)| =",worst)
print("max |ell(p) - (r + t^T Q^-1 t)| over tree paths =",worst_lift)
print(f"Babai rounding suboptimal on {babai_bad}/{tot} pairs")
