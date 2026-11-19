# CLEARMIND: code accompanying the CLEARMIND paper.
# Copyright (c) 2026 the CLEARMIND authors. All rights reserved.
# You may read this code and run it to check the reported results. Any other use, copying,
# rewriting or redistribution of this code or of code derived from it, in whole or in part,
# requires the authors' written permission. See LICENSE.
"""Largest errors of closed forms on random trees with chords.

Qformula: period matrix as LCA depths (Prop. A.18); morph: morphological closed form (Cor. A.28);
dtrop_on_aug: d_Trop^2 = d_Gamma - M_AG on the augmented graph; sherman: rank-one chord update
(Prop. A.21); bridge: bridge contraction (Prop. A.23); lip: Lipschitz bound (Prop. A.41); edm: sign
pattern of the spectrum (Thm. A.39); kron: Kron reduction and subdivision invariance (Lemma A.25);
fmax: unit-current bound (Lemma A.9). Then the strongly regular example of Cor. A.45.
Usage (from the package root):  python 7_appendix_numerical_checks/verify_propositions.py
"""
import numpy as np, networkx as nx
random_tree = nx.random_labeled_tree if hasattr(nx, 'random_labeled_tree') else nx.random_tree
rng=np.random.default_rng(1)
def resist(edges,n):
    W=np.zeros((n,n))
    for u,v,l in edges: W[u,v]+=1/l; W[v,u]+=1/l
    L=np.diag(W.sum(1))-W; Lp=np.linalg.pinv(L); dg=np.diag(Lp)
    return dg[:,None]+dg[None,:]-2*Lp, L, Lp
def sp(edges,n):
    G=nx.Graph()
    G.add_nodes_from(range(n))
    for u,v,l in edges:
        if G.has_edge(u,v): G[u][v]['weight']=min(G[u][v]['weight'],l)
        else: G.add_edge(u,v,weight=l)
    return nx.floyd_warshall_numpy(G,nodelist=range(n))
errs={k:0.0 for k in ['Qformula','morph','sherman','bridge','lip','edm','kron','fmax','dtrop_on_aug']}
for trial in range(200):
    n=int(rng.integers(6,22)); T=random_tree(n,seed=int(rng.integers(1e9)))
    tedges=[(u,v,float(rng.uniform(0.3,4))) for u,v in T.edges]
    for u,v,l in tedges: T[u][v]['l']=l
    root=0; leaves=[v for v in T if T.degree(v)==1 and v!=root]
    k=int(rng.integers(1,min(4,len(leaves))+1)); aug=[int(x) for x in rng.choice(leaves,size=k,replace=False)]
    a={v:float(rng.uniform(0.2,5)) for v in aug}
    edges=tedges+[(v,root,a[v]) for v in aug]; m=len(edges); ell=np.array([e[2] for e in edges])
    tid={}
    for i,(u,v,l) in enumerate(tedges): tid[(u,v)]=(i,1.0); tid[(v,u)]=(i,-1.0)
    def tchain(path):
        c=np.zeros(m)
        for x,y in zip(path[:-1],path[1:]): i,sg=tid[(x,y)]; c[i]+=sg
        return c
    C=[]
    for j,v in enumerate(aug):
        c=tchain(nx.shortest_path(T,root,v)); c[len(tedges)+j]=1.0  # edge oriented v->root
        C.append(c)
    C=np.array(C); Q=C@np.diag(ell)@C.T
    depth=nx.single_source_dijkstra_path_length(T,root,weight='l')
    Tr=nx.bfs_tree(T,root)
    def lca(u,v):
        pu=nx.shortest_path(Tr,root,u); pv=nx.shortest_path(Tr,root,v); c=root
        for x,y in zip(pu,pv):
            if x==y: c=x
            else: break
        return c
    Qf=np.array([[depth[lca(u,v)] if u!=v else depth[u]+a[u] for v in aug] for u in aug])
    errs['Qformula']=max(errs['Qformula'],np.abs(Q-Qf).max())
    R,L,Lp=resist(edges,n); dT=sp(tedges,n); dG=sp(edges,n); Qi=np.linalg.inv(Q)
    for x in range(n):
        for y in range(n):
            p=tchain(nx.shortest_path(T,x,y)); s=C@np.diag(ell)@p
            errs['morph']=max(errs['morph'],abs(R[x,y]-(dT[x,y]-s@Qi@s)))
            # exact tropical distance on augmented graph via brute-force small box
            import itertools
            if k<=3:
                best=min((s-Q@np.array(z))@Qi@(s-Q@np.array(z)) for z in itertools.product(range(-3,4),repeat=k))
                errs['dtrop_on_aug']=max(errs['dtrop_on_aug'],abs(best-(dG[x,y]-R[x,y])))
    # Sherman-Morrison for single augmented edge
    v=aug[0]; R0,L0,L0p=resist(tedges,n); R1,_,_=resist(tedges+[(v,root,a[v])],n)
    b=np.zeros(n); b[v]=1; b[root]=-1
    for x in range(n):
        for y in range(n):
            dxy=np.zeros(n); dxy[x]+=1; dxy[y]-=1
            errs['sherman']=max(errs['sherman'],abs(R1[x,y]-(R0[x,y]-(dxy@L0p@b)**2/(a[v]+b@L0p@b))))
    # bridge contraction exactness
    Gs=nx.Graph(); Gs.add_nodes_from(range(n))
    for u,w,l in edges: Gs.add_edge(u,w)
    from collections import Counter
    mult=Counter(frozenset((p,q)) for p,q,l in edges)
    br=[e for e in nx.bridges(Gs) if mult[frozenset(e)]==1]
    if br:
        u,w=br[0]; lb=[l for (p,q,l) in edges if {p,q}=={u,w}][0]
        remap=lambda z: u if z==w else z
        H=[(remap(p),remap(q),l) for (p,q,l) in edges if {p,q}!={u,w}]
        keep=[z for z in range(n) if z!=w]; mp={z:i for i,z in enumerate(keep)}
        H2=[(mp[p],mp[q],l) for p,q,l in H]; Rh,_,_=resist(H2,n-1)
        Gc=Gs.copy(); Gc.remove_edge(u,w); comp=nx.node_connected_component(Gc,u)
        for x in range(n):
            for y in range(n):
                sep=(x in comp)!=(y in comp)
                errs['bridge']=max(errs['bridge'],abs(R[x,y]-(Rh[mp[remap(x)],mp[remap(y)]]+lb*sep)))
    # Lipschitz
    edges2=[(u,w,l*float(rng.uniform(0.7,1.3))) for u,w,l in edges]; delta=sum(abs(e2[2]-e[2]) for e,e2 in zip(edges,edges2))
    R2,_,_=resist(edges2,n); errs['lip']=max(errs['lip'],(np.abs(R2-R)-delta).max())
    ev=np.linalg.eigvalsh(R); pos=(ev>1e-9).sum()
    errs['edm']=max(errs['edm'],abs(pos-1)+abs(ev.max()-np.abs(ev[ev<0]).sum()))
    # Kron / subdivision invariance on tree+aug graph
    sub=[]; nid=n
    for u,w,l in edges:
        parts=int(rng.integers(1,4)); cuts=np.sort(rng.uniform(0,1,parts-1)); pts=np.concatenate([[0],cuts,[1]]); prev=u
        for i in range(1,len(pts)):
            nxt=w if i==len(pts)-1 else nid
            if i<len(pts)-1: nid+=1
            sub.append((prev,nxt,l*(pts[i]-pts[i-1]))); prev=nxt
    RS,_,_=resist(sub,nid); errs['kron']=max(errs['kron'],np.abs(RS[:n,:n]-R).max())
    for _ in range(5):
        x,y=rng.choice(n,2,replace=False); dxy=np.zeros(n); dxy[x]=1; dxy[y]=-1; u_=Lp@dxy
        for p,q,l in edges: errs['fmax']=max(errs['fmax'],abs((u_[p]-u_[q])/l)-1)
print({k:float(v) for k,v in errs.items()})
R1=nx.convert_node_labels_to_integers(nx.cartesian_product(nx.complete_graph(4),nx.complete_graph(4)))
Sh=nx.Graph()
for i in range(4):
    for j in range(4):
        for di,dj in [(0,1),(1,0),(1,1)]: Sh.add_edge(i*4+j,((i+di)%4)*4+(j+dj)%4)
print('SRG(16,6,2,2) rook vs Shrikhande isomorphic?',nx.is_isomorphic(R1,Sh))
for Gs in [R1,Sh]:
    A=nx.to_numpy_array(Gs,nodelist=range(16)); L=np.diag(A.sum(1))-A; Lp=np.linalg.pinv(L); dg=np.diag(Lp); M=dg[:,None]+dg[None,:]-2*Lp
    off=~np.eye(16,dtype=bool)
    print(' r(adj) values',np.unique(np.round(M[A==1],10)),' r(nonadj)',np.unique(np.round(M[(A==0)&off],10)),' spec',np.unique(np.round(np.linalg.eigvalsh(M),8)))
