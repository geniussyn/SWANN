"""phi_h: into how many pieces the serving graph cuts a query's exact top-k.

Definition 2.1/3.1 of the manuscript, unchanged: build a graph on the k
ground-truth documents in which u and v are joined when each is reachable from
the other within h hops of G, and count its connected components. phi_h = 1 is a
query whose answer is one connected piece; phi_h = k is a query whose answer the
graph has scattered into k mutually unreachable singletons.

This is the instrument the whole multi-entry argument needs, because "recover
each fragment and stop" is only measurable once the fragments have names. What
this writes, per query: phi, and a component label per ground-truth slot, so an
arm can be scored on *which pieces it touched* instead of on how many documents
it happened to return.

Reachability is computed with sparse boolean products over the GT nodes only --
the h-hop reach of every GT node, restricted to the GT nodes -- so the cost is
set by how many distinct GT documents the query set has, not by N.
"""
import argparse, json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def adjacency(G, n):
    deg = G.shape[1]
    rows = np.repeat(np.arange(n, dtype=np.int32), deg)
    cols = G.reshape(-1).astype(np.int32)
    keep = cols >= 0
    return sp.csr_matrix((np.ones(int(keep.sum()), np.bool_), (rows[keep], cols[keep])),
                         shape=(n, n))


def reach_among(A, targets, h, chunk=2048):
    """[T, T] boolean: can source i reach target j within h hops (i != j)."""
    n = A.shape[0]
    T = len(targets)
    col = sp.csr_matrix((np.ones(T, np.bool_), (targets.astype(np.int64), np.arange(T))),
                        shape=(n, T))
    out = np.zeros((T, T), dtype=bool)
    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        rows = np.arange(e - s)
        cur = sp.csr_matrix((np.ones(e - s, np.bool_), (rows, targets[s:e].astype(np.int64))),
                            shape=(e - s, n))
        acc = cur.copy()
        for _ in range(h):
            cur = (cur @ A).astype(bool)
            acc = (acc + cur).astype(bool)
        out[s:e] = np.asarray((acc @ col).todense())
    return out


def components(labels_reach):
    """Union-find over one query's k x k mutual-reachability matrix."""
    k = labels_reach.shape[0]
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x

    for a in range(k):
        for b in range(a + 1, k):
            if labels_reach[a, b] and labels_reach[b, a]:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    roots = {}
    lab = np.empty(k, dtype=np.int8)
    for x in range(k):
        r = find(x)
        if r not in roots:
            roots[r] = len(roots)
        lab[x] = roots[r]
    return len(roots), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--graph", default="G.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--tag", default=None, help="output suffix; defaults to the graph name")
    a = ap.parse_args()

    G = np.load(a.data / a.graph)
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.nq:
        gt = gt[: a.nq]
    nq, k = gt.shape
    A = adjacency(G, G.shape[0])

    targets, inv = np.unique(gt.reshape(-1), return_inverse=True)
    inv = inv.reshape(nq, k)
    print(f"graph={a.graph} h={a.h} queries={nq} k={k}  distinct GT docs {len(targets):,}")
    R = reach_among(A, targets, a.h)

    phi = np.empty(nq, dtype=np.int32)
    labels = np.empty((nq, k), dtype=np.int8)
    for q in range(nq):
        sub = R[np.ix_(inv[q], inv[q])]
        phi[q], labels[q] = components(sub)

    tag = a.tag or Path(a.graph).stem
    np.save(a.data / f"phi{a.h}_{a.split}_{tag}.npy", phi)
    np.save(a.data / f"frag{a.h}_{a.split}_{tag}.npy", labels)

    hist = np.bincount(phi, minlength=k + 1)[1:k + 1]
    print(f"\nphi_{a.h} distribution (1 = answer is one connected piece):")
    for i, c in enumerate(hist, start=1):
        if c:
            print(f"  phi={i:2d}  {c:6d}  {100*c/nq:5.1f}%")
    print(f"\nmean phi_{a.h} = {phi.mean():.3f}   one piece: {100*(phi==1).mean():.1f}%   "
          f"fully scattered: {100*(phi==k).mean():.1f}%")
    (a.data / f"phi{a.h}_{a.split}_{tag}.json").write_text(json.dumps(
        dict(graph=a.graph, h=a.h, split=a.split, nq=int(nq), k=int(k),
             mean_phi=float(phi.mean()), frac_one=float((phi == 1).mean()),
             hist={str(i + 1): int(c) for i, c in enumerate(hist)}), indent=2))


if __name__ == "__main__":
    main()
