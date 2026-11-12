"""Fill a pruned graph's free row slots with the reverse edges the prune dropped.

phi_1 joins two answer documents exactly when the graph holds both directions of
an edge (offline/phi_h.py: `reach[a,b] and reach[b,a]`, and a one-hop reach is
the neighbour set), so reciprocity is not a proxy for fragmentation at h = 1 --
it is the same quantity read on the whole graph. Measured on MS MARCO 1M, our
pruned graph is 59.5% reciprocal and mean phi_1 is 3.083; CAGRA's is 82.1% and
2.784. The prune buys navigability by dropping one direction of an edge, and
that is exactly what phi_1 charges for.

The rows are stored dense and front-compacted at a fixed width -- 96 for a mean
out-degree of 52.1 -- so 46% of the allocated slots hold -1. An edge put into a
free slot therefore costs no memory at all. It costs work, because the kernel
bounds an expansion by deg[p], and that is what has to be measured rather than
assumed.

    python offline/reciprocate.py --data /dev/shm/gpuann/msmarco-1m \
        --graph G_d64.npy --out G_recip.npy
"""
import argparse, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


def edge_lengths(X, G, chunk=20000):
    """d(u, G[u,j]) for every slot, +inf where the slot is empty. IP metric."""
    n, W = G.shape
    out = np.full((n, W), np.inf, dtype=np.float32)
    Gt = torch.from_numpy(G.astype(np.int64))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        g = Gt[s:e].cuda()
        m = g >= 0
        nb = X[g.clamp(min=0)]                       # [b, W, d]
        d = 1.0 - (X[s:e].unsqueeze(1) * nb).sum(-1)
        d[~m] = float("inf")
        out[s:e] = d.cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--out", default="G_recip.npy")
    ap.add_argument("--budget", type=float, default=1.0,
                    help="fraction of each row's free slots that may be filled")
    ap.add_argument("--max-deg", type=int, default=0,
                    help="cap the repaired out-degree (0 = the row width)")
    a = ap.parse_args()

    G = np.load(a.data / a.graph)
    n, W = G.shape
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    t = time.perf_counter()
    elen = edge_lengths(X, G)
    print(f"edge lengths in {time.perf_counter()-t:.1f}s")

    deg = (G >= 0).sum(1)
    rows = np.repeat(np.arange(n, dtype=np.int64), W)
    cols = G.reshape(-1).astype(np.int64)
    lens = elen.reshape(-1)
    keep = cols >= 0
    rows, cols, lens = rows[keep], cols[keep], lens[keep]
    A = sp.csr_matrix((np.ones(len(rows), np.bool_), (rows, cols)), shape=(n, n))
    both = A.multiply(A.T).tocsr()
    print(f"{a.graph}: {len(rows):,} edges, {both.nnz:,} reciprocal "
          f"({100*both.nnz/len(rows):.1f}%), mean deg {deg.mean():.1f} of width {W}")

    # Proposals: for every one-way edge u -> v, offer v -> u at the same length.
    rec = np.asarray(both[rows, cols]).ravel().astype(bool)
    pu, pv, pl = rows[~rec], cols[~rec], lens[~rec]      # u -> v one-way
    print(f"one-way edges: {len(pu):,}")

    cap = a.max_deg or W
    room = np.maximum(cap - deg, 0)                       # free slots per node
    print(f"free slots: {room.sum():,} ({room.mean():.1f} per node)")

    # Each proposal lands in pv's row. Keep the shortest `room[v]` of them.
    order = np.lexsort((pl, pv))                          # by target, then length
    pv_s, pu_s, pl_s = pv[order], pu[order], pl[order]
    start = np.searchsorted(pv_s, np.arange(n), "left")
    rank = np.arange(len(pv_s)) - start[pv_s]             # position within the target
    allow = np.minimum(room[pv_s], np.floor(room[pv_s] * a.budget).astype(np.int64))
    take = rank < allow
    print(f"accepted {int(take.sum()):,} of {len(pu):,} proposals "
          f"({100*take.sum()/max(1,len(pu)):.1f}%)")

    out = G.copy()
    write = deg.copy()
    tv, tu = pv_s[take], pu_s[take]
    # Append in target order; a target's accepted proposals are contiguous and
    # already sorted by length, so the row stays distance-ordered only if every
    # appended edge is longer than the row's last -- it is not, so the row is
    # re-sorted below.
    for v, u in zip(tv.tolist(), tu.tolist()):
        out[v, write[v]] = u
        write[v] += 1
    # Re-sort every touched row by distance and keep the front compaction.
    touched = np.unique(tv)
    print(f"re-sorting {len(touched):,} rows")
    Xc = X
    for s in range(0, len(touched), 20000):
        idx = touched[s:s + 20000]
        g = torch.from_numpy(out[idx].astype(np.int64)).cuda()
        m = g >= 0
        d = 1.0 - (Xc[idx].unsqueeze(1) * Xc[g.clamp(min=0)]).sum(-1)
        d[~m] = float("inf")
        o = d.argsort(dim=1)
        out[idx] = torch.gather(g, 1, o).cpu().numpy().astype(np.int32)
        out[idx][~torch.gather(m, 1, o).cpu().numpy()] = -1

    nd = (out >= 0).sum(1)
    r2 = np.repeat(np.arange(n, dtype=np.int64), W)
    c2 = out.reshape(-1).astype(np.int64)
    k2 = c2 >= 0
    A2 = sp.csr_matrix((np.ones(int(k2.sum()), np.bool_), (r2[k2], c2[k2])), shape=(n, n))
    b2 = A2.multiply(A2.T)
    print(f"-> {a.out}: {int(k2.sum()):,} edges (mean deg {nd.mean():.1f}), "
          f"{b2.nnz:,} reciprocal ({100*b2.nnz/int(k2.sum()):.1f}%)")
    np.save(a.data / a.out, np.ascontiguousarray(out.astype(np.int32)))


if __name__ == "__main__":
    main()
