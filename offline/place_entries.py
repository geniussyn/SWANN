"""Where to put the entry candidates, before anything decides which to use.

Entry selection can only choose from the pool it is given, so the pool is a
separate design decision from the selector and is measured separately: the H0
oracle reads the ceiling of a pool, and that ceiling is a property of placement
alone. Five placements, all offline, none of which sees a query or GT:

  random     P uniform nodes. The baseline, and the one already measured.
  metis      METIS k-way on the symmetrised graph, R regions, entries drawn
             inside each region. This is the design document's placement.
  kmeans     the same shape but with geometric regions: k-means on the vectors.
             It separates "regions of the graph" from "regions of the space".
  farthest   greedy max-min over a candidate subset: maximally spread, with no
             region structure at all.
  degree     the graph's hubs, by in-degree. The cheapest thing that is not
             random, and a useful spoiler: if hubs win, spread is not the point.

metis and kmeans also write `region[N]` and `reps[R]`, which A needs for its
region weights; the other placements have no regions and write only the pool.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch


def symmetric_csr(G, n):
    """G's edges plus their reverses, no self-loops, as (xadj, adjncy)."""
    import scipy.sparse as sp
    deg = G.shape[1]
    rows = np.repeat(np.arange(n, dtype=np.int64), deg)
    cols = G.reshape(-1).astype(np.int64)
    keep = (cols >= 0) & (cols != rows)
    A = sp.csr_matrix((np.ones(keep.sum(), np.int8), (rows[keep], cols[keep])), shape=(n, n))
    S = (A + A.T).tocsr()
    S.setdiag(0); S.eliminate_zeros()
    return S.indptr.astype(np.int64), S.indices.astype(np.int64)


def region_entries(X, region, R, per_region, sample=32, seed=0):
    """Representative and entries inside each region, the document's recipe:
    of `sample` fixed random nodes in a region, the one nearest the region mean
    is the representative and `per_region` others are entries. Regions too small
    to supply their share hand the deficit to the others, without duplicates."""
    rng = np.random.default_rng(seed)
    members = [np.flatnonzero(region == r) for r in range(R)]
    reps = np.full(R, -1, dtype=np.int32)
    pool, taken = [], set()
    deficit = 0
    for r, mem in enumerate(members):
        if mem.size == 0:
            deficit += per_region
            continue
        pick = rng.choice(mem, size=min(sample, mem.size), replace=False)
        mean = X[mem].mean(0)
        mean /= max(np.linalg.norm(mean), 1e-12)
        rep = int(pick[np.argmax(X[pick] @ mean)])
        reps[r] = rep
        others = [int(v) for v in pick if v != rep and int(v) not in taken]
        use = others[:per_region]
        deficit += per_region - len(use)
        for v in use:
            taken.add(v); pool.append(v)
    if deficit:                                    # fill from the largest regions
        order = np.argsort([-m.size for m in members])
        for r in order:
            if deficit <= 0:
                break
            for v in members[r]:
                v = int(v)
                if v not in taken:
                    taken.add(v); pool.append(v); deficit -= 1
                    if deficit <= 0:
                        break
    return np.sort(np.array(pool, dtype=np.int32)), reps


def place_random(X, G, P, R, rng, **kw):
    return np.sort(rng.choice(X.shape[0], P, replace=False).astype(np.int32)), None, None


def place_metis(X, G, P, R, rng, seed=0, **kw):
    import pymetis
    xadj, adjncy = symmetric_csr(G, X.shape[0])
    _, parts = pymetis.part_graph(R, xadj=xadj, adjncy=adjncy)
    region = np.asarray(parts, dtype=np.uint16)
    pool, reps = region_entries(X, region, R, max(1, P // R), seed=seed)
    return pool, region, reps


def place_kmeans(X, G, P, R, rng, iters=25, seed=0, **kw):
    Xg = torch.from_numpy(X).cuda()
    C = Xg[torch.from_numpy(rng.choice(X.shape[0], R, replace=False)).cuda()].clone()
    for _ in range(iters):
        assign = torch.empty(Xg.shape[0], dtype=torch.long, device="cuda")
        for s in range(0, Xg.shape[0], 20000):
            assign[s:s + 20000] = (Xg[s:s + 20000] @ C.T).argmax(1)
        newC = torch.zeros_like(C)
        newC.index_add_(0, assign, Xg)
        cnt = torch.bincount(assign, minlength=R).unsqueeze(1).clamp(min=1)
        newC = newC / cnt
        nrm = newC.norm(dim=1, keepdim=True)
        empty = (nrm.squeeze(1) < 1e-8)
        newC[empty] = C[empty]                      # an empty cluster keeps its centre
        C = newC / newC.norm(dim=1, keepdim=True).clamp(min=1e-12)
    region = assign.cpu().numpy().astype(np.uint16)
    del Xg, C
    torch.cuda.empty_cache()
    pool, reps = region_entries(X, region, R, max(1, P // R), seed=seed)
    return pool, region, reps


def place_farthest(X, G, P, R, rng, candidates=8192, **kw):
    """Greedy max-min inner-product spread over a random candidate subset."""
    cand = rng.choice(X.shape[0], min(candidates, X.shape[0]), replace=False)
    C = torch.from_numpy(np.ascontiguousarray(X[cand])).cuda()
    chosen = [int(rng.integers(len(cand)))]
    best = C @ C[chosen[0]]
    for _ in range(P - 1):
        nxt = int(torch.argmin(best))
        chosen.append(nxt)
        best = torch.maximum(best, C @ C[nxt])
    del C
    torch.cuda.empty_cache()
    return np.sort(cand[np.array(chosen)].astype(np.int32)), None, None


def place_degree(X, G, P, R, rng, **kw):
    n = X.shape[0]
    indeg = np.bincount(G.reshape(-1)[G.reshape(-1) >= 0], minlength=n)
    return np.sort(np.argsort(-indeg, kind="stable")[:P].astype(np.int32)), None, None


PLACERS = dict(random=place_random, metis=place_metis, kmeans=place_kmeans,
               farthest=place_farthest, degree=place_degree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--which", nargs="+", default=list(PLACERS))
    a = ap.parse_args()

    X = np.load(a.data / "X.npy"); G = np.load(a.data / "G.npy")
    out = a.data / "pools"; out.mkdir(exist_ok=True)
    meta = {}
    for name in a.which:
        rng = np.random.default_rng(a.seed)
        import time
        t = time.perf_counter()
        pool, region, reps = PLACERS[name](X, G, a.P, a.R, rng, seed=a.seed)
        secs = time.perf_counter() - t
        assert len(set(pool.tolist())) == len(pool), f"{name}: duplicate entry"
        np.save(out / f"pool_{name}.npy", pool)
        note = ""
        if region is not None:
            np.save(out / f"region_{name}.npy", region)
            np.save(out / f"reps_{name}.npy", reps)
            sizes = np.bincount(region, minlength=a.R)
            note = (f"  regions {a.R}: size min/med/max "
                    f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}, "
                    f"empty {int((sizes == 0).sum())}")
        meta[name] = dict(P=int(len(pool)), seconds=round(secs, 2), regions=a.R if region is not None else None)
        print(f"{name:9s} P={len(pool):4d}  {secs:7.2f}s{note}")
    (out / "manifest.json").write_text(json.dumps(dict(P=a.P, R=a.R, seed=a.seed, placements=meta), indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
