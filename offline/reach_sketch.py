"""The reach sketch: which regions each pool entry can see, as a bitset.

For every entry in the pool, walk the *original directed* graph forward to depth
h with a cap on how many nodes are visited, and set a bit for every region those
nodes fall in. The result is mask[P, ceil(R/32)] uint32 -- static, query-free,
and small enough to sit in shared memory (8 KiB at P=512, R=128).

This is a surrogate borrowed from reverse-reachable sets, not a reachability
proof: a set bit says "a short walk from this entry reaches that region", which
is not "this entry's search will recover that region's documents for this
query". The forward form is used as the default because the reverse form, with
few sampled targets, leaves most masks empty -- and an empty mask is not
evidence of unreachability, only of not having looked. Both are written, so the
ablation the document asks for is a flag, not a rewrite.

Bit density and the overlap between entries' masks are reported, because a
sketch where every entry sees everything cannot discriminate and a sketch where
no entry sees anything cannot either.
"""
import argparse, json, time
from pathlib import Path

import numpy as np


def forward_mask(G, region, pool, R, h=2, cap=256):
    """Depth-h forward walk from each entry, capped at `cap` visited nodes."""
    P, W = len(pool), (R + 31) // 32
    mask = np.zeros((P, W), dtype=np.uint32)
    deg = G.shape[1]
    for i, s in enumerate(pool):
        seen = {int(s)}
        frontier = [int(s)]
        mask[i, region[s] >> 5] |= np.uint32(1 << (int(region[s]) & 31))
        for _ in range(h):
            nxt = []
            for p in frontier:
                for j in range(deg):
                    v = int(G[p, j])
                    if v < 0 or v in seen:
                        continue
                    seen.add(v)
                    r = int(region[v])
                    mask[i, r >> 5] |= np.uint32(1 << (r & 31))
                    nxt.append(v)
                    if len(seen) >= cap:
                        break
                if len(seen) >= cap:
                    break
            if len(seen) >= cap or not nxt:
                break
            frontier = nxt
    return mask


def reverse_mask(G, region, pool, R, n, targets_per_region=64, h=2, seed=0):
    """The ablation: sample targets in each region, walk the reverse graph, and
    set a region's bit on every entry that reaches one of its samples."""
    import scipy.sparse as sp
    rng = np.random.default_rng(seed)
    P, W = len(pool), (R + 31) // 32
    mask = np.zeros((P, W), dtype=np.uint32)
    rows = np.repeat(np.arange(n, dtype=np.int64), G.shape[1])
    cols = G.reshape(-1).astype(np.int64)
    keep = cols >= 0
    RT = sp.csr_matrix((np.ones(keep.sum(), np.int8), (cols[keep], rows[keep])), shape=(n, n))
    inpool = {int(v): i for i, v in enumerate(pool)}
    for r in range(R):
        mem = np.flatnonzero(region == r)
        if mem.size == 0:
            continue
        tg = rng.choice(mem, size=min(targets_per_region, mem.size), replace=False)
        seen = set(int(t) for t in tg)
        frontier = list(seen)
        for _ in range(h):
            if not frontier:
                break
            nxt = []
            for p in frontier:
                for v in RT.indices[RT.indptr[p]:RT.indptr[p + 1]]:
                    v = int(v)
                    if v not in seen:
                        seen.add(v); nxt.append(v)
            frontier = nxt
        for v in seen:
            i = inpool.get(v)
            if i is not None:
                mask[i, r >> 5] |= np.uint32(1 << (r & 31))
    return mask


def report(mask, R, name):
    bits = np.unpackbits(mask.view(np.uint8), bitorder="little").reshape(len(mask), -1)[:, :R]
    dens = bits.sum(1)
    inter = bits.astype(np.float32) @ bits.astype(np.float32).T
    iu = np.triu_indices(len(mask), 1)
    union = dens[iu[0]] + dens[iu[1]] - inter[iu]
    jac = np.where(union > 0, inter[iu] / np.maximum(union, 1), 0.0)
    print(f"{name:8s} regions per entry: min/med/max {dens.min()}/{int(np.median(dens))}/{dens.max()}"
          f"  empty entries {int((dens == 0).sum())}  mean pairwise Jaccard {jac.mean():.3f}")
    return dict(density_min=int(dens.min()), density_med=int(np.median(dens)),
                density_max=int(dens.max()), empty=int((dens == 0).sum()),
                mean_jaccard=float(jac.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--pool", default="metis")
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--cap", type=int, default=256)
    ap.add_argument("--reverse", action="store_true", help="also build the reverse ablation")
    a = ap.parse_args()

    G = np.load(a.data / "G.npy")
    pools = a.data / "pools"
    pool = np.load(pools / f"pool_{a.pool}.npy")
    region = np.load(pools / f"region_{a.pool}.npy")
    stats = {}
    t = time.perf_counter()
    fwd = forward_mask(G, region, pool, a.R, a.h, a.cap)
    stats["forward"] = report(fwd, a.R, "forward")
    stats["forward"]["seconds"] = round(time.perf_counter() - t, 2)
    np.save(pools / f"mask_{a.pool}_fwd.npy", fwd)
    if a.reverse:
        t = time.perf_counter()
        rev = reverse_mask(G, region, pool, a.R, G.shape[0], h=a.h)
        stats["reverse"] = report(rev, a.R, "reverse")
        stats["reverse"]["seconds"] = round(time.perf_counter() - t, 2)
        np.save(pools / f"mask_{a.pool}_rev.npy", rev)
    (pools / f"sketch_{a.pool}.json").write_text(json.dumps(
        dict(pool=a.pool, R=a.R, h=a.h, cap=a.cap, stats=stats), indent=2))
    print(f"wrote mask_{a.pool}_*.npy")


if __name__ == "__main__":
    main()
