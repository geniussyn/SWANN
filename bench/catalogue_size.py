"""How large should the entry catalogue be, as a function of the corpus size?

IVF's nlist heuristic (4*sqrt(N)..16*sqrt(N)) comes from balancing a coarse scan
of O(nlist*d) against a list scan of O(nprobe * N/nlist * d): the cells partition
the corpus, so nlist appears in both terms and the optimum is sqrt-shaped. A
SWANN catalogue partitions nothing -- it only chooses starting points -- so P
appears in the scan term and only indirectly, through entry quality, in the
search term. The optimum is therefore a different shape and has to be measured.

For each corpus size and each P, eps is swept to build a recall/time frontier,
and the total (selection + search) is read off it at a fixed recall. The P that
minimises that total is the optimum at that size.

Measured, Q=2000, best P by target recall (MS MARCO ladder, d=1024):

    N        0.980   0.985   0.990   0.992   0.994
    125K       --     8192     8192     --      --
    250K       --    32768*   32768*  32768*   --
    500K       --    32768*   32768*    --      --
    1M       2048     8192    32768   32768   262144
    (* ladder topped out at 32768)

So the optimum is set by the recall target and is nearly independent of N -- the
same P = 32768 wins at 250K, 500K and 1M, which is P/N of 13%, 6.6% and 3.3%.
Expressing it as a fraction of N is the wrong parameterisation. On GloVe
(d = 104, N = 1.18M) the optimum is 8192 and the curve turns over above it,
because the scan is bound by the [Q, P] score traffic rather than by the
dimension and so does not get cheaper as d falls.

The ceiling is brute force. `bench/brute_force_floor.py` on the same ladder gives
8.2 / 16.2 / 32.4 / 64.4 ms at recall 0.9998, against this search at recall 0.990
of 22.0 / 28.7 / 34.4 / 35.2 -- so brute force wins by 2.7x at 125K, 1.8x at
250K, ties at 500K, and loses by 1.83x at 1M.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a
from gpuann.pools import load_pool


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def frontier(rows):
    pts = sorted({(r["recall"], r["total"]) for r in rows})
    out, best = [], float("inf")
    for rc, t in sorted(pts, key=lambda p: (-p[0], p[1])):
        if t < best:
            out.append((rc, t)); best = t
    return sorted(out)


def time_at(front, rc):
    if not front or rc < front[0][0] or rc > front[-1][0]:
        return None
    for (r0, t0), (r1, t1) in zip(front, front[1:]):
        if r0 <= rc <= r1:
            return t0 if r1 == r0 else t0 + (t1 - t0) * (rc - r0) / (r1 - r0)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--corpora", nargs="+",
                default=["ms-125k", "ms-250k", "ms-500k", "msmarco-1m"])
ap.add_argument("--root", type=Path, default=Path("/dev/shm/gpuann"))
ap.add_argument("--Ps", type=int, nargs="+",
                default=[512, 1024, 2048, 4096, 8192, 16384, 32768, 65536])
ap.add_argument("--epss", type=float, nargs="+", default=[0.05, 0.10, 0.15, 0.22, 0.32])
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--nq", type=int, default=2000)
ap.add_argument("--split", default=None, help="val, else test when no val exists")
ap.add_argument("--targets", type=float, nargs="+", default=[0.98, 0.99])
ap.add_argument("--repeats", type=int, default=2)
ap.add_argument("--out", type=Path, default=None)
a = ap.parse_args()

allrows = []
for corpus in a.corpora:
    d = a.root / corpus
    man = json.loads((d / "manifest.json").read_text())
    graph = "G_ptf32.npy" if (d / "G_ptf32.npy").exists() else "G_d64.npy"
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    Xg = torch.from_numpy(np.load(d / "X.npy")).cuda()
    split = a.split or ("val" if (d / "Q_val.npy").exists() else "test")
    Q = np.ascontiguousarray(np.load(d / f"Q_{split}.npy")[: a.nq])
    gt = np.load(d / f"gt_{split}_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    Gg = torch.from_numpy(np.ascontiguousarray(np.load(d / graph))).cuda()
    dg = engine.graph_degrees(Gg)
    N = Xg.shape[0]
    print(f"\n### {corpus}  n={N:,} d={Xg.shape[1]} graph={graph} {split} q={len(Q)}")
    print(f"{'P':>7s} {'P/N':>8s} {'N/P':>7s} {'eps':>5s} {'recall':>8s} "
          f"{'sel ms':>7s} {'srch ms':>8s} {'total':>8s}")
    fronts = {}
    for P in a.Ps:
        if P > N // 2:
            continue
        pool = load_pool(d, "random", P)
        pool_t = torch.from_numpy(pool).cuda()
        catX = select_a.pool_vectors(Xg, pool_t)
        rows = []
        for eps in a.epss:
            sb = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                near = select_a.select_nearest(Qg, catX, a.V, metric=metric,
                                               precision="tf32")
                torch.cuda.synchronize(); sel = (time.perf_counter() - t) * 1e3
                if r and (sb is None or sel < sb[0]):
                    sb = (sel, near)
            sel_ms, near = sb
            seeds = pool_t[near].unsqueeze(-1).contiguous()
            gb = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(
                    Xg, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                    shared_cache=True, exclusive=True, shared_bound=True)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (gb is None or ms < gb[0]):
                    gb = (ms, ids)
            srch_ms, ids = gb
            rc = recall_at_k(ids.cpu().numpy(), gt)
            rows.append(dict(corpus=corpus, N=int(N), P=P, eps=eps, recall=rc,
                             sel=sel_ms, search=srch_ms, total=sel_ms + srch_ms))
            print(f"{P:7d} {P/N:8.5f} {N/P:7.0f} {eps:5.2f} {rc:8.4f} "
                  f"{sel_ms:7.3f} {srch_ms:8.2f} {sel_ms+srch_ms:8.2f}")
            del ids, dist, st
        fronts[P] = frontier(rows)
        allrows += rows
        del catX, pool_t
        torch.cuda.empty_cache()
    print(f"\n  total ms at matched recall, by P  (n={N:,})")
    hdr = "  ".join(f"{p:>8d}" for p in sorted(fronts))
    print(f"  {'recall':>7s}  {hdr}")
    for tgt in a.targets:
        cells, bestP, bestT = [], None, float("inf")
        for p in sorted(fronts):
            t = time_at(fronts[p], tgt)
            cells.append("      --" if t is None else f"{t:8.2f}")
            if t is not None and t < bestT:
                bestT, bestP = t, p
        print(f"  {tgt:7.3f}  " + "  ".join(cells) +
              (f"   <- best P={bestP} (P/N={bestP/N:.5f}, N/P={N/bestP:.0f})"
               if bestP else ""))
    del Xg, Gg, Qg
    torch.cuda.empty_cache()

if a.out:
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(V=a.V, L=a.L, B=a.B, nq=a.nq, rows=allrows), indent=2))
    print(f"\nwrote {a.out}")
