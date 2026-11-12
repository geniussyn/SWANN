"""Throughput and latency against batch size, for every system on one axis.

A GPU index is usually reported at one batch -- whatever fills the device --
and that hides the question an online system actually asks: what does the first
query cost when there is only one. The three designs answer it differently. A
graph search issues one block per (query, branch), so a single query already has
V blocks of work; a partitioned index scans lists whose size does not shrink
with the batch; and every one of them pays a fixed launch cost that only
amortises when there is something to amortise it over.

Each system is held at a setting that reaches a comparable recall, so the curves
differ in how they scale and not in what they are solving. Latency is the batch's
own wall clock, not a per-query division, because that is what a caller waits.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    import torch
    from gpuann import engine, select_a
    from gpuann.pools import load_pool

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q0 = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    print(f"{man['corpus']} d={X.shape[1]} n={man['n']:,}  "
          f"V={a.V} L={a.L} B={a.B} eps={a.stop_eps}\n")
    dim = X.shape[1]
    esz = X.element_size()
    print(f"{'batch':>6} {'blocks':>9} {'latency ms':>11} {'QPS':>11} "
          f"{'us / query':>11} {'GB/s':>8}")
    rows = []
    for nq in a.batches:
        Q = (Q0[:nq] if nq <= Q0.shape[0]
             else Q0.repeat((nq + Q0.shape[0] - 1) // Q0.shape[0], 1)[:nq])
        Q = Q.contiguous()
        seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)].to(torch.int32)
        best = None
        for r in range(a.repeats + 1):
            torch.cuda.synchronize(); t = time.perf_counter()
            _, _, st = engine.search(X, G, Q, seeds, k=a.k, L=a.L, B=a.B,
                                     metric=metric, team_size=a.team,
                                     stop_eps=a.stop_eps, shared_cache=True,
                                     shared_bound=True, exclusive=True,
                                     elide_visited=True)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        # Vector traffic, which is what the expansion is bound by: every
        # distance the kernel counted read one whole vector. Counted rather
        # than modelled -- the kernel keeps the tally -- and it excludes the
        # graph rows and the queue, which are small beside it.
        de = float(st.dist_evals.sum().item())
        gbs = de * dim * esz / 1e9 / (best / 1e3)
        print(f"{nq:6,} {nq * a.V:9,} {best:11.3f} {nq/(best/1e3):11,.0f} "
              f"{1000*best/nq:11.1f} {gbs:8.0f}")
        rows.append(dict(system="SWANN", batch=nq, blocks=nq * a.V,
                         latency_ms=best, qps=nq / (best / 1e3),
                         us_per_query=1000 * best / nq,
                         vector_gb_per_s=gbs, dist_evals=de))
        del Q, seeds
    print(f"\n(the device holds 2,256 blocks at once)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, V=a.V, L=a.L, B=a.B,
                                         stop_eps=a.stop_eps, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
