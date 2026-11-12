"""The shipped configuration: V independent branches, one seed each (J = V).

This is the mode the results carry (`fanout: branch`), and the only one in which
the shared distance table, the exclusive claims and the shared bound exist at
all: `init_states` gates each of them on J > 1, so a [Q, 1, V] shared state
silently runs without any of them. Arms add one mechanism at a time, and the
achieved rate is the algorithmic bytes over the measured time.
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


ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--eps", type=float, default=0.10)
ap.add_argument("--nq", type=int, nargs="+", default=[2000])
ap.add_argument("--pool", default="random4k")
ap.add_argument("--P", type=int, default=4096)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--half", action="store_true")
ap.add_argument("--peak", type=float, default=1523.0)
ap.add_argument("--deg", action="store_true", help="also run each arm deg-bounded")
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
X = Xg.half().contiguous() if a.half else Xg
esz = 2 if a.half else 4
Qall = np.load(a.data / "Q_val.npy"); gt_all = np.load(a.data / "gt_val_ids.npy")
Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
dg = engine.graph_degrees(Gg)
pool = load_pool(a.data, a.pool, a.P); pool_t = torch.from_numpy(pool).cuda()
catX = select_a.pool_vectors(Xg, pool_t)
N, d = Xg.shape

ARMS = [("none",              dict()),
        ("+table",            dict(shared_cache=True)),
        ("+table+claims",     dict(shared_cache=True, exclusive=True)),
        ("+table+claims+bnd", dict(shared_cache=True, exclusive=True, shared_bound=True))]

print(f"{man['corpus']} n={N} d={d} graph={a.graph}  fanout=branch  V={a.V} L={a.L} "
      f"B={a.B} eps={a.eps}" + ("  X=fp16" if a.half else "  X=fp32") + "\n")
for nq in a.nq:
    Q = np.ascontiguousarray(Qall[:nq]); gt = gt_all[:nq]
    Qg = torch.from_numpy(Q).cuda()
    Qs = Qg.half() if a.half else Qg
    near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
    seeds = pool_t[near].unsqueeze(-1).contiguous()           # [Q, V, 1] -> J = V
    print(f"  Q={nq}  blocks = V*Q = {a.V*nq}")
    print(f"  {'arm':20s} {'deg':>4s} {'recall@10':>10s} {'dist/q':>8s} {'ms':>8s} "
          f"{'QPS':>8s} {'GB/s':>7s} {'of peak':>8s}")
    for tag, kw in ARMS:
        for dtag, dd in ((" no", None), ("yes", dg)) if a.deg else ((" no", None),):
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(X, Gg, Qg, seeds, k=10, L=a.L, B=a.B,
                                              metric=metric, policy=engine.POLICY_BEST,
                                              stop_eps=a.eps, deg=dd, **kw)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (best is None or ms < best[0]):
                    best = (ms, ids, st)
            ms, ids, st = best
            rc = recall_at_k(ids.cpu().numpy(), gt)
            ev = float(st.dist_evals.sum())
            gbs = ev * d * esz / (ms * 1e-3) / 1e9
            print(f"  {tag:20s} {dtag:>4s} {rc:10.4f} {ev/nq:8.0f} {ms:8.2f} "
                  f"{1e3*nq/ms:8.0f} {gbs:7.0f} {100*gbs/a.peak:7.1f}%")
            del st, ids, dist
            torch.cuda.empty_cache()
    print()
