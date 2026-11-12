"""What the entry catalogue scan actually costs next to the search it seeds.

The scan is a dense [Q, P] matmul and the search is a scattered gather, so their
ratio moves with the batch, the dimension and the stored precision rather than
being a property of P. Selection and search are timed separately on the shipped
branch-mode configuration.

Measured on MS MARCO 1M (V=32, L=16, B=16, eps=0.10), selection as a share of
the total:

    P        Q=1     Q=32    Q=250   Q=2000
    4096     10.3%    3.6%    1.4%     0.5%
    16384    21.3%    8.4%    3.2%     1.7%
    131072     --      --    21.6%    13.0%

So P=4096 is not where the time goes: 0.24 ms of a 47 ms batch. At Q=1 the share
looks large and is not arithmetic -- 0.051 ms at 0.17 TFLOP/s, which is launch
overhead on an idle device -- so the fix there is to stop paying for a launch,
not to score fewer entries. The dense catalogue is the only configuration where
the scan is worth attacking, and FAISS's own guideline puts P=4096 at the bottom
of its recommended 4*sqrt(N)..16*sqrt(N) range for N=1M, so it is not an unusual
size either.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a
from gpuann.pools import load_pool


def bench(fn, repeats=4):
    best = None
    for r in range(repeats + 1):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
        if r and (best is None or ms < best[0]):
            best = (ms, out)
    return best


ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--eps", type=float, default=0.10)
ap.add_argument("--Ps", type=int, nargs="+", default=[4096, 16384])
ap.add_argument("--nq", type=int, nargs="+", default=[1, 32, 250, 2000])
ap.add_argument("--pool", default="random4k")
ap.add_argument("--half", action="store_true")
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Xs = Xg.half().contiguous() if a.half else Xg
Qall = np.load(a.data / "Q_val.npy")
Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
dg = engine.graph_degrees(Gg)
N, d = Xg.shape
print(f"{man['corpus']} n={N} d={d}  branch mode V={a.V} L={a.L} B={a.B} "
      f"eps={a.eps}" + ("  X=fp16" if a.half else "  X=fp32") + "\n")
print(f"{'P':>7s} {'Q':>5s} {'select ms':>10s} {'search ms':>10s} {'total':>8s} "
      f"{'select %':>9s} {'sel TFLOP/s':>12s}")
for P in a.Ps:
    pool = load_pool(a.data, a.pool, P)
    if len(pool) < P:
        print(f"{P:7d}  (pool '{a.pool}' only has {len(pool)}; skipped)")
        continue
    pool_t = torch.from_numpy(pool).cuda()
    catX = select_a.pool_vectors(Xg, pool_t)
    for nq in a.nq:
        Qg = torch.from_numpy(np.ascontiguousarray(Qall[:nq])).cuda()
        sel_ms, near = bench(lambda: select_a.select_nearest(
            Qg, catX, a.V, metric=metric, precision="tf32"))
        seeds = pool_t[near].unsqueeze(-1).contiguous()
        srch_ms, _ = bench(lambda: engine.search(
            Xs, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
            policy=engine.POLICY_BEST, stop_eps=a.eps, deg=dg,
            shared_cache=True, exclusive=True, shared_bound=True))
        tot = sel_ms + srch_ms
        tfl = 2.0 * nq * P * d / (sel_ms * 1e-3) / 1e12
        print(f"{P:7d} {nq:5d} {sel_ms:10.3f} {srch_ms:10.3f} {tot:8.2f} "
              f"{100*sel_ms/tot:8.1f}% {tfl:11.2f}")
        del Qg
        torch.cuda.empty_cache()
    del catX, pool_t
    torch.cuda.empty_cache()
