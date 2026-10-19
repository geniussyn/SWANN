"""The regime the batch benchmarks never entered: one query at a time.

Everything measured so far ran 3490 queries at once, where the GPU is saturated
and an extra branch costs its own work in wall time. A single query does not
saturate anything: one CTA per branch means E=100 branches occupy 100 of 188
SMs, and they run at the same time as one. So the question "is wide fan-out
free" has two different answers depending on Q, and only one of them has been
measured.

Reports per-query latency at several batch sizes, so the crossover is visible
rather than assumed, and records the occupancy side of it (Q*E CTAs against the
SM count).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402
from bench.entry_arms import Arms, recall_at_k                      # noqa: E402


def timed(fn, repeats, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t) * 1e3)
    return float(np.median(ts)), float(np.percentile(ts, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--graph", default="G_lid.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--Q", type=int, nargs="+", default=[1, 8, 64, 256, 1024])
    ap.add_argument("--configs", nargs="+",
                    default=["shared:distance:8:128:512", "branch:distance:100:16:16"],
                    help="fanout:arm:E:m:B")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--stop-eps", type=float, default=None,
                    help="retirement slack; default 0.15 for shared, off for branch")
    ap.add_argument("--exclusive", action="store_true",
                    help="a node claimed by one branch is off limits to the others")
    ap.add_argument("--bound", action="store_true",
                    help="share the query's k-th best distance across its CTAs")
    ap.add_argument("--cache", action="store_true",
                    help="share one distance table per query across its branch CTAs")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Qa = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Xs = Xg.half().contiguous() if a.half else Xg
    pool = load_pool(a.data, a.pool, a.P)
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"{a.graph}  {torch.cuda.get_device_name(0)}  {sm} SMs  "
          f"X={'fp16' if a.half else 'fp32'}\n")

    rows = []
    print(f"{'config':32s} {'Q':>5s} {'CTAs':>6s} {'recall@10':>10s} "
          f"{'p50 ms':>8s} {'p95 ms':>8s} {'ms/query':>9s}")
    for cfg in a.configs:
        fan, arm, E, m, B = cfg.split(":")
        E, m, B = int(E), int(m), int(B)
        for nq in a.Q:
            Qg = torch.from_numpy(np.ascontiguousarray(Qa[:nq])).cuda()
            arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.1, 32, 24)
            seeds, _, _ = arms.seeds(arm, E)
            if fan == "shared":
                seeds = seeds.unsqueeze(1)
                policy = engine.POLICY_ROUND_ROBIN
                eps = 0.15 if a.stop_eps is None else a.stop_eps
            else:
                policy = engine.POLICY_BEST
                eps = -1.0 if a.stop_eps is None else a.stop_eps
            out = {}

            def run():
                out["r"] = engine.search(Xs, Gg, Qg, seeds, k=a.k, m=m, budget=B,
                                         team_size=16, policy=policy, stop_eps=eps,
                                         shared_cache=a.cache, shared_bound=a.bound,
                                         exclusive=a.exclusive)
            p50, p95 = timed(run, a.repeats)
            ids = out["r"][0]
            rec = recall_at_k(ids.cpu().numpy(), gt[:nq])
            ctas = nq * (1 if fan == "shared" else E)
            name = f"{fan} E={E} m={m} B={B}"
            print(f"{name:32s} {nq:5d} {ctas:6d} {rec:10.4f} {p50:8.2f} {p95:8.2f} "
                  f"{p50/nq:9.3f}")
            rows.append(dict(config=cfg, Q=nq, ctas=ctas, recall=rec, p50=p50, p95=p95,
                             ms_per_query=p50 / nq))
        print()

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(graph=a.graph, sms=sm, half=a.half, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
