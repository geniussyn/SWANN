"""How much of the GPU does a static one-CTA-per-branch schedule waste?

Each branch runs until its own budget or its own eps retirement, so branches of
one batch finish at different times. Whether that costs anything depends on how
many waves of CTAs the batch produces: with many waves the hardware scheduler
backfills a finished CTA immediately and the imbalance averages out; with one
partial wave -- the low-latency regime, a handful of queries -- a finished CTA's
slot stays idle until the whole kernel drains.

Reported per batch size: the expansion distribution across branches, the
utilisation a static schedule can reach (mean/max within a wave), and the wave
count. The gap to 1.0 at small Q is what a persistent kernel with a task queue
(Atos-style) could recover; at large Q there is nothing to recover.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402
from bench.entry_arms import Arms                                   # noqa: E402

SMS, BLOCK, THREADS_PER_SM = 188, 128, 1536


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--Q", type=int, nargs="+", default=[1, 4, 16, 64, 256, 1024, 3490])
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Qa = np.load(a.data / "Q_test.npy")
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    pool = load_pool(a.data, "random4k", 4096)
    cap = SMS * min(THREADS_PER_SM // BLOCK, 12)
    print(f"{a.data.name} {a.graph} V={a.V} L={a.L} eps={a.stop_eps}   "
          f"동시 CTA 한도 {cap}\n")
    print(f"{'Q':>6} {'CTA':>9} {'웨이브':>7} {'확장 평균':>9} {'p95':>6} {'최대':>6} "
          f"{'정적 활용률':>11} {'회수 여지':>9}")
    rows = []
    for Q in a.Q:
        if Q > len(Qa):
            continue
        Qg = torch.from_numpy(Qa[:Q]).cuda()
        arms = Arms(Xg, Qg, pool, a.data, "random4k", 128, 0.05, 32, 24)
        seeds, _, _ = arms.seeds("distance", a.V)
        st = engine.init_states(Xg, Qg, seeds.to(torch.int32).cuda(), L=a.L, B=a.B,
                                D=G.shape[1], metric=metric, team_size=a.team,
                                stop_eps=a.stop_eps, kstop=10, shared_cache=True,
                                shared_bound=True, exclusive=True, elide_visited=True)
        engine.advance(Xg, Gg, Qg, st, a.B)
        e = st.expansions.cpu().numpy().reshape(-1).astype(np.float64)
        ctas = Q * a.V
        waves = ctas / cap
        # a static schedule's wall time is set by the slowest CTA in each wave;
        # with w full waves the expected utilisation is mean / max-of-a-wave
        per_wave = max(1, min(cap, ctas))
        if ctas <= cap:
            util = e.mean() / max(e.max(), 1e-9)
        else:
            # sample wave-sized groups to estimate the mean of per-wave maxima
            rng = np.random.default_rng(0)
            mx = [e[rng.choice(len(e), per_wave, replace=False)].max()
                  for _ in range(32)]
            util = e.mean() / max(float(np.mean(mx)), 1e-9)
        print(f"{Q:6} {ctas:9,} {waves:7.2f} {e.mean():9.1f} "
              f"{np.percentile(e,95):6.0f} {e.max():6.0f} {util:10.1%} {1-util:8.1%}")
        rows.append(dict(Q=Q, ctas=int(ctas), waves=float(waves), mean=float(e.mean()),
                         p95=float(np.percentile(e, 95)), max=float(e.max()),
                         util=float(util)))
        del st
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, graph=a.graph, V=a.V,
                                         cap=cap, rows=rows), indent=2))


if __name__ == "__main__":
    main()
