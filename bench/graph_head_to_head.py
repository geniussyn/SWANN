"""Several graphs, one kernel: which builder's graph is the better graph?

The question this answers is narrower than the recall-QPS tables and is the one
that decides whether the *builder* needs work. Everything but the graph is held
fixed -- same corpus, same queries, same entry pool, same budget, same search
kernel -- so a difference is the graph and nothing else. Native CAGRA is not run
here on purpose: comparing against its kernel conflates the two, and
`bench/compare.py` already does that comparison.

Each graph is read off its own frontier rather than at one point, because the
binding knob is not the same for every graph: at V=32 recall saturates by L=64
and by B=128, and what moves it after that is `eps`, the retirement slack.

    python bench/graph_head_to_head.py --data /dev/shm/gpuann/msmarco-1m \
        --graphs G_ptf32.npy G_cagra_d64.npy --epss 0.05 0.15 0.30
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graphs", nargs="+", required=True,
                    help="graph files inside --data, each searched identically")
    ap.add_argument("--epss", type=float, nargs="+", default=[0.05, 0.15, 0.30],
                    help="retirement slack: the knob that still moves recall once "
                         "L and B have saturated")
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--B", type=int, default=512)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--deg", action="store_true",
                    help="bound each expansion by the node's own degree "
                         "(engine.graph_degrees); the answer is unchanged")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Q = np.ascontiguousarray(np.load(a.data / f"Q_{a.split}.npy")[: a.nq])
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pool_t = torch.from_numpy(pool).cuda()
    catX = select_a.pool_vectors(Xg, pool_t)
    near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
    seeds = pool_t[near].unsqueeze(1).contiguous()          # [Q, 1, V], shared state

    print(f"{man['corpus']} n={man['n']} d={man['d']}  {a.split} q={len(Q)}  "
          f"V={a.V} L={a.L} B={a.B} shared+rr+cache+bound"
          + ("  deg-bounded" if a.deg else "") + "\n")
    print(f"{'graph':18s} {'W':>4s} {'deg':>5s} {'eps':>5s} {'recall@10':>10s} "
          f"{'dist/q':>8s} {'edge/q':>8s} {'ms':>7s} {'QPS':>8s}")
    rows = []
    for gname in a.graphs:
        G0 = np.load(a.data / gname)
        real = (G0 >= 0).sum(1)
        Gg = torch.from_numpy(np.ascontiguousarray(G0)).cuda()
        dg = engine.graph_degrees(Gg) if a.deg else None
        for eps in a.epss:
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(
                    Xg, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_ROUND_ROBIN, stop_eps=eps,
                    shared_cache=True, shared_bound=True, deg=dg)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (best is None or ms < best[0]):
                    best = (ms, ids, st)
            ms, ids, st = best
            rc = recall_at_k(ids.cpu().numpy(), gt)
            dq = float(st.dist_evals.sum()) / len(Q)
            eq = float(st.expansions.sum()) / len(Q) * real.mean()
            rows.append(dict(graph=gname, W=int(G0.shape[1]), deg=float(real.mean()),
                             eps=eps, recall=rc, dist=dq, edges=eq, ms=ms,
                             qps=1e3 * len(Q) / ms))
            print(f"{gname:18s} {G0.shape[1]:4d} {real.mean():5.1f} {eps:5.2f} "
                  f"{rc:10.4f} {dq:8.1f} {eq:8.0f} {ms:7.2f} {1e3*len(Q)/ms:8.0f}")
        del Gg
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, nq=len(Q),
                                         V=a.V, L=a.L, B=a.B, pool=a.pool, P=a.P,
                                         deg_bounded=bool(a.deg), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
