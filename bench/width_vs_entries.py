"""Spend a fixed budget on queue width or on entries, and see which the corpus wants.

A query's pieces can be far apart or merely several. Where they are far, nothing
but a separate arrival reaches them and entries are the only lever. Where they
are close -- GIST has twice MS MARCO's piece count but only 2.3% of its
inter-piece pairs beyond mutual rank 20,000, against 9.4% -- a queue wide enough
to carry a temporarily worse candidate bridges them, which is what a single wide
queue does and what a wide queue is cheap at.

So the same total expansion budget is spent two ways along a line from many
narrow branches to few wide ones. If the corpus with the short tail prefers width
and the corpus with the long tail prefers entries, the tail is the instrument the
paper currently says it lacks.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["msmarco-1m", "gist-960", "glove-100"])
    ap.add_argument("--root", type=Path, default=Path("/dev/shm/gpuann"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--plan", nargs="+",
                    default=["32:12", "16:24", "8:48", "4:96", "2:192", "1:384"],
                    help="V:L pairs, budget per branch is total_B // V")
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    allrows = []
    for name in a.data:
        d = a.root / name
        man = json.loads((d / "manifest.json").read_text())
        ip = man["metric"].startswith("ip")
        metric = engine.METRIC_IP if ip else engine.METRIC_L2
        X = torch.from_numpy(np.load(d / "X.npy")).cuda()
        G = torch.from_numpy(np.load(d / a.graph)).cuda()
        Q = torch.from_numpy(np.load(d / "Q_test.npy")[: a.nq]).cuda()
        gt = np.load(d / "gt_test_ids.npy")[: a.nq, : a.k]
        NQ = Q.shape[0]
        gts = [set(int(t) for t in r) for r in gt]
        pool = load_pool(d, a.pool, a.P)
        pt = torch.from_numpy(pool.astype(np.int64)).cuda()
        Vp = select_a.pool_vectors(X, pt)
        near = select_a.select_nearest(Q, Vp, 1024, metric=metric)
        print(f"\n{name}  n={man['n']:,} d={X.shape[1]} metric={man['metric']}  "
              f"{NQ} queries, total budget {a.total_B}")
        print(f"  {'V':>4} {'L':>5} {'B':>5} {'recall':>8} {'dist/q':>8} "
              f"{'ms':>7} {'QPS':>9}")
        for spec in a.plan:
            V, L = (int(x) for x in spec.split(":"))
            Bx = max(1, a.total_B // V)
            best = None
            for r in range(3):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, _, st = engine.search(X, G, Q, pt[near[:, :V]].to(torch.int32),
                                           k=a.k, L=L, B=Bx, metric=metric,
                                           team_size=a.team, stop_eps=a.stop_eps,
                                           shared_cache=True, shared_bound=True,
                                           exclusive=True, elide_visited=True)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r: best = ms if best is None else min(best, ms)
            got = ids.cpu().numpy()
            rec = np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                           for q in range(NQ)])
            de = float(st.dist_evals.sum(1).float().mean())
            print(f"  {V:4} {L:5} {Bx:5} {rec:8.4f} {de:8.0f} {best:7.2f} "
                  f"{NQ/(best/1e3):9,.0f}")
            allrows.append(dict(corpus=name, V=V, L=L, B=Bx, recall=float(rec),
                                dist=de, ms=best, qps=NQ / (best / 1e3)))
        del X, G, Q, Vp, pt
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(allrows, indent=2))
    print(f"\nfor each corpus, the row with the best recall says whether the budget\n"
          f"belongs in entries or in queue width.")


if __name__ == "__main__":
    main()
