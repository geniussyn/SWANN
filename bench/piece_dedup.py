"""Spend V on distinct pieces instead of on distinct entries.

Recall restricted to answers whose piece an entry landed in is 99.2-99.7% at
every V, so the premise behind a large V holds: landing in the subspace is
enough, the greedy finishes. What a large V does not do is land in many
subspaces. The answers of a query occupy 3.18 pieces on average, and 32 entries
chosen by distance land on 13.59 -- two thirds of them arrive where another entry
already is, and the pieces they do reach are mostly not the ones holding answers.

So the test is whether coverage can be bought at a fixed V without moving the
entries away from the query: among the nearest M entries, keep the first of each
distinct attractor. Earlier attempts failed on both counts at once -- they used
pool-graph basins as a stand-in for the real attractors, and they filled slots
from catalogue ranks in the thousands. This uses the attractors the corpus graph
actually has, and never looks past the nearest M.
"""
import argparse, json, sys
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--M", type=int, default=256,
                    help="how far down the query's own ordering dedup may look")
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip")
    metric = engine.METRIC_IP
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    nearQ = select_a.select_nearest(Q, Vp, a.M, metric=metric)
    att_en = climb(X, G, pt[nearQ], Q).cpu().numpy()                 # [NQ, M]
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q).cpu().numpy()
    print(f"{a.data.name}  {NQ} queries, catalogue {a.P:,}, dedup within the "
          f"nearest {a.M}\n")

    def dedup(V):
        """First entry of each distinct attractor, nearest first, then the rest."""
        out = np.empty((NQ, V), np.int64)
        for q in range(NQ):
            seen, pick = set(), []
            for j in range(a.M):
                A = att_en[q, j]
                if A not in seen:
                    seen.add(A); pick.append(j)
                    if len(pick) == V:
                        break
            if len(pick) < V:                      # top up with the nearest unused
                for j in range(a.M):
                    if j not in pick:
                        pick.append(j)
                        if len(pick) == V:
                            break
            out[q] = pick[:V]
        return torch.from_numpy(out).cuda()

    def report(label, cols, V, Bx):
        sel = torch.gather(nearQ, 1, cols)
        ids, _, st = engine.search(X, G, Q, pt[sel].to(torch.int32), k=a.k, L=a.L,
                                   B=Bx, metric=metric, team_size=a.team,
                                   stop_eps=a.stop_eps, shared_cache=True,
                                   shared_bound=True, exclusive=True,
                                   elide_visited=True)
        got = ids.cpu().numpy()
        pq = np.array([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        c = cols.cpu().numpy()
        ep, cov = [], []
        for q in range(NQ):
            ea = set(att_en[q, c[q]].tolist()); ga = set(att_gt[q].tolist())
            ep.append(len(ea)); cov.append(len(ea & ga) / len(ga))
        print(f"  {label:34} {pq.mean():8.4f} {np.mean(ep):8.2f} "
              f"{np.mean(cov):10.1%} {float(st.dist_evals.sum(1).float().mean()):9.0f}")
        return pq, np.mean(cov)

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    rows = []
    for V in a.V:
        Bx = max(1, a.total_B // V)
        print(f"V={V}, B={Bx}")
        print(f"  {'entries':34} {'recall':>8} {'pieces':>8} {'coverage':>10} "
              f"{'dist/q':>9}")
        base, cb = report("nearest to the query", 
                          torch.arange(V, device=X.device).view(1, -1).expand(NQ, -1).contiguous(), V, Bx)
        dd, cd = report(f"one per piece, within nearest {a.M}", dedup(V), V, Bx)
        w, l, p = sg(dd, base)
        print(f"  {'':34} {dd.mean()-base.mean():+8.4f} {'':8} "
              f"{cd-cb:+9.1%}   better {w} worse {l}  p={p:.1e}\n")
        rows.append(dict(V=V, B=Bx, recall_near=float(base.mean()),
                         recall_dedup=float(dd.mean()), cov_near=float(cb),
                         cov_dedup=float(cd), p=float(p)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, M=a.M, rows=rows), indent=2))


if __name__ == "__main__":
    main()
