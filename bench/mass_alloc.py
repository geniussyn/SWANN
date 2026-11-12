"""Allocate V over the answers' pieces in proportion to their mass. Oracle bound.

bench/piece_mass.py closed one direction: spending j of V entries on pieces the
nearest entries did not reach moves recall by +-0.0004 for j in 1..4 and by
-0.0081 at j=V, and it is flat even on the most fragmented third of queries. But
that sweep only ever moved the budget OUTWARD, and it chose the extra pieces
blindly -- any unseen attractor, whether or not it holds an answer.

The allocation this tests is the one the mass numbers actually suggest: a query's
answers divide 71.5% / 86.4% between their largest one and two pieces, so give
each ANSWER-HOLDING piece a quota proportional to the answers it holds and fill
each quota with the nearest entries that reach that piece.

That needs the answers, so it is an oracle and a bound: no estimator of the mass
can beat it. Two arms separate the two things it knows.

  equal over GT pieces   quota V/|pieces| each. Isolates "the right pieces" from
                         piece_dedup's "any distinct piece".
  mass over GT pieces    quota proportional to the answers in the piece. This is
                         the proposal.

If neither beats nearest-V, the family is closed by arithmetic and the reason is
that proximity already allocates by mass, because the dominant piece is also the
nearest one. If one of them wins, the open question is estimating the mass, and
varphi_h is the index-free instrument for it.
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
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
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

    mass = []                                   # [(piece, count)] by count desc
    for q in range(NQ):
        u, c = np.unique(att_gt[q], return_counts=True)
        o = np.argsort(-c)
        mass.append(list(zip(u[o].tolist(), c[o].tolist())))

    print(f"{a.data.name} {a.split} {NQ} queries, pool {a.pool} P={a.P:,}, "
          f"dedup/alloc within the nearest {a.M}, k={a.k}\n")

    def quotas(q, V, weighted):
        ms = mass[q]
        if weighted:
            tot = sum(c for _, c in ms)
            raw = [V * c / tot for _, c in ms]
        else:
            raw = [V / len(ms)] * len(ms)
        qa = [int(np.floor(r)) for r in raw]
        # hand out the remainder by largest fractional part, mass order breaking ties
        rem = V - sum(qa)
        order = sorted(range(len(ms)), key=lambda i: -(raw[i] - np.floor(raw[i])))
        for i in range(rem):
            qa[order[i % len(ms)]] += 1
        return [(ms[i][0], qa[i]) for i in range(len(ms))]

    def alloc(V, weighted):
        out = np.empty((NQ, V), np.int64)
        served = np.zeros(NQ)
        for q in range(NQ):
            want = dict(quotas(q, V, weighted))
            pick, got = [], {A: 0 for A in want}
            for j in range(a.M):                        # nearest first
                A = att_en[q, j]
                if A in want and got[A] < want[A]:
                    got[A] += 1; pick.append(j)
                    if len(pick) == V:
                        break
            served[q] = (sum(c for A, c in mass[q] if got.get(A, 0) > 0)
                         / sum(c for _, c in mass[q]))
            for j in range(a.M):                        # top up with the nearest
                if len(pick) == V:
                    break
                if j not in pick:
                    pick.append(j)
            out[q] = pick[:V]
        return torch.from_numpy(out).cuda(), served.mean()

    def run(cols, V, Bx):
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
        cov, inmass = [], []
        for q in range(NQ):
            ea = set(att_en[q, c[q]].tolist()); ga = {A for A, _ in mass[q]}
            cov.append(len(ea & ga) / len(ga))
            # share of the V entries that land in an answer-holding piece
            inmass.append(np.mean([att_en[q, t] in ga for t in c[q]]))
        return (pq, np.mean(cov), np.mean(inmass),
                float(st.dist_evals.sum(1).float().mean()))

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    rows = []
    for V in a.V:
        Bx = max(1, a.total_B // V)
        base_cols = torch.arange(V, device=X.device).view(1, -1).expand(NQ, -1).contiguous()
        print(f"V={V}, B={Bx}")
        print(f"  {'allocation':34} {'recall':>8} {'cover':>7} {'in-mass':>8} "
              f"{'dist/q':>8} {'vs base':>9} {'p':>9}")
        base, cb, ib, deb = run(base_cols, V, Bx)
        print(f"  {'nearest-V':34} {base.mean():8.4f} {cb:7.1%} {ib:8.1%} {deb:8.0f}")
        rows.append(dict(V=V, arm="nearest", recall=float(base.mean()), cover=float(cb),
                         in_mass=float(ib), dist=deb, delta=0.0, served=None))
        for label, wt in [("equal over GT pieces", False),
                          ("mass over GT pieces", True)]:
            cols, srv = alloc(V, wt)
            pq, cov, inm, de = run(cols, V, Bx)
            w, l, p = sg(pq, base)
            print(f"  {label:34} {pq.mean():8.4f} {cov:7.1%} {inm:8.1%} {de:8.0f} "
                  f"{pq.mean()-base.mean():+9.4f} {p:9.1e}"
                  f"   (mass served {srv:.1%})")
            rows.append(dict(V=V, arm=label, recall=float(pq.mean()), cover=float(cov),
                             in_mass=float(inm), dist=de,
                             delta=float(pq.mean()-base.mean()), p=float(p),
                             served=float(srv)))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, M=a.M, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
