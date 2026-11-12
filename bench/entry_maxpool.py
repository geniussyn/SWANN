"""Score a pool node by its NEIGHBOURHOOD's best match, not by its own.

Proposition 1 (docs/SUBSPACE-ENTRIES.md): an ADDITIVE, LINEAR augmentation
`<q,x_p> + <W^T q, b_p>` equals `<q, x_p + W b_p>`, so it is entry placement in
the original d dims and placement is dead in ten arms. A **max** escapes that
proposition, because it is not linear in the document side:

    s(q, p) = max_{y in N(p)} <q, y>,     N(p) = p's R nearest corpus nodes, offline

That is the `1024 + D*N` proposal in the one form the proposition does not kill,
and it is the same asymmetry ColBERT/MUVERA late interaction use and that RaBitQ
uses here (4.8) -- the side that is expanded is never the query's.

Note what this arm is NOT. It does not spread entries over pieces; that family is
closed by bench/piece_mass.py -- a query's answers divide 71.5% / 86.4% between
their largest one and two pieces, 25.8% of queries have them all in one piece, and
spending j of V entries on unseen pieces moves recall by +-0.0004 for j in 1..4 and
by -0.0081 at j=V. Proximity and mass coincide, so nearest-V already allocates
near-optimally. This arm changes WHICH node is near, not how the budget is spread.

The motivation is a measurement, not an analogy. The oracle entry arm wins
1.2-1.9x at matched recall (section 1) and it is defined as *the pool node
nearest each answer*. Coverage-maximising selection does the opposite and loses:
bench/piece_dedup.py takes distinct pieces from 12.36 to 31.36 and GT-piece
coverage from 85.5% to 92.1% at V=32, on 16% fewer distances, and recall falls
0.9852 -> 0.9771 (p=5e-23). So what the oracle has is not coverage, it is
**proximity to an answer** -- and max-pooling computes exactly that forward: p
scores well when p has a neighbour that scores well.

Cost, and the baseline that makes it honest. The scan touches |union N(p)| <=
P*R vectors, so at P=4096, R=32 it costs what a P=131072 catalogue costs -- which
this lab has already measured at 3.5-6.8 ms for Q=2000 and 1.13-1.24x at matched
recall on msmarco. So the arm to beat is not the P=4096 scan, it is **nest128k's
plain nearest selection at the same scan cost**, and that arm is here.
"""
import argparse, json, sys, time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def neighbourhoods(X, Vp, R, chunk=512):
    """[P, R] -- each pool node's R nearest corpus nodes. Offline, query-free."""
    out = []
    for s in range(0, Vp.shape[0], chunk):
        sc = Vp[s:s + chunk] @ X.T
        out.append(torch.topk(sc, R, dim=1).indices)
        del sc
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--R", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--dense-pool", default="nest128k")
    ap.add_argument("--dense-P", type=int, default=131072)
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
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q).cpu().numpy()

    print(f"{a.data.name} {a.split} {NQ} queries, pool {a.pool} P={a.P:,}, "
          f"k={a.k}, total_B={a.total_B}\n")

    def search(seed_ids, V, Bx):
        ids, _, st = engine.search(X, G, Q, seed_ids.to(torch.int32), k=a.k, L=a.L,
                                   B=Bx, metric=metric, team_size=a.team,
                                   stop_eps=a.stop_eps, shared_cache=True,
                                   shared_bound=True, exclusive=True,
                                   elide_visited=True)
        got = ids.cpu().numpy()
        pq = np.array([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        return pq, float(st.dist_evals.sum(1).float().mean())

    def pieces_cov(seed_ids):
        att = climb(X, G, seed_ids.to(torch.int64), Q).cpu().numpy()
        ep, cov = [], []
        for q in range(NQ):
            ea = set(att[q].tolist()); ga = set(att_gt[q].tolist())
            ep.append(len(ea)); cov.append(len(ea & ga) / len(ga))
        return np.mean(ep), np.mean(cov)

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    # ---- the candidate scores, one per arm ---------------------------------
    scores = {}
    t0 = time.perf_counter()
    with select_a.reduced_precision(True):
        scores["near P=%d" % a.P] = ((Q @ Vp.T), pt)
    torch.cuda.synchronize()
    print(f"  plain scan P={a.P:,}: {(time.perf_counter()-t0)*1e3:.1f} ms")

    dpool = load_pool(a.data, a.dense_pool, a.dense_P)
    dpt = torch.from_numpy(dpool.astype(np.int64)).cuda()
    dVp = select_a.pool_vectors(X, dpt)
    t0 = time.perf_counter()
    with select_a.reduced_precision(True):
        sd = torch.empty(NQ, a.dense_P, device=X.device)
        for s in range(0, NQ, 512):
            sd[s:s + 512] = Q[s:s + 512] @ dVp.T
    torch.cuda.synchronize()
    print(f"  dense scan P={a.dense_P:,}: {(time.perf_counter()-t0)*1e3:.1f} ms "
          f"(the cost-matched baseline)")
    scores["near P=%d" % a.dense_P] = (sd, dpt)
    del dVp

    for R in a.R:
        nbr = neighbourhoods(X, Vp, R)                               # [P, R]
        NV = X[nbr.reshape(-1)]                                      # gathered once
        t0 = time.perf_counter()
        sm = torch.empty(NQ, a.P, device=X.device)
        am = torch.empty(NQ, a.P, dtype=torch.long, device=X.device)
        with select_a.reduced_precision(True):
            for s in range(0, NQ, 256):
                e = min(s + 256, NQ)
                blk = (Q[s:e] @ NV.T).view(e - s, a.P, R)
                sm[s:e], am[s:e] = blk.max(2)
        torch.cuda.synchronize()
        print(f"  max-pool R={R} (touches {a.P*R:,}): "
              f"{(time.perf_counter()-t0)*1e3:.1f} ms")
        del NV
        scores[f"maxpool R={R}"] = (sm, pt)
        # entering at the neighbour that carried the max, not at the pool node
        scores[f"maxpool R={R}, enter at argmax"] = (sm, (nbr, am))
    print()

    rows = []
    for V in a.V:
        Bx = max(1, a.total_B // V)
        print(f"V={V}, B={Bx}")
        print(f"  {'selector':38} {'recall':>8} {'pieces':>7} {'cover':>7} "
              f"{'dist/q':>8} {'vs base':>9} {'p':>9}")
        base = None
        for name, (sc, tbl) in scores.items():
            top = torch.topk(sc, V, dim=1).indices
            if isinstance(tbl, tuple):
                # enter at the neighbour that carried the max, not at the pool node
                nbr, am = tbl
                seed = nbr[top].gather(2, am.gather(1, top).unsqueeze(2)).squeeze(2)
            else:
                seed = tbl[top]
            pq, de = search(seed, V, Bx)
            ep, cov = pieces_cov(seed)
            if base is None:
                base = pq
                print(f"  {name:38} {pq.mean():8.4f} {ep:7.2f} {cov:7.1%} {de:8.0f}")
            else:
                w, l, p = sg(pq, base)
                print(f"  {name:38} {pq.mean():8.4f} {ep:7.2f} {cov:7.1%} {de:8.0f} "
                      f"{pq.mean()-base.mean():+9.4f} {p:9.1e}")
            rows.append(dict(V=V, B=Bx, arm=name, recall=float(pq.mean()),
                             pieces=float(ep), coverage=float(cov), dist=de,
                             delta=float(pq.mean()-base.mean())))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, R=a.R, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
