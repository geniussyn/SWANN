"""Pieces are unequal, so spend entries in proportion to their mass.

bench/piece_dedup.py buys piece coverage and loses recall: at V=32 it takes the
distinct attractors its entries reach from 12.36 to 31.36 and the share of the
answers' own pieces it covers from 85.5% to 92.1%, on 16% fewer distance
evaluations, and recall falls 0.9852 -> 0.9771 (p=5e-23). Coverage counts pieces
equally; recall counts answers. A query's 10 answers occupy ~3.18 pieces, so the
pieces are unequal, and an entry diverted to a piece holding one answer is taken
from the piece holding the rest.

Two things here.

  the mass        how the k answers actually divide between their pieces. If the
                  largest piece holds most of them, the optimal allocation is
                  nearly all-nearest and dedup is refuted by arithmetic.
  the tail arm    keep the V-j nearest entries and spend only the last j on
                  unseen pieces. j=0 is nearest-V, j=V is piece_dedup. If the
                  mass is skewed the best j is small and positive, and that is
                  the only allocation with a chance.
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
    ap.add_argument("--j", type=int, nargs="+", default=[0, 1, 2, 4])
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
    att_en = climb(X, G, pt[nearQ], Q).cpu().numpy()
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q).cpu().numpy()

    # ---- the mass ----------------------------------------------------------
    sizes = []
    for q in range(NQ):
        _, cnt = np.unique(att_gt[q], return_counts=True)
        sizes.append(np.sort(cnt)[::-1])
    npieces = np.array([len(s) for s in sizes])
    top1 = np.array([s[0] for s in sizes]) / a.k
    top2 = np.array([s[:2].sum() for s in sizes]) / a.k
    print(f"{a.data.name} {a.split} {NQ} queries, k={a.k}, catalogue {a.P:,}\n")
    print("how the k answers divide between their pieces:")
    print(f"  pieces per query          mean {npieces.mean():.2f}  "
          f"median {np.median(npieces):.0f}  max {npieces.max()}")
    print(f"  largest piece's share     mean {top1.mean():.1%}  "
          f"median {np.median(top1):.0%}")
    print(f"  largest two pieces' share mean {top2.mean():.1%}")
    print(f"  queries whose answers are all in one piece   {(npieces==1).mean():.1%}")
    print(f"  queries with >= 4 pieces                     {(npieces>=4).mean():.1%}\n")

    def seeds(V, j):
        """V-j nearest, then j from pieces none of those already reached."""
        out = np.empty((NQ, V), np.int64)
        for q in range(NQ):
            pick = list(range(V - j))
            seen = set(att_en[q, :V - j].tolist())
            for t in range(V - j, a.M):
                if len(pick) == V:
                    break
                A = att_en[q, t]
                if A not in seen:
                    seen.add(A); pick.append(t)
            for t in range(V - j, a.M):              # top up if pieces ran out
                if len(pick) == V:
                    break
                if t not in pick:
                    pick.append(t)
            out[q] = pick[:V]
        return torch.from_numpy(out).cuda()

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
        ep, cov = [], []
        for q in range(NQ):
            ea = set(att_en[q, c[q]].tolist()); ga = set(att_gt[q].tolist())
            ep.append(len(ea)); cov.append(len(ea & ga) / len(ga))
        return pq, np.mean(ep), np.mean(cov), float(st.dist_evals.sum(1).float().mean())

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    strata = {"1 piece": npieces == 1, "2-3 pieces": (npieces >= 2) & (npieces <= 3),
              ">=4 pieces": npieces >= 4}
    rows = []
    per_q = {}
    for V in a.V:
        Bx = max(1, a.total_B // V)
        print(f"V={V}, B={Bx}")
        print(f"  {'j (entries spent on unseen pieces)':36} {'recall':>8} "
              f"{'pieces':>8} {'coverage':>9} {'dist/q':>8} {'vs j=0':>9} {'p':>9}")
        base = None
        for j in a.j:
            if j > V:
                continue
            pq, ep, cov, de = run(seeds(V, j), V, Bx)
            per_q[(V, j)] = pq
            if base is None:
                base = pq
                print(f"  {j:<36d} {pq.mean():8.4f} {ep:8.2f} {cov:9.1%} {de:8.0f}")
            else:
                w, l, p = sg(pq, base)
                print(f"  {j:<36d} {pq.mean():8.4f} {ep:8.2f} {cov:9.1%} {de:8.0f} "
                      f"{pq.mean()-base.mean():+9.4f} {p:9.1e}")
            rows.append(dict(V=V, j=j, B=Bx, recall=float(pq.mean()), pieces=float(ep),
                             coverage=float(cov), dist=de,
                             delta=float(pq.mean()-base.mean())))
        print()
        # the same deltas, split by how fragmented the query's answers are
        print(f"  {'by fragmentation':36} " + "".join(f"{f'j={j}':>10}" for j in a.j if j <= V))
        for sname, mask in strata.items():
            cells = []
            for j in a.j:
                if j > V:
                    continue
                if j == 0:
                    cells.append(f"{per_q[(V,0)][mask].mean():10.4f}")
                else:
                    d = per_q[(V, j)][mask] - per_q[(V, 0)][mask]
                    cells.append(f"{d.mean():+10.4f}")
            print(f"  {sname + f' (n={int(mask.sum())})':36} " + "".join(cells))
            rows.append(dict(V=V, stratum=sname, n=int(mask.sum()),
                             base=float(per_q[(V,0)][mask].mean()),
                             deltas={str(j): float((per_q[(V,j)][mask]
                                                    - per_q[(V,0)][mask]).mean())
                                     for j in a.j if j <= V and j != 0}))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, split=a.split, nq=NQ, pool=a.pool, P=a.P, M=a.M,
            L=a.L, total_B=a.total_B, eps=a.stop_eps,
            mass=dict(pieces_mean=float(npieces.mean()),
                      top1_share=float(top1.mean()), top2_share=float(top2.mean()),
                      one_piece=float((npieces == 1).mean()),
                      ge4_pieces=float((npieces >= 4).mean())),
            rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
