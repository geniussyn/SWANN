"""Is recall a function of V, or of how many pieces the entries land in?

The design intent behind a large V is to spread the search across the semantic
subspaces the answers occupy, on the premise that landing in the right one is
enough -- and the premise holds: a missed answer is its own local maximum 85.8%
of the time, and seeding on the answers themselves returns recall 1.0000. But 32
entries chosen by distance land on only 12.4 distinct attractors, so raising V
buys both more coverage and more duplication, and the diminishing return could be
either running out of reachable pieces or piling entries into pieces already
held.

This separates them. For each query: the attractors its V entries climb to, the
attractors its true answers climb to, and what fraction of the latter the former
covers. If recall tracks that coverage rather than V, the lever is coverage, and
spending entries on it is not the same thing as spending them on V.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def climb(X, G, starts, Qv, steps=200, chunk=4096, ip=True):
    """Where a best-neighbour walk from each start ends, per query.

    starts is [n, m] of node ids, Qv is [n, d]; returns [n, m].
    """
    n, m = starts.shape
    cur = starts.reshape(-1).clone()
    qv = Qv.unsqueeze(1).expand(-1, m, -1).reshape(-1, Qv.shape[1])
    for _ in range(steps):
        nxt = torch.empty_like(cur)
        for s in range(0, cur.numel(), chunk):
            e = min(s + chunk, cur.numel())
            nb = G[cur[s:e]].long()
            if ip:
                sc = (X[nb.clamp(min=0)] * qv[s:e].unsqueeze(1)).sum(2)
                here = (X[cur[s:e]] * qv[s:e]).sum(1)
            else:                       # climb on -distance, so larger is nearer
                sc = -((X[nb.clamp(min=0)] - qv[s:e].unsqueeze(1)) ** 2).sum(2)
                here = -((X[cur[s:e]] - qv[s:e]) ** 2).sum(1)
            sn = torch.where(nb >= 0, sc, torch.full((1,), -1e30, device=X.device))
            bs, bi = sn.max(1)
            nxt[s:e] = torch.where(bs > here,
                                   torch.gather(nb, 1, bi.view(-1, 1)).view(-1),
                                   cur[s:e])
        if torch.equal(nxt, cur):
            break
        cur = nxt
    return cur.view(n, m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[4, 8, 16, 32, 64, 128])
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
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    nearQ = select_a.select_nearest(Q, Vp, max(a.V), metric=metric)

    gtt = torch.from_numpy(gt.astype(np.int64)).cuda()
    att_gt = climb(X, G, gtt, Q, ip=ip).cpu().numpy()                     # [NQ, k]
    att_en = climb(X, G, pt[nearQ], Q, ip=ip).cpu().numpy()               # [NQ, maxV]
    npieces = np.array([len(set(r)) for r in att_gt])
    print(f"{a.data.name} {a.graph}  {NQ} queries, catalogue {a.P:,}\n"
          f"pieces the answers occupy: mean {npieces.mean():.2f}, "
          f"p50 {np.percentile(npieces,50):.0f}, p90 {np.percentile(npieces,90):.0f}\n")

    print(f"{'V':>4} {'B':>5} {'recall':>8} {'entry':>7} {'answer pieces':>14} "
          f"{'recall of the':>14}")
    print(f"{'':4} {'':5} {'':8} {'pieces':>7} {'covered':>14} "
          f"{'covered ones':>14}")
    rows = []
    for V in a.V:
        Bx = max(1, a.total_B // V)
        ids, _, _ = engine.search(X, G, Q, pt[nearQ[:, :V]].to(torch.int32), k=a.k,
                                  L=a.L, B=Bx, metric=metric, team_size=a.team,
                                  stop_eps=a.stop_eps, shared_cache=True,
                                  shared_bound=True, exclusive=True,
                                  elide_visited=True)
        got = ids.cpu().numpy()
        rec = np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        ep, cov, rec_cov = [], [], []
        for q in range(NQ):
            ea = set(att_en[q, :V].tolist())
            ga = set(att_gt[q].tolist())
            ep.append(len(ea))
            cov.append(len(ea & ga) / len(ga))
            # recall restricted to answers whose piece an entry landed in
            inside = [i for i in range(a.k) if att_gt[q, i] in ea]
            if inside:
                found = sum(int(gt[q, i]) in {int(v) for v in got[q] if v >= 0}
                            for i in inside)
                rec_cov.append(found / len(inside))
        print(f"{V:4} {Bx:5} {rec:8.4f} {np.mean(ep):7.2f} {np.mean(cov):13.1%} "
              f"{np.mean(rec_cov):13.1%}")
        rows.append(dict(V=V, B=Bx, recall=float(rec), entry_pieces=float(np.mean(ep)),
                         coverage=float(np.mean(cov)),
                         recall_in_covered=float(np.mean(rec_cov))))
    print(f"\n'recall of the covered ones' is recall restricted to answers whose\n"
          f"piece at least one entry climbed into -- the design premise says this\n"
          f"should be near 1 at every V, and the gap to the recall column is what\n"
          f"coverage costs.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, P=a.P, nq=NQ,
                                         pieces_mean=float(npieces.mean()),
                                         rows=rows), indent=2))


if __name__ == "__main__":
    main()
