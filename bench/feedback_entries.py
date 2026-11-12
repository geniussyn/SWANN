"""Use a cheap first search to guess where the answers are, then seed on them.

Placing an entry beside each true answer recovers 80% of the remaining recall at
P=131072, and nothing computable from the query alone gets near it -- six
decompositions, all flat or worse. But the oracle's input is not the query, it is
the answers, and a small search already produces an approximation of those. So:
run a narrow search, take what it returns, and seed a second search from the
catalogue entries beside those results.

The comparison has to charge for both passes. A two-stage run at V1 + V2 is only
interesting if it beats a single pass given the same total work, so every row
reports distance evaluations summed over the passes it makes.
"""
import argparse, json, sys
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V1", type=int, nargs="+", default=[4, 8, 16])
    ap.add_argument("--V2", type=int, default=32)
    ap.add_argument("--single", type=int, nargs="+", default=[32, 48, 64])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest128k")
    ap.add_argument("--P", type=int, default=131072)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    nearQ = select_a.select_nearest(Q, Vp, max(a.single) + a.k, metric=metric)

    def search(seeds):
        ids, _, st = engine.search(X, G, Q, pt[seeds.clamp(min=0)].to(torch.int32),
                                   k=a.k, L=a.L, B=a.B, metric=metric,
                                   team_size=a.team, stop_eps=a.stop_eps,
                                   shared_cache=True, shared_bound=True,
                                   exclusive=True, elide_visited=True)
        return ids, float(st.dist_evals.sum(1).float().mean())

    def recall(ids):
        got = ids.cpu().numpy()
        return np.array([len(gts[q] & set(int(v) for v in got[q] if v >= 0)) / a.k
                         for q in range(NQ)])

    def entries_beside(docs):
        """Catalogue entries nearest the documents handed in."""
        d = docs.clamp(min=0).reshape(-1)
        v = X[d.long()]
        s = (v @ Vp.T) if ip else -torch.cdist(v, Vp)
        return s.argmax(1).view(docs.shape)

    def fill(primary, V):
        out = torch.cat([primary, nearQ], 1)
        k = out >= 0
        for j in range(1, out.shape[1]):
            k[:, j] &= (out[:, j:j+1] != out[:, :j]).all(1)
        rank = torch.where(k, torch.arange(out.shape[1], device=out.device).expand_as(out),
                           torch.full_like(out, 10**6))
        return torch.gather(out, 1, torch.argsort(rank, 1)[:, :V])

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    print(f"{a.data.name}  {NQ} queries, catalogue {a.P:,}, V2={a.V2}, eps={a.stop_eps}\n")
    print(f"{'arm':46} {'recall':>8} {'total ndis':>11}")
    singles = {}
    for V in a.single:
        ids, de = search(nearQ[:, :V])
        r = recall(ids); singles[V] = (r, de)
        print(f"{f'one pass, nearest {V}':46} {r.mean():8.4f} {de:11,.0f}")
    print()
    rows = []
    for V1 in a.V1:
        ids1, de1 = search(nearQ[:, :V1])
        r1 = recall(ids1)
        fb = entries_beside(ids1)                                  # [NQ, k]
        ids2, de2 = search(fill(fb, a.V2))
        r2 = recall(ids2)
        # the honest output of two passes is the union of what they returned
        u = ids1.cpu().numpy(); v = ids2.cpu().numpy()
        ru = np.array([len(gts[q] & ({int(x) for x in u[q] if x >= 0} |
                                     {int(x) for x in v[q] if x >= 0})) / a.k
                       for q in range(NQ)])
        print(f"{f'pass 1: nearest {V1}':46} {r1.mean():8.4f} {de1:11,.0f}")
        print(f"{f'pass 2: beside pass-1 results, V={a.V2}':46} {r2.mean():8.4f} "
              f"{de1+de2:11,.0f}")
        print(f"{'   both passes together':46} {ru.mean():8.4f} {de1+de2:11,.0f}")
        best = min(singles, key=lambda V: abs(singles[V][1] - (de1 + de2)))
        w, l, p = sg(ru, singles[best][0])
        print(f"{f'   vs one pass of {best} at {singles[best][1]:,.0f} ndis':46} "
              f"{ru.mean()-singles[best][0].mean():+8.4f} {'':>11}  p={p:.1e}\n")
        rows.append(dict(V1=V1, V2=a.V2, recall1=float(r1.mean()),
                         recall2=float(r2.mean()), recall_union=float(ru.mean()),
                         ndis=de1 + de2, vs_single=best,
                         delta=float(ru.mean() - singles[best][0].mean()), p=float(p)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, P=a.P, rows=rows,
                                         singles={str(k): [float(v[0].mean()), v[1]]
                                                  for k, v in singles.items()}), indent=2))


if __name__ == "__main__":
    main()
