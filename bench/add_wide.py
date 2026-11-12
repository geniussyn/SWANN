"""Keep the usual allocation and add a few wide entries on top of it.

Every earlier test of wide entries replaced near ones at a fixed V, and lost
monotonically. That is a different proposal from adding a handful: the near
entries are doing real work -- they concentrate on the piece that holds most of
the answers, which is why deduplicating them raises piece coverage and lowers
recall -- so taking them away was always going to cost. Adding instead costs
budget rather than coverage, and the honest comparison is against the same total
V spent entirely on near entries.

Small counts, because the proposal is a handful: the wide entries have to earn
their place against the 33rd, 34th, 36th nearest.
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
    ap.add_argument("--V", type=int, default=32, help="the near entries kept as they are")
    ap.add_argument("--add", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--depth", type=int, nargs="+", default=[128, 512, 2048])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
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
    deep = max(a.depth)
    order = select_a.select_nearest(Q, Vp, min(deep, a.P), metric=metric)
    g = torch.Generator(device="cuda").manual_seed(0)

    def run(cols):
        ids, _, st = engine.search(X, G, Q, pt[torch.gather(order, 1, cols)]
                                   .to(torch.int32), k=a.k, L=a.L, B=a.B,
                                   metric=metric, team_size=a.team,
                                   stop_eps=a.stop_eps, shared_cache=True,
                                   shared_bound=True, exclusive=True,
                                   elide_visited=True)
        got = ids.cpu().numpy()
        return (np.array([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                          for q in range(NQ)]),
                float(st.dist_evals.sum(1).float().mean()))

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return w, l, (1.0 if n == 0 else
                      min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    base, deb = run(torch.arange(a.V, device=X.device).view(1, -1).expand(NQ, -1).contiguous())
    print(f"{a.data.name}  {NQ} queries, catalogue {a.P:,}, eps={a.stop_eps}\n"
          f"{'entries':46} {'recall':>8} {'dist/q':>8} {'vs same total V':>16} {'p':>9}")
    print(f"{f'the nearest {a.V}':46} {base.mean():8.4f} {deb:8.0f}")
    rows = []
    for c in a.add:
        tot = a.V + c
        allnear, dan = run(torch.arange(tot, device=X.device).view(1, -1)
                           .expand(NQ, -1).contiguous())
        print(f"\n{f'the nearest {tot} (the reference for +{c})':46} "
              f"{allnear.mean():8.4f} {dan:8.0f}")
        for dep in a.depth:
            # c entries spread evenly over catalogue ranks [V, dep)
            pos = torch.linspace(a.V, min(dep, a.P) - 1, c, device=X.device).round().long()
            cols = torch.cat([torch.arange(a.V, device=X.device), pos]).view(1, -1).expand(NQ, -1).contiguous()
            pq, de = run(cols)
            w, l, p = sg(pq, allnear)
            print(f"{f'  + {c} spread over ranks {a.V}..{dep}':46} {pq.mean():8.4f} "
                  f"{de:8.0f} {pq.mean()-allnear.mean():+16.4f} {p:9.1e}")
            rows.append(dict(add=c, rule=f"spread{dep}", recall=float(pq.mean()),
                             delta=float(pq.mean() - allnear.mean()), p=float(p)))
        rnd = torch.randint(a.V, min(deep, a.P), (NQ, c), generator=g, device=X.device)
        cols = torch.cat([torch.arange(a.V, device=X.device).view(1, -1).expand(NQ, -1), rnd], 1)
        pq, de = run(cols.contiguous())
        w, l, p = sg(pq, allnear)
        print(f"{f'  + {c} at random ranks {a.V}..{deep}':46} {pq.mean():8.4f} "
              f"{de:8.0f} {pq.mean()-allnear.mean():+16.4f} {p:9.1e}")
        rows.append(dict(add=c, rule="random", recall=float(pq.mean()),
                         delta=float(pq.mean() - allnear.mean()), p=float(p)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, V=a.V, P=a.P, rows=rows),
                                    indent=2))


if __name__ == "__main__":
    main()
