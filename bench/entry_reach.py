"""Is there an entry near the answers we miss, and would we ever pick it?

A missed ground-truth document is almost always alone in its piece: no
reciprocal edge to any other answer, so reaching the other nine says nothing
about it. It has to be arrived at on its own, which means some entry's descent
has to start near it. Two things can go wrong, and they want opposite fixes.

  the pool has nothing near it        a denser catalogue
  the pool has something near it,     a larger V, or a selector that does not
  but we never select that entry      order purely by distance to the query

So: for each ground-truth document, find the nearest pool node, and report that
node's *rank* in the query's own ordering of the pool. If the rank is inside V
we already take it; if it sits at several hundred, that is exactly the entry the
answer needed and distance ordering put out of reach.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")[: a.nq]
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Pv = Xg[pt]                                                    # [P, d]

    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.05, 32, 24)
    seeds, _, _ = arms.seeds("distance", a.V)
    ids, _, st = engine.search(Xg, Gg, Qg, seeds.to(torch.int32).cuda(), k=a.k,
                               L=a.L, B=a.B, metric=metric, team_size=a.team,
                               stop_eps=a.stop_eps, shared_cache=True,
                               shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.zeros(gt.shape, bool)
    for q in range(len(gt)):
        s = set(int(v) for v in got[q] if v >= 0)
        hit[q] = [int(t) in s for t in gt[q]]

    with torch.no_grad():
        # query's ordering of the pool: rank 0 is the entry we always take first
        qp = (1.0 - Qg @ Pv.T) if ip else torch.cdist(Qg, Pv)
        qrank = qp.argsort(1).argsort(1)                           # [nq, P]
        gtv = Xg[torch.from_numpy(gt.astype(np.int64)).cuda().reshape(-1)]
        gp = (1.0 - gtv @ Pv.T) if ip else torch.cdist(gtv, Pv)    # [nq*k, P]
        near = gp.argmin(1).view(len(gt), a.k)                     # nearest pool node
        nd = gp.min(1).values.view(len(gt), a.k)
        rank_of_near = torch.gather(qrank, 1, near).cpu().numpy()
        nd = nd.cpu().numpy()
        del qp, gp, gtv

    sizes = np.zeros_like(frag)
    for q in range(len(frag)):
        c = np.bincount(frag[q]); sizes[q] = c[frag[q]]
    alone = sizes == 1
    print(f"{a.data.name} {a.graph}  V={a.V}  recall {hit.mean():.4f}  "
          f"{a.nq} queries, pool {a.P}\n")
    print(f"{'group':26} {'n':>8} {'rank of nearest entry':>22}")
    print(f"{'':26} {'':8} {'p50':>7} {'p90':>7} {'in V':>7}")
    rows = []
    for lab, m in (("found", hit), ("missed", ~hit),
                   ("missed & alone", (~hit) & alone),
                   ("found & alone", hit & alone)):
        if not m.any():
            continue
        r = rank_of_near[m]
        print(f"{lab:26} {int(m.sum()):8,} {np.percentile(r,50):7.0f} "
              f"{np.percentile(r,90):7.0f} {100*(r < a.V).mean():6.1f}%")
        rows.append(dict(group=lab, n=int(m.sum()), p50=float(np.percentile(r, 50)),
                         p90=float(np.percentile(r, 90)),
                         frac_in_V=float((r < a.V).mean())))
    print(f"\nV를 키웠을 때 '가장 가까운 엔트리'를 포함하게 되는 비율 (missed & alone):")
    r = rank_of_near[(~hit) & alone]
    for v in (32, 64, 128, 256, 512, 1024, 4096):
        print(f"   V={v:5}: {100*(r < v).mean():5.1f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, V=a.V,
                                         recall=float(hit.mean()), rows=rows), indent=2))


if __name__ == "__main__":
    main()
