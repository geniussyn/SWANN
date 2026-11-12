"""Hand each missed answer the entry that was closest to it, and see if a
descent from there arrives.

Placing entries deep in the query's pool ordering is strictly dominated: at a
fixed V, every entry diverted away from the nearest ranks costs recall and saves
distances, and the diverted configuration is beaten on both axes by a smaller V
of plain nearest entries. But the pool does hold a node near the answers we
miss -- median query-rank 139 -- so the question is why an entry placed there
does not arrive. Two things could be in the way:

  the descent itself      the entry is near the answer in the embedding but the
                          greedy walk from it is pulled elsewhere, so no
                          selector that picks that entry would help
  the coordination        the descent would arrive, but the shared bound carries
                          the k-th best distance the near branches already found,
                          and every candidate on the far branch is beyond it

This separates them. One branch per missed answer, seeded at the pool node
nearest to that answer, run alone with nothing else competing: no shared bound,
no claims, no rival branches. If it still fails to arrive, the ceiling is the
descent and entry placement is closed. If it arrives, the ceiling is ours.
"""
import argparse, json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402
from bench.entry_arms import Arms                                    # noqa: E402


def run(Xg, Gg, qv, seed, k, L, B, metric, team, eps, cache, bound, excl):
    st_seed = seed.view(-1, 1, 1).to(torch.int32).to(Xg.device)
    ids, _, st = engine.search(Xg, Gg, qv, st_seed, k=k, L=L, B=B, metric=metric,
                               team_size=team, stop_eps=eps, shared_cache=cache,
                               shared_bound=bound, exclusive=excl)
    return ids.cpu().numpy(), float(st.dist_evals.sum(1).float().mean())


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
    Pv = Xg[pt]

    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.05, 32, 24)
    seeds, _, _ = arms.seeds("distance", a.V)
    ids, _, _ = engine.search(Xg, Gg, Qg, seeds.to(torch.int32).cuda(), k=a.k,
                              L=a.L, B=a.B, metric=metric, team_size=a.team,
                              stop_eps=a.stop_eps, shared_cache=True,
                              shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])

    sizes = np.zeros_like(frag)
    for q in range(len(frag)):
        c = np.bincount(frag[q]); sizes[q] = c[frag[q]]
    sel = (~hit) & (sizes == 1)
    qi, ki = np.nonzero(sel)
    tgt = gt[qi, ki]
    print(f"{a.data.name} {a.graph}  baseline recall {hit.mean():.4f}  "
          f"{len(qi)} missed-and-alone answers in {a.nq} queries\n")

    with torch.no_grad():
        tv = Xg[torch.from_numpy(tgt.astype(np.int64)).cuda()]
        gp = (1.0 - tv @ Pv.T) if ip else torch.cdist(tv, Pv)
        near = gp.argmin(1)                                # pool slot nearest the answer
        del gp, tv
    ent = pt[near]                                         # the entry we hand it
    qv = Qg[torch.from_numpy(qi.astype(np.int64)).cuda()].contiguous()

    print(f"{'what the branch gets':38} {'arrives':>8} {'dist/q':>8}")
    rows = []
    trials = [
        ("shipped knobs, alone",            a.L, a.B, a.stop_eps),
        ("+ no eps gate",                   a.L, a.B, -1.0),
        ("+ wider queue (L=64)",            64,  a.B, -1.0),
        ("+ 8x budget (B=512)",             64,  512, -1.0),
        ("+ 32x budget (B=2048), L=128",    128, 2048, -1.0),
    ]
    for lab, L, B, eps in trials:
        out, de = run(Xg, Gg, qv, ent, a.k, L, B, metric, a.team, eps,
                      False, False, False)
        arr = np.array([t in set(int(v) for v in o if v >= 0) for t, o in zip(tgt, out)])
        print(f"{lab:38} {100*arr.mean():7.1f}% {de:8.1f}")
        rows.append(dict(trial=lab, L=L, B=B, stop_eps=eps,
                         arrives=float(arr.mean()), dist_per_q=de))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, graph=a.graph,
                                         baseline_recall=float(hit.mean()),
                                         n_pairs=int(len(qi)), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
