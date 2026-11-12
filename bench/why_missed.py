"""Why does a query we fail, fail?

All of the loss is arrival (a piece the search never touched), but "never
touched" has several causes and they want different fixes. Attribute them by
relaxing one knob at a time and asking which of the base run's missed
ground-truth documents come back:

    more entries        the piece was in nobody's basin
    looser eps          a lineage was retired on the way to it
    more budget         it ran out of steps
    no claims           another branch held the only route
    none of the above   structural: nothing reaches it

Also reported for the missed documents against the found ones: in-degree, and
the size of the piece they sit in. A missed document that nothing points at is
a build defect; a missed document alone in its piece is the fragmentation the
method exists for.
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
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")
    if a.nq:
        Q, gt, frag = Q[: a.nq], gt[: a.nq], frag[: a.nq]
    n = G.shape[0]
    ind = np.zeros(n, np.int32)
    for s in range(0, n, 200_000):
        e = min(s + 200_000, n); blk = G[s:e]; m = blk >= 0
        np.add.at(ind, blk[m].astype(np.int64), 1)

    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, "random4k", 4096)
    arms = Arms(Xg, Qg, pool, a.data, "random4k", 128, 0.05, 32, 24)

    def run(V, eps, B, excl):
        seeds, _, _ = arms.seeds("distance", V)
        ids, _, st = engine.search(Xg, Gg, Qg, seeds.to(torch.int32).cuda(), k=a.k,
                                   L=a.L, B=B, metric=metric, team_size=a.team,
                                   stop_eps=eps, shared_cache=True, shared_bound=True,
                                   exclusive=excl, elide_visited=excl)
        got = ids.cpu().numpy()
        hit = np.zeros(gt.shape, bool)
        for q in range(len(gt)):
            s = set(int(v) for v in got[q] if v >= 0)
            hit[q] = [int(t) in s for t in gt[q]]
        return hit

    base = run(a.V, a.stop_eps, a.B, True)
    miss = ~base
    nmiss = int(miss.sum())
    print(f"{a.data.name} {a.graph}  V={a.V} L={a.L} B={a.B} eps={a.stop_eps}")
    print(f"base recall {base.mean():.4f},  {nmiss:,} missed ground-truth "
          f"documents of {base.size:,}\n")

    arms_list = [("more entries  V x4", (a.V * 4, a.stop_eps, a.B, True)),
                 ("looser eps    0.5",  (a.V, 0.5, a.B, True)),
                 ("eps off",            (a.V, -1.0, a.B, True)),
                 ("more budget   B x4", (a.V, a.stop_eps, a.B * 4, True)),
                 ("no claims",          (a.V, a.stop_eps, a.B, False))]
    print(f"{'relaxation':22} {'recall':>8} {'of the misses recovered':>24}")
    rows, recovered_any = [], np.zeros_like(miss)
    for name, args in arms_list:
        h = run(*args)
        rec = (h & miss).sum() / max(nmiss, 1)
        recovered_any |= (h & miss)
        print(f"{name:22} {h.mean():8.4f} {rec:23.1%}")
        rows.append(dict(relaxation=name, recall=float(h.mean()), recovered=float(rec)))
    never = miss & ~recovered_any
    print(f"\n{'none of the above':22} {'':8} {never.sum()/max(nmiss,1):23.1%}  "
          f"({int(never.sum()):,} documents)")

    gi, gf = ind[gt], frag
    sizes = np.zeros_like(gf)
    for q in range(len(gf)):
        c = np.bincount(gf[q])
        sizes[q] = c[gf[q]]
    print(f"\n{'':22} {'found':>10} {'missed':>10} {'never recovered':>17}")
    for lab, arr in (("in-degree (median)", gi), ("piece size (median)", sizes)):
        print(f"{lab:22} {np.median(arr[base]):10.0f} {np.median(arr[miss]):10.0f} "
              f"{(np.median(arr[never]) if never.any() else float('nan')):17.0f}")
    print(f"{'in-degree < 16':22} {100*(gi[base]<16).mean():9.1f}% "
          f"{100*(gi[miss]<16).mean():9.1f}% "
          f"{(100*(gi[never]<16).mean() if never.any() else float('nan')):16.1f}%")
    print(f"{'alone in its piece':22} {100*(sizes[base]==1).mean():9.1f}% "
          f"{100*(sizes[miss]==1).mean():9.1f}% "
          f"{(100*(sizes[never]==1).mean() if never.any() else float('nan')):16.1f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, V=a.V, eps=a.stop_eps,
            base_recall=float(base.mean()), missed=int(nmiss),
            never_recovered=float(never.sum() / max(nmiss, 1)), rows=rows), indent=2))


if __name__ == "__main__":
    main()
