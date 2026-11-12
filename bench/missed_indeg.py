"""Are the ground-truth documents we miss the ones nothing points at?

The in-degree floor is worth building only if the misses concentrate on
low-in-degree nodes. Out-degree is what the prune spends; in-degree is what
decides whether a node can be arrived at, and our builder optimises the first
without ever looking at the second. This compares the in-degree of the GT a run
returned against the in-degree of the GT it missed, on the same run.
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
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    n = G.shape[0]
    ind = np.zeros(n, np.int32)
    for s in range(0, n, 200_000):
        e = min(s + 200_000, n); blk = G[s:e]; m = blk >= 0
        np.add.at(ind, blk[m].astype(np.int64), 1)

    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.05, 32, 24)
    seeds, _, _ = arms.seeds("distance", a.V)
    ids, _, st = engine.search(Xg, Gg, Qg, seeds.to(torch.int32).cuda(), k=a.k, L=a.L,
                               B=a.B, metric=metric, team_size=a.team,
                               stop_eps=a.stop_eps, shared_cache=True,
                               shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.zeros(gt.shape, bool)
    for q in range(len(gt)):
        s = set(int(v) for v in got[q] if v >= 0)
        hit[q] = [int(t) in s for t in gt[q]]
    gi = ind[gt]
    rec = hit.mean()
    print(f"{a.data.name} {a.graph} V={a.V} L={a.L} eps={a.stop_eps}  recall {rec:.4f}\n")
    print(f"{'입차수 구간':>14} {'GT 수':>9} {'회수율':>8}")
    edges = [0, 1, 8, 16, 32, 64, 1 << 30]
    rows = []
    for i in range(len(edges) - 1):
        m = (gi >= edges[i]) & (gi < edges[i + 1])
        if not m.any():
            continue
        lab = f"{edges[i]}-{edges[i+1]-1}" if edges[i + 1] < (1 << 30) else f"{edges[i]}+"
        print(f"{lab:>14} {int(m.sum()):9,} {hit[m].mean():8.4f}")
        rows.append(dict(lo=edges[i], hi=int(edges[i + 1]), n=int(m.sum()),
                         recall=float(hit[m].mean())))
    print(f"\n찾은 GT 입차수 중위 {int(np.median(gi[hit]))}   "
          f"놓친 GT 입차수 중위 {int(np.median(gi[~hit]))}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, graph=a.graph, recall=float(rec),
                                         median_in_hit=int(np.median(gi[hit])),
                                         median_in_miss=int(np.median(gi[~hit])),
                                         rows=rows), indent=2))


if __name__ == "__main__":
    main()
