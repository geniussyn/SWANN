"""How much of a dense catalogue's recall is already in the entries themselves?

Scoring 131,072 of a million nodes is a flat scan over 13% of the corpus, so the
obvious objection to the result is that the recall is coming from the scan rather
than from the search it seeds: if the selected entries already contain the
answers, the graph is decoration. This measures that directly -- recall@10 of the
V selected entries, before a single expansion -- for both catalogues, and
alongside it the share of the final answers that were entries.

If the entries carry little of the recall, the scan is doing what it is supposed
to do: placing the search near the answers, not finding them.
"""
import argparse, json, sys
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
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--pools", nargs="+", default=["nest4k:4096", "nest128k:131072"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")[: a.nq]
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    gts = [set(int(t) for t in row) for row in gt]

    print(f"{a.data.name} {a.graph}  V={a.V} k={a.k}  {len(Q)} queries\n")
    print(f"{'catalogue':12} {'entries are':>12} {'search':>8} {'answers that':>13}")
    print(f"{'':12} {'of the top-k':>12} {'recall':>8} {'were entries':>13}")
    rows = []
    for spec in a.pools:
        name, P = spec.split(":"); P = int(P)
        pool = load_pool(a.data, name, P)
        pt = torch.from_numpy(pool.astype(np.int64)).cuda()
        Vp = select_a.pool_vectors(Xg, pt)
        near = select_a.select_nearest(Qg, Vp, a.V, metric=metric)
        seeds = pt[near]
        sarr = seeds.cpu().numpy()
        # recall the entries already hold, with no expansion at all
        pre = np.mean([len(gts[q] & set(int(v) for v in sarr[q])) / a.k
                       for q in range(len(Q))])
        ids, _, _ = engine.search(Xg, Gg, Qg, seeds.to(torch.int32), k=a.k, L=a.L,
                                  B=a.B, metric=metric, team_size=a.team,
                                  stop_eps=a.stop_eps, shared_cache=True,
                                  shared_bound=True, exclusive=True,
                                  elide_visited=True)
        got = ids.cpu().numpy()
        rec = np.mean([len(gts[q] & set(int(v) for v in got[q] if v >= 0)) / a.k
                       for q in range(len(Q))])
        # of the answers we did return, how many came for free as entries
        free = np.mean([len(gts[q] & set(int(v) for v in got[q] if v >= 0)
                            & set(int(v) for v in sarr[q])) /
                        max(len(gts[q] & set(int(v) for v in got[q] if v >= 0)), 1)
                        for q in range(len(Q))])
        print(f"{name:12} {pre:11.4f} {rec:8.4f} {100*free:12.1f}%")
        rows.append(dict(pool=name, P=P, entry_recall=float(pre),
                         search_recall=float(rec), answers_that_were_entries=float(free)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, V=a.V, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
