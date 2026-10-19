"""Where a search's time goes, by dimension.

Two budgets, differenced, so the per-expansion and per-distance costs come out
without any setup in them. Then the per-distance time is compared against what
reading the candidate vector alone would cost at HBM speed. The gap is
everything else the kernel does -- two hash probes per neighbour, the candidate
sort, the merge, the queue bookkeeping -- and whether that gap is visible is the
whole story of which corpora this engine wins on.
"""
import argparse, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402
from bench.entry_arms import Arms                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", action="append", required=True, help="name:path")
    ap.add_argument("--graph", default="G_lid.npy")
    ap.add_argument("--hbm", type=float, default=1.8e12, help="bytes/s of the part")
    ap.add_argument("--m", type=int, default=128)
    ap.add_argument("--budgets", type=int, nargs=2, default=[4, 32])
    a = ap.parse_args()

    for spec in a.data:
        name, d = spec.split(":", 1)
        X = np.load(Path(d) / "X.npy"); G = np.load(Path(d) / a.graph)
        Q = np.load(Path(d) / "Q_test.npy")
        dim = X.shape[1]
        Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
        Qg = torch.from_numpy(Q).cuda()
        pool = load_pool(d, "random4k", 4096)
        arms = Arms(Xg, Qg, pool, d, "random4k", 128, 0.1, 32, 24)
        seeds, _, _ = arms.seeds("distance", 8)
        sd = seeds.unsqueeze(1)
        res = {}
        for B in a.budgets:
            for _ in range(2):
                engine.search(Xg, Gg, Qg, sd, k=10, m=a.m, budget=B, team_size=16,
                              policy=engine.POLICY_ROUND_ROBIN)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(3):
                _, _, st = engine.search(Xg, Gg, Qg, sd, k=10, m=a.m, budget=B,
                                         team_size=16, policy=engine.POLICY_ROUND_ROBIN)
            torch.cuda.synchronize()
            res[B] = ((time.perf_counter() - t) / 3 * 1e3,
                      float(st.dist_evals.sum(1).float().mean()),
                      float(st.expansions.sum(1).float().mean()))
        (m1, d1, e1), (m2, d2, e2) = res[a.budgets[0]], res[a.budgets[1]]
        nq = Q.shape[0]
        per_dist = (m2 - m1) / 1e3 / ((d2 - d1) * nq)
        read = dim * 4 / a.hbm
        print(f"{name}  d={dim} deg={G.shape[1]}")
        print(f"   {(d2-d1)/(e2-e1):5.1f} new distances per expansion, "
              f"{per_dist*1e9:5.2f} ns each")
        print(f"   reading the vector alone would take {read*1e9:5.2f} ns "
              f"({100*read/per_dist:.0f}% of it)")
        print(f"   effective bandwidth {dim*4/per_dist/1e12:5.2f} TB/s "
              f"({100*(dim*4/per_dist)/a.hbm:.0f}% of HBM)\n")
        del Xg, Gg, Qg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
