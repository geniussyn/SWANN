"""Stage 1 integration run: the engine on real vectors, under the baseline arms.

Three entry policies that need no new research -- one seed, E random seeds, the
E pool seeds nearest the query -- measured against the same graph, the same
vectors, the same exact distance and the same *total* expansion budget. These
are the arms A and B have to beat, and running them now is what turns the engine
from "it compiles" into "it searches".

What is deliberately not here: any recall claim about CAGRA (the graph is an
exact kNN stand-in), and any tuned constant (nothing is fitted, and the default
query split is the validation half).
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                            # noqa: E402


def recall_at_k(got, gt):
    """Fraction of each query's exact top-k that the search returned."""
    k = gt.shape[1]
    return np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                    for g, t in zip(got, gt)])


def pool_ids(n, P, seed=0):
    """Offline candidate pool. Fixed random nodes: no GT, no query ever sees it."""
    return np.sort(np.random.default_rng(seed).choice(n, size=P, replace=False)).astype(np.int32)


def seeds_for(arm, Qg, Xg, pool, E, rng, k_near=None):
    """(seeds[Q,E], pool-scoring time in ms). Scoring the pool is a real cost and
    is charged to the arms that do it."""
    Qn = Qg.shape[0]
    if arm == "random":
        idx = torch.from_numpy(rng.integers(0, len(pool), (Qn, E)).astype(np.int64))
        return torch.from_numpy(pool)[idx], 0.0
    t0 = torch.cuda.Event(True); t1 = torch.cuda.Event(True)
    t0.record()
    pv = Xg[torch.from_numpy(pool).long().to(Xg.device)]
    sim = Qg @ pv.T                                   # [Q, P]
    near = torch.topk(sim, max(E, k_near or E), dim=1).indices[:, :E]
    t1.record(); torch.cuda.synchronize()
    return torch.from_numpy(pool).to(Xg.device)[near].cpu(), t0.elapsed_time(t1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--total-budget", type=int, default=128,
                    help="node expansions per query, split evenly across branches")
    ap.add_argument("--E", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--team", type=int, default=32)
    ap.add_argument("--nq", type=int, default=0, help="0 = all")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy"); G = np.load(a.data / "G.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()
    pool = pool_ids(X.shape[0], a.P)
    # Warm the context up before anything is timed: the first matmul on a fresh
    # context pays for module load, and that cost is not pool scoring.
    seeds_for("distance", Qg[:8], Xg, pool, 2, np.random.default_rng(0))
    print(f"{man['corpus']} n={man['n']} d={man['d']} deg={man['deg']} graph={man['graph']}  "
          f"{a.split} queries={Q.shape[0]}  budget={a.total_budget} expansions/query\n")

    rows = []
    print(f"{'arm':10s} {'E':>3s} {'B/branch':>8s} {'recall@10':>10s} {'dist/q':>9s} "
          f"{'exp/q':>6s} {'ms(batch)':>10s} {'QPS':>9s} {'sel ms':>7s}")
    for E in a.E:
        B = max(1, a.total_budget // E)
        for arm in ("random", "distance"):
            if E == 1 and arm == "random":
                continue                                   # single-seed baseline is "distance" at E=1
            rng = np.random.default_rng(1234)
            seeds, sel_ms = seeds_for(arm, Qg, Xg, pool, E, rng)
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(Xg, Gg, Qg, seeds, k=a.k, m=a.m,
                                              budget=B, team_size=a.team)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r:                                      # first pass is warm-up
                    best = ms if best is None else min(best, ms)
            st.check()
            rec = recall_at_k(ids.cpu().numpy(), gt)
            de = float(st.dist_evals.sum(1).float().mean())
            ex = float(st.expansions.sum(1).float().mean())
            name = "single" if E == 1 else arm
            rows.append(dict(arm=name, E=E, B=B, recall=rec, dist_per_q=de, exp_per_q=ex,
                             batch_ms=best, qps=Q.shape[0] / (best / 1e3), select_ms=sel_ms))
            print(f"{name:10s} {E:3d} {B:8d} {rec:10.4f} {de:9.1f} {ex:6.1f} "
                  f"{best:10.2f} {Q.shape[0]/(best/1e3):9.0f} {sel_ms:7.2f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, m=a.m,
                                         P=a.P, total_budget=a.total_budget,
                                         nq=int(Q.shape[0]), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
