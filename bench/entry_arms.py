"""Entry-selection arms on one graph, one corpus, one budget.

Arms differ in which entries a query starts from and in nothing else: same
vectors, same graph, same exact distance, same per-branch budget, same candidate
pool, and -- for every arm that uses one -- the same candidate generator. The
cost of choosing is charged to the arm that pays it: catalog scoring and the
greedy are timed and reported next to the search, because an entry policy that
is free only in the search kernel is not free.

  single/distance  the E pool entries nearest the query (E=1 is the single-seed
                   baseline). Cheap, and the arm A and B have to beat.
  random           E uniform pool entries. Isolates "more starting points" from
                   "better starting points".
  sketch           A: weighted region coverage over static reach masks.

The row that actually matters is none of these: it is E=1 with the queue widened
to spend the same budget. --E 1 --m 64 is that row.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def pool_ids(n, P, seed=0):
    """Kept for continuity: identical to the 'random' placement."""
    return np.sort(np.random.default_rng(seed).choice(n, size=P, replace=False)).astype(np.int32)


class Arms:
    """Holds whatever the arms need, so selection can be timed on its own."""

    def __init__(self, Xg, Qg, pool, data, pool_name, R, tau, L, n_near, rng_seed=1234):
        self.Xg, self.Qg = Xg, Qg
        self.pool_t = torch.from_numpy(pool).to(Xg.device)
        self.P = len(pool)
        self.rng = np.random.default_rng(rng_seed)
        self.tau, self.L, self.n_near, self.rng_seed = tau, L, n_near, rng_seed
        self.maskd = self.reps_t = None
        mpath = Path(data) / "pools" / f"mask_{pool_name}_fwd.npy"
        rpath = Path(data) / "pools" / f"reps_{pool_name}.npy"
        if mpath.exists() and rpath.exists():
            self.maskd = torch.from_numpy(select_a.unpack_mask(np.load(mpath), R)).to(Xg.device)
            self.reps_t = torch.from_numpy(np.load(rpath)).to(Xg.device)

    def supports(self, arm):
        return arm != "sketch" or self.maskd is not None

    def seeds(self, arm, E):
        """(seeds[Q,E] int32 on cpu, selection ms, extra info)."""
        if arm == "random":
            idx = torch.from_numpy(self.rng.integers(0, self.P, (self.Qg.shape[0], E)).astype(np.int64))
            return self.pool_t.cpu()[idx], 0.0, {}
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); t0.record()
        if arm in ("distance", "single"):
            cat = select_a.catalog_scores(self.Qg, self.Xg, self.pool_t)
            near = torch.topk(cat, E, dim=1, largest=False).indices
            seeds = self.pool_t[near]
            info = {}
        elif arm == "spread":
            seeds = select_a.select_spread(self.Qg, self.Xg, self.pool_t, E)
            info = {}
        elif arm == "sketch":
            seeds, fb = select_a.select(self.Qg, self.Xg, self.pool_t, self.reps_t, self.maskd,
                                        E, self.tau, L=self.L, n_near=self.n_near,
                                        rng_seed=self.rng_seed)
            info = {"fallback_rate": round(fb, 4)}
        else:
            raise ValueError(arm)
        t1.record(); torch.cuda.synchronize()
        return seeds.to(torch.int32).cpu(), t0.elapsed_time(t1), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--graph", default="G.npy", help="graph file inside --data")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--pool", default="random")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--n-near", type=int, default=24)
    ap.add_argument("--total-budget", type=int, default=128)
    ap.add_argument("--per-branch", type=int, default=0, help="0 = total-budget // E")
    ap.add_argument("--E", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--arms", nargs="+", default=["distance", "random", "sketch"])
    ap.add_argument("--fanout", default="branch", choices=["branch", "shared"],
                    help="branch: E independent states. shared: E seeds in one state, "
                         "sharing its queue, visited table and budget.")
    ap.add_argument("--policy", default="best", choices=["best", "rr"],
                    help="shared only: always the nearest candidate, or each lineage in turn")
    ap.add_argument("--team", type=int, default=32)
    ap.add_argument("--half", action="store_true",
                    help="store the corpus in fp16 for the search; GT and the "
                         "catalog stay fp32, so recall is still measured against "
                         "the exact fp32 answer")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, a.R, a.tau, a.L, a.n_near)
    Xs = Xg.half().contiguous() if a.half else Xg       # what the search reads
    for w in a.arms:                                              # warm every arm it will time
        if arms.supports(w) and w != "random":
            arms.seeds(w, max(a.E))

    print(f"{man['corpus']} n={man['n']} graph={a.graph}  {a.split} q={Q.shape[0]}  "
          f"m={a.m} pool={a.pool}(P={len(pool)}) fanout={a.fanout}"
          + (f"/{a.policy}" if a.fanout == "shared" else "")
          + (" X=fp16" if a.half else "") + f" tau={a.tau}\n")
    print(f"{'arm':13s} {'E':>3s} {'B':>4s} {'recall@10':>10s} {'dist/q':>8s} {'sel ms':>7s} "
          f"{'search ms':>10s} {'total ms':>9s} {'QPS':>8s}")
    rows = []
    policy = engine.POLICY_ROUND_ROBIN if a.policy == "rr" else engine.POLICY_BEST
    for E in a.E:
        # A shared state spends the whole budget; independent branches split it.
        B = a.per_branch or (a.total_budget if a.fanout == "shared"
                             else max(1, a.total_budget // E))
        for arm in a.arms:
            if not arms.supports(arm):
                continue
            if E == 1 and arm == "random":
                continue
            seeds, sel_ms, info = arms.seeds(arm, E)
            if a.fanout == "shared":
                seeds = seeds.unsqueeze(1)                        # [Q, 1, E]
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(Xs, Gg, Qg, seeds, k=a.k, m=a.m,
                                              budget=B, team_size=a.team, policy=policy)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r:
                    best = ms if best is None else min(best, ms)
            st.check()
            rec = recall_at_k(ids.cpu().numpy(), gt)
            de = float(st.dist_evals.sum(1).float().mean())
            tot = best + sel_ms
            name = "single" if (E == 1 and arm == "distance") else arm
            if a.fanout == "shared" and E > 1:
                name = f"{name}/{a.policy}"
            rows.append(dict(arm=name, E=E, B=B, m=a.m, fanout=a.fanout, policy=a.policy,
                             recall=rec, dist_per_q=de,
                             select_ms=sel_ms, search_ms=best, total_ms=tot,
                             qps=Q.shape[0] / (tot / 1e3), **info))
            print(f"{name:13s} {E:3d} {B:4d} {rec:10.4f} {de:8.1f} {sel_ms:7.2f} "
                  f"{best:10.2f} {tot:9.2f} {Q.shape[0]/(tot/1e3):8.0f}"
                  + (f"  fallback {info['fallback_rate']:.2f}" if info else ""))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, m=a.m, P=a.P,
                                         pool=a.pool, tau=a.tau, L=a.L, n_near=a.n_near,
                                         total_budget=a.total_budget, per_branch=a.per_branch,
                                         nq=int(Q.shape[0]), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
