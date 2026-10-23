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
from swann import _engine as engine, _select as select_a                                 # noqa: E402
from swann._pools import load_pool                                  # noqa: E402


def per_query_recall(got, gt):
    """Recall@k for each query separately. The own-decile protocol needs the
    vector, not the mean: an arm's hard stratum is defined by where *it* does
    badly, so the mean cannot produce it."""
    k = gt.shape[1]
    return np.array([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                     for g, t in zip(got, gt)])


def recall_at_k(got, gt):
    return float(per_query_recall(got, gt).mean())


def pool_ids(n, P, seed=0):
    """Kept for continuity: identical to the 'random' placement."""
    return np.sort(np.random.default_rng(seed).choice(n, size=P, replace=False)).astype(np.int32)


class Arms:
    """Holds whatever the arms need, so selection can be timed on its own."""

    def __init__(self, Xg, Qg, pool, data, pool_name, R, tau, L, n_near, rng_seed=1234,
                 gt=None, metric=0):
        self.Xg, self.Qg, self.gt, self.metric = Xg, Qg, gt, metric
        self.pool_t = torch.from_numpy(pool).to(Xg.device)
        self.P = len(pool)
        self.rng = np.random.default_rng(rng_seed)
        self.tau, self.L, self.n_near, self.rng_seed = tau, L, n_near, rng_seed
        self.maskd = self.reps_t = None
        # Optional second level: a graph over the pool, so the nearest entries
        # are found by a short search instead of a scan of all P.
        self.poolX = self.poolG = None
        # The catalogue is offline metadata, so gather it once rather than once
        # per selection: 537 MB of copy at P=131072, charged to every batch.
        self.catX = select_a.pool_vectors(Xg, self.pool_t)
        self.cat_precision = "tf32"
        self.cat_slice = 8192
        px = Path(data) / "pools" / f"poolX_{pool_name}.npy"
        pg = Path(data) / "pools" / f"poolG_{pool_name}.npy"
        if px.exists() and pg.exists():
            self.poolX = torch.from_numpy(np.load(px)).to(Xg.device).contiguous()
            self.poolG = torch.from_numpy(np.load(pg)).to(Xg.device).contiguous()
        mpath = Path(data) / "pools" / f"mask_{pool_name}_fwd.npy"
        rpath = Path(data) / "pools" / f"reps_{pool_name}.npy"
        if mpath.exists() and rpath.exists():
            self.maskd = torch.from_numpy(select_a.unpack_mask(np.load(mpath), R)).to(Xg.device)
            self.reps_t = torch.from_numpy(np.load(rpath)).to(Xg.device)

    def supports(self, arm):
        if arm == "sketch":
            return self.maskd is not None
        if arm == "hier":
            return self.poolG is not None
        return True

    mmr_lam = 0.5
    strata_near = 0.5      # fraction of E taken as the plain nearest entries
    strata_depth = 512     # how deep into the query's pool ordering the rest reach
    hier_budget = 24
    hier_entry_seeds = 4
    hier_m = 32

    def hier_seeds(self, E):
        """Where the pool search itself starts: a fixed handful of pool nodes.
        Fixed rather than random so the arm is reproducible, and few because the
        pool index is small and converges fast."""
        g = torch.Generator().manual_seed(7)
        s = torch.randperm(self.P, generator=g)[: self.hier_entry_seeds]
        return s.view(1, 1, -1).expand(self.Qg.shape[0], 1, -1).to(torch.int32).to(self.Qg.device)

    def seeds(self, arm, E):
        """(seeds[Q,E] int32 on cpu, selection ms, extra info)."""
        if arm == "random":
            idx = torch.from_numpy(self.rng.integers(0, self.P, (self.Qg.shape[0], E)).astype(np.int64))
            return self.pool_t.cpu()[idx], 0.0, {}
        t0, t1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); t0.record()
        if arm in ("distance", "single"):
            near = select_a.select_nearest(self.Qg, self.catX, E, metric=self.metric,
                                           precision=self.cat_precision,
                                           slice_P=self.cat_slice)
            seeds = self.pool_t[near]
            info = {}
        elif arm == "strata":
            # The entries that would reach an isolated answer sit well outside
            # the nearest E: for answers we miss and that are alone in their
            # piece, the nearest pool node to the answer has median query-rank
            # 139, against 6 for the answers we find. Distance ordering alone
            # never offers that entry at E=32. So take the nearest half as
            # before and spend the other half on ranks spread evenly down to
            # strata_depth, buying coverage of the far pieces at the same E.
            cat = select_a.catalog_scores(self.Qg, self.Xg, self.pool_t, self.metric,
                                          Vp=self.catX, precision=self.cat_precision)
            nn = max(1, int(round(E * self.strata_near)))
            nn = min(nn, E)
            depth = min(self.strata_depth, self.P)
            order = torch.topk(cat, depth, dim=1, largest=False).indices    # [Q, depth]
            far = E - nn
            if far > 0:
                step = torch.linspace(nn, depth - 1, far, device=order.device)
                pick = torch.cat([torch.arange(nn, device=order.device),
                                  step.round().to(torch.long)])
                pick = torch.unique(pick)
                # unique() can collapse ties when depth is barely above E; top up
                # from the next unused near ranks so every query still gets E.
                if pick.numel() < E:
                    spare = torch.arange(depth, device=order.device)
                    spare = spare[~torch.isin(spare, pick)][: E - pick.numel()]
                    pick = torch.cat([pick, spare])
            else:
                pick = torch.arange(E, device=order.device)
            seeds = self.pool_t[order[:, pick[:E]]]
            info = {"near": nn, "depth": depth}
        elif arm == "oracle":
            # Diagnostic only: start from the query's exact top-E. No mechanism
            # can place entries better than this, so whatever a hierarchy could
            # buy is bounded by what this row shows.
            seeds = torch.from_numpy(self.gt[:, :E].astype(np.int32)).to(self.Qg.device)
            info = {}
        elif arm == "hier":
            # One short search in the pool index gives the E nearest entries, at
            # a few hundred distances instead of P.
            st = engine.init_states(self.poolX, self.Qg, self.hier_seeds(E),
                                    L=max(self.hier_m, E), B=self.hier_budget,
                                    D=self.poolG.shape[1], team_size=16,
                                    metric=self.metric, kstop=E,
                                    policy=engine.POLICY_ROUND_ROBIN)
            engine.advance(self.poolX, self.poolG, self.Qg, st, self.hier_budget)
            idx = st.ids[:, 0, :E].long().clamp(min=0)
            seeds = self.pool_t[idx]
            info = {"hier_dist": float(st.dist_evals.sum(1).float().mean())}
        elif arm == "mmr":
            seeds = select_a.select_mmr(self.Qg, self.Xg, self.pool_t, E, lam=self.mmr_lam,
                                       Vp=self.catX, precision=self.cat_precision)
            info = {"lam": self.mmr_lam}
        elif arm == "spread":
            seeds = select_a.select_spread(self.Qg, self.Xg, self.pool_t, E,
                                          Vp=self.catX, precision=self.cat_precision)
            info = {}
        elif arm == "sketch":
            seeds, fb = select_a.select(self.Qg, self.Xg, self.pool_t, self.reps_t, self.maskd,
                                        E, self.tau, L=self.L, n_near=self.n_near,
                                        rng_seed=self.rng_seed, Vp=self.catX,
                                        precision=self.cat_precision)
            info = {"fallback_rate": round(fb, 4)}
        else:
            raise ValueError(arm)
        t1.record(); torch.cuda.synchronize()
        return seeds.to(torch.int32).cpu(), t0.elapsed_time(t1), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m-100k"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--graph", default="G.npy", help="graph file inside --data")
    ap.add_argument("--L", type=int, default=16, help="queue width per branch")
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--pool", default="random")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--L-cand", dest="L_cand", type=int, default=32,
                    help="candidate list size for the A/MMR selectors")
    ap.add_argument("--n-near", type=int, default=24)
    ap.add_argument("--cat-slice", type=int, default=8192,
                    help="entries scored per slice in the fused nearest-entry "
                         "selection; the result is exact at any value")
    ap.add_argument("--cat-precision", default="tf32", choices=["fp32", "tf32", "fp16"],
                    help="precision of the catalogue matmul only; the search "
                         "itself is unaffected. tf32 picks the identical entries")
    ap.add_argument("--strata-near", type=float, default=0.5,
                    help="strata arm: fraction of V taken as the plain nearest "
                         "entries; the rest are spread down to --strata-depth")
    ap.add_argument("--strata-depth", type=int, default=512,
                    help="strata arm: deepest query-rank in the pool an entry "
                         "may come from")
    ap.add_argument("--total-B", dest="total_B", type=int, default=128,
                    help="expansions per query when B is not given, split across branches")
    ap.add_argument("--B", type=int, default=0, help="expansions per branch; 0 = total // V")
    ap.add_argument("--V", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                    help="entries (branches, or lineages when shared)")
    ap.add_argument("--arms", nargs="+", default=["distance", "random", "sketch"])
    ap.add_argument("--fanout", default="branch", choices=["branch", "shared"],
                    help="branch: E independent states. shared: E seeds in one state, "
                         "sharing its queue, visited table and budget.")
    ap.add_argument("--stop-eps", type=float, default=-1.0,
                    help="retire a lineage once its frontier is beyond (1+eps) x "
                         "the current k-th best distance; <0 runs the queue out")
    ap.add_argument("--policy", default="best", choices=["best", "rr"],
                    help="shared only: always the nearest candidate, or each lineage in turn")
    ap.add_argument("--metric", default=None, choices=["ip", "l2"],
                    help="default: read it from the corpus manifest")
    ap.add_argument("--team", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=0,
                    help="advance in chunks of this many expansions, re-forming "
                         "the grid over the branches still running; 0 = one call")
    ap.add_argument("--elide-visited", action="store_true",
                    help="drop the per-branch visited table and let the shared "
                         "cache do the dedup (exclusive mode, one seed per branch)")
    ap.add_argument("--crowd", type=float, default=0.0,
                    help="retire a lineage after two expansions in which this "
                         "fraction of neighbours was already covered (own visited "
                         "or another branch's claim); 0 disables it")
    ap.add_argument("--stall", type=int, default=0,
                    help="give up on a branch after this many expansions without "
                         "improving its k-th best distance; 0 disables it")
    ap.add_argument("--exclusive", action="store_true",
                    help="a node claimed by one branch is off limits to the others")
    ap.add_argument("--bound", action="store_true",
                    help="share the query's k-th best distance across its CTAs")
    ap.add_argument("--cache", action="store_true",
                    help="share one distance table per query across its branch CTAs")
    ap.add_argument("--half", action="store_true",
                    help="store the corpus in fp16 for the search; GT and the "
                         "catalog stay fp32, so recall is still measured against "
                         "the exact fp32 answer")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--subset", type=Path, default=None,
                    help="npy of query indices (e.g. the hard decile)")
    ap.add_argument("--mmr-lam", type=float, default=0.5)
    ap.add_argument("--hier-budget", type=int, default=24)
    ap.add_argument("--hier-seeds", type=int, default=4)
    ap.add_argument("--hier-m", type=int, default=32)
    ap.add_argument("--hard", type=Path, default=None,
                    help="npy of query indices to report recall on separately, while "
                         "the batch itself stays whole -- otherwise a small subset is "
                         "measured at a batch size that underfills the GPU, and the "
                         "number says more about occupancy than about hardness")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--save-perquery", type=Path, default=None,
                    help="npz of per-query recall for every row, for own-decile work")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.subset is not None:
        sel = np.load(a.subset)
        Q, gt = np.ascontiguousarray(Q[sel]), gt[sel]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()
    hard_idx = np.load(a.hard) if a.hard is not None else None
    metric = a.metric or ("ip" if man["metric"].startswith("ip") else "l2")
    metric_id = engine.METRIC_IP if metric == "ip" else engine.METRIC_L2
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, a.R, a.tau, a.L, a.n_near, gt=gt,
                metric=metric_id)
    arms.hier_budget, arms.hier_entry_seeds, arms.hier_m = a.hier_budget, a.hier_seeds, a.hier_m
    arms.mmr_lam = a.mmr_lam
    arms.cat_precision = a.cat_precision
    arms.cat_slice = a.cat_slice
    arms.strata_near = a.strata_near
    arms.strata_depth = a.strata_depth
    Xs = Xg.half().contiguous() if a.half else Xg       # what the search reads
    for w in a.arms:                                              # warm every arm it will time
        if arms.supports(w) and w != "random":
            arms.seeds(w, max(a.V))

    print(f"{man['corpus']} n={man['n']} graph={a.graph}  {a.split} q={Q.shape[0]}  "
          f"L={a.L} pool={a.pool}(P={len(pool)}) fanout={a.fanout}"
          + (f"/{a.policy}" if a.fanout == "shared" else "")
          + (" X=fp16" if a.half else "") + f" tau={a.tau}\n")
    print(f"{'arm':13s} {'V':>3s} {'B':>4s} {'recall@10':>10s} {'dist/q':>8s} {'sel ms':>7s} "
          f"{'search ms':>10s} {'total ms':>9s} {'QPS':>8s}"
          + (f" {'hard@10':>9s}" if a.hard is not None else ""))
    rows, perq, perexp = [], [], []
    policy = engine.POLICY_ROUND_ROBIN if a.policy == "rr" else engine.POLICY_BEST
    for V in a.V:
        # A shared state spends the whole budget; independent branches split it.
        B = a.B or (a.total_B if a.fanout == "shared"
                    else max(1, a.total_B // V))
        for arm in a.arms:
            if not arms.supports(arm):
                continue
            if V == 1 and arm == "random":
                continue
            seeds, sel_ms, info = arms.seeds(arm, V)
            if a.fanout == "shared":
                seeds = seeds.unsqueeze(1)                        # [Q, 1, E]
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(Xs, Gg, Qg, seeds, k=a.k, L=a.L,
                                              B=B, team_size=a.team, policy=policy,
                                              metric=metric_id,
                                              stop_eps=a.stop_eps, stall_patience=a.stall,
                                              crowd_frac=a.crowd,
                                              elide_visited=a.elide_visited,
                                              chunk=a.chunk,
                                              shared_cache=a.cache,
                                              shared_bound=a.bound, exclusive=a.exclusive)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r:
                    best = ms if best is None else min(best, ms)
            st.check()
            got = ids.cpu().numpy()
            pq = per_query_recall(got, gt)
            rec = float(pq.mean())
            if a.save_perquery:
                perq.append(pq)
                perexp.append(st.expansions.sum(1).cpu().numpy())
            rec_hard = recall_at_k(got[hard_idx], gt[hard_idx]) if hard_idx is not None else None
            de = float(st.dist_evals.sum(1).float().mean())
            tot = best + sel_ms
            name = "single" if (V == 1 and arm == "distance") else arm
            if a.fanout == "shared" and V > 1:
                name = f"{name}/{a.policy}"
            rows.append(dict(arm=name, E=V, B=B, L=a.L, graph=a.graph,
                             fanout=a.fanout, policy=a.policy,
                             J=int(st.ids.shape[1]), V_per_branch=int(st.V),
                             cache=bool(st.cache_id.numel() > 0),
                             bound=bool(st.bound.numel() > 0),
                             exclusive=bool(st.exclusive), stall=a.stall, crowd=a.crowd,
                             H=int(st.visited.shape[2]), chunk=a.chunk,
                             eff_fanout=float(st.counters[..., 8].sum().item() /
                                              max(int(st.counters[..., 9].sum().item()), 1)),
                             crowded_frac=float((st.crowded > 0).float().mean()),
                             stalled_frac=float((st.stalled > 0).float().mean()),
                             stop_eps=a.stop_eps, expansions=float(st.expansions.sum(1).float().mean()),
                             recall=rec, recall_hard=rec_hard, dist_per_q=de,
                             select_ms=sel_ms, search_ms=best, total_ms=tot,
                             qps=Q.shape[0] / (tot / 1e3), **info))
            print(f"{name:13s} {V:3d} {B:4d} {rec:10.4f} {de:8.1f} {sel_ms:7.2f} "
                  f"{best:10.2f} {tot:9.2f} {Q.shape[0]/(tot/1e3):8.0f}"
                  + (f" {rec_hard:9.4f}" if rec_hard is not None else "")
                  + ("  " + " ".join(f"{k} {v}" for k, v in info.items()) if info else ""))

    if a.save_perquery:
        a.save_perquery.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save_perquery, perq=np.array(perq),
                 expansions=np.array(perexp),
                 labels=np.array([f"L{r['L']}_V{r['E']}_B{r['B']}_e{r['stop_eps']}"
                                  for r in rows]))
        print(f"\nwrote {a.save_perquery}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, L=a.L, P=a.P,
                                         pool=a.pool, tau=a.tau, cand_L=a.L_cand, n_near=a.n_near,
                                         total_budget=a.total_B, per_branch=a.B,
                                         nq=int(Q.shape[0]), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
