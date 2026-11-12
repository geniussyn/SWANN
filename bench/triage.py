"""Oversubscribe, then prune: start cV entries cheaply, keep the V going somewhere.

bench/mass_alloc.py: allocating V over the ANSWER-HOLDING basins is worth +0.0196
at V=8 (p=2e-47) and takes the share of entries landing in such a basin from
64.9% to 98.6%. bench/basin_predict.py: the entry's own score predicts that event
at AUC 0.5393 -- a coin flip, and this is what 4.7's "proximity is worth almost
nothing" actually was -- while the similarity reached after a few best-neighbour
steps predicts it at 0.735 (2 steps), 0.933 (4) and 0.9995 (8).

So the signal is free: it is the search's own first few expansions. Start cV
branches with a budget of t expansions each, keep the V whose queues look best,
and spend the rest of the budget there.

The engine needs no change. `init_states` then `advance(t)` leaves the state in
caller-owned tensors, `st.dist[:, j, 0]` is branch j's best so far and
`st.ids[:, j, 0]` is where it got to, and a second search runs from those.

Accounting. Expansions are held equal to the baseline: the baseline spends
total_B over V branches, this spends cV*t on the triage and the remainder over V.
Distance evaluations are reported as the SUM of both passes with no cache carried
between them, which overcounts this arm -- the real implementation would keep the
per-query table -- so the number here is a floor on the saving, not a ceiling.
"""
import argparse, json, sys, time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--c", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--t", type=int, nargs="+", default=[2, 4, 8])
    ap.add_argument("--resume-at", default="frontier", choices=["frontier", "entry"],
                    help="seed pass 2 at where the branch got to, or back at its entry")
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--mode", default="inplace", choices=["inplace", "restart"],
                    help="inplace: one state, retire the losers, let the winners "
                         "keep going -- cache, visited and claims stay consistent "
                         "and nothing is double counted. restart: a second search "
                         "from the survivors' seeds, which recomputes pass 1's "
                         "distances (carrying the table over instead is wrong: it "
                         "carries pass 1's CLAIMS, and exclusive mode then locks "
                         "the survivors out of the territory they need -- measured "
                         "recall 0.004).")
    ap.add_argument("--codes", action="store_true",
                    help="run the triage pass on the 1-bit codes (128 B/vector "
                         "against 4096). Its output is an index, not a distance, "
                         "which is the one place this lab allows reduced precision.")
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip")
    metric = engine.METRIC_IP
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    nearQ = select_a.select_nearest(Q, Vp, max(a.V) * max(a.c), metric)
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q)
    gtsets = [set(r.tolist()) for r in att_gt.cpu()]

    codes = engine.load_codes(a.data) if a.codes else None
    print(f"{a.data.name} {a.split} {NQ} queries, pool {a.pool} P={a.P:,}, "
          f"k={a.k}, total_B={a.total_B}, mode {a.mode}, "
          f"triage on {'1-bit codes' if codes else 'fp32'}\n")

    def recall_of(ids):
        got = ids.cpu().numpy()
        return np.array([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                         for q in range(NQ)])

    def in_mass(seed_nodes):
        att = climb(X, G, seed_nodes.to(torch.int64), Q).cpu().numpy()
        return float(np.mean([np.mean([A in gtsets[q] for A in att[q]])
                              for q in range(NQ)]))

    def plain(seed_nodes, V, B, reps=3):
        for r in range(reps + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids, _, st = engine.search(X, G, Q, seed_nodes.to(torch.int32), k=a.k,
                                       L=a.L, B=B, metric=metric, team_size=a.team,
                                       stop_eps=a.stop_eps, shared_cache=True,
                                       shared_bound=True, exclusive=True,
                                       elide_visited=True)
            torch.cuda.synchronize()
            if r == 0:
                continue                                  # warm up
            ms = (time.perf_counter() - t0) * 1e3
            best = ms if r == 1 else min(best, ms)
        return (recall_of(ids), float(st.dist_evals.sum(1).float().mean()), best)

    def triaged_inplace(V, c, t, B2):
        """One state. Retire the losers in place; the winners keep their queues."""
        Vp_ = V * c
        st = engine.init_states(X, Q, pt[nearQ[:, :Vp_]].to(torch.int32),
                                L=a.L, B=t + B2, D=G.shape[1], metric=metric,
                                team_size=a.team, stop_eps=a.stop_eps, kstop=a.k,
                                shared_cache=True, shared_bound=True,
                                exclusive=True, elide_visited=True)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        engine.advance(X, G, Q, st, t, codes=codes)
        d1 = float(st.dist_evals.sum(1).float().mean())
        bq = st.dist[:, :, 0]
        keep = torch.topk(-bq, V, dim=1).indices                  # [NQ, V]
        drop = torch.ones_like(bq, dtype=torch.bool)
        drop.scatter_(1, keep, False)
        # a branch with nothing unexpanded stops: that is the documented exit, so
        # emptying the losers' queues frees them without touching the kernel.
        st.ids[drop] = -1
        st.expanded[drop] = 1
        engine.advance(X, G, Q, st, B2)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
        seed = pt[nearQ[:, :Vp_]].gather(1, keep)
        d = float(st.dist_evals.sum(1).float().mean())
        bi, bd = engine.branch_topk(st, a.k)
        ids, _ = engine.merge_branches(bi, bd, a.k)
        return recall_of(ids), d, d1, seed, ms

    def triaged(V, c, t, B2):
        Vp_ = V * c
        st = engine.init_states(X, Q, pt[nearQ[:, :Vp_]].to(torch.int32),
                                L=a.L, B=t, D=G.shape[1], metric=metric,
                                team_size=a.team, stop_eps=-1.0, kstop=a.k,
                                shared_cache=True, elide_visited=True)
        engine.advance(X, G, Q, st, t, codes=codes)
        d1 = float(st.dist_evals.sum(1).float().mean())
        best = st.dist[:, :, 0]                                   # [NQ, V*c]
        keep = torch.topk(-best, V, dim=1).indices                # smallest distance
        if a.resume_at == "frontier":
            seed = st.ids[:, :, 0].gather(1, keep).to(torch.int64)
        else:
            seed = pt[nearQ[:, :Vp_]].gather(1, keep)
        st2 = engine.init_states(X, Q, seed.to(torch.int32), L=a.L, B=B2,
                                 D=G.shape[1], metric=metric, team_size=a.team,
                                 stop_eps=a.stop_eps, kstop=a.k,
                                 shared_cache=True, shared_bound=True,
                                 exclusive=True, elide_visited=True)
        engine.advance(X, G, Q, st2, B2)
        bi, bd = engine.branch_topk(st2, a.k)
        ids, _ = engine.merge_branches(bi, bd, a.k)
        d2 = float(st2.dist_evals.sum(1).float().mean())
        return recall_of(ids), d1 + d2, d1, seed, float("nan")

    rows = []
    for V in a.V:
        B0 = max(1, a.total_B // V)
        print(f"V={V}  (baseline: {V} branches x {B0} expansions)")
        print(f"  {'arm':30} {'recall':>8} {'in-mass':>8} {'dist/q':>8} "
              f"{'triage':>7} {'ms':>7} {'vs base':>9} {'p':>9}")
        base, deb, msb = plain(pt[nearQ[:, :V]], V, B0)
        imb = in_mass(pt[nearQ[:, :V]])
        print(f"  {'nearest-V':30} {base.mean():8.4f} {imb:8.1%} {deb:8.0f} "
              f"{'':>7} {msb:7.2f}")
        rows.append(dict(V=V, arm="nearest", recall=float(base.mean()),
                         in_mass=imb, dist=deb, ms=msb, delta=0.0))
        for c in a.c:
            for t in a.t:
                spent = V * c * t
                if spent >= a.total_B:
                    continue
                B2 = max(1, (a.total_B - spent) // V)
                fn = triaged_inplace if a.mode == "inplace" else triaged
                fn(V, c, t, B2)                                   # warm up
                best_ms = float("inf")
                for _ in range(3):
                    pq, de, d1, seed, ms = fn(V, c, t, B2)
                    best_ms = min(best_ms, ms)
                im = in_mass(seed)
                d = pq - base
                w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
                kk = min(w, l)
                p = (1.0 if n == 0 else
                     min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))
                print(f"  {f'triage c={c} t={t} (B2={B2})':30} {pq.mean():8.4f} "
                      f"{im:8.1%} {de:8.0f} {d1:7.0f} {best_ms:7.2f} "
                      f"{pq.mean()-base.mean():+9.4f} {p:9.1e}")
                rows.append(dict(V=V, arm=f"triage c={c} t={t}", c=c, t=t, B2=B2,
                                 recall=float(pq.mean()), in_mass=im, dist=de,
                                 triage_dist=d1, ms=best_ms,
                                 delta=float(pq.mean()-base.mean()), p=float(p)))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         mode=a.mode, resume_at=a.resume_at,
                                         codes=bool(a.codes), rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
