"""Seeds directed at the query's components, instead of at the query.

The fragmentation a query's answer suffers comes from the embedding: a query
with two components lands between two groups of documents, each holding one of
them, and the nearest catalogue nodes to the mixed query sit in whichever
component the encoder weighted more. Spreading seeds down the catalogue
ranking has already been measured and loses (the `strata` arm). This asks a
different question: if a cheap first pass says where the mass actually is, do
seeds aimed at each lump reach more of the answer than seeds aimed at the
query?

Three arms, all at the same total distance evaluations, which is the only
comparison that means anything here:

  base        the V nearest catalogue nodes to q. What the paper ships.
  prf-cat     pass 1 with V1 seeds, cluster what it returns into c lumps, then
              the V/c catalogue nodes nearest each lump centroid.
  prf-direct  the same lumps, but the seeds are the returned nodes themselves,
              which is local density bought without a denser catalogue.

Work is counted across both passes, so the second pass is charged for the first.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def pad_to(t, n):
    """Repeat a short seed list up to n. A query whose first pass returned few
    distinct lumps gets duplicates rather than a different arm's seeds."""
    if t.numel() == 0:
        raise ValueError("no seeds to pad")
    if t.numel() >= n:
        return t[:n]
    reps = (n + t.numel() - 1) // t.numel()
    return t.repeat(reps)[:n]


def kmeans(Xs, c, iters=8, gen=None):
    """c centroids over [m, d] rows, one query's candidates. Tiny and exact
    enough: m is a few dozen and the output is a direction, not an answer."""
    m = Xs.shape[0]
    if m <= c:
        return Xs.clone(), torch.zeros(m, dtype=torch.long, device=Xs.device)
    idx = torch.randperm(m, generator=gen, device=Xs.device)[:c]
    C = Xs[idx].clone()
    for _ in range(iters):
        d = torch.cdist(Xs, C)
        a = d.argmin(1)
        for j in range(c):
            sel = a == j
            if sel.any():
                C[j] = Xs[sel].mean(0)
    return C, torch.cdist(Xs, C).argmin(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.12, 0.30])
    ap.add_argument("--v1", type=int, default=8, help="seeds in the first pass")
    ap.add_argument("--eps1", type=float, default=0.02, help="first pass, kept cheap")
    ap.add_argument("--m", type=int, default=50, help="candidates the first pass hands on")
    ap.add_argument("--c", type=int, nargs="+", default=[2, 4], help="lumps per query")
    ap.add_argument("--oracle", action="store_true",
                    help="also seed from the answer's own pieces: the ceiling a "
                         "perfect query decomposition could reach, since no "
                         "decomposition can name the components better than the "
                         "answer does")
    ap.add_argument("--h", type=int, default=1, help="hop count of the piece labels")
    ap.add_argument("--tag", default=None, help="phi/frag suffix; default the graph stem")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pool_t = torch.from_numpy(pool).cuda()
    catX = Xg[pool_t.long()].contiguous()
    nq = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]

    def recall(ids):
        g = ids.cpu().numpy()
        return float(np.mean([len(gts[q] & {int(v) for v in g[q] if v >= 0}) / a.k
                              for q in range(nq)]))

    def run(seeds, eps, L=None, B=None, repeats=3):
        """Returns ids, distance evaluations per query, and the best whole-batch
        search time. Seed selection is timed by the caller, because the oracle
        arm does not have a selector to time."""
        best = None
        for r in range(repeats + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            ids, _, st = engine.search(Xg, Gg, Qg, seeds, k=a.k, L=L or a.L,
                                       B=B or a.B, policy=engine.POLICY_BEST,
                                       team_size=a.team, stop_eps=eps,
                                       shared_cache=True, shared_bound=True,
                                       exclusive=True, elide_visited=True,
                                       metric=metric)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        return ids, float(st.dist_evals.sum(1).float().mean()), best

    def nearest_cat(vecs, e):
        """e catalogue nodes nearest each row of vecs [n, d]."""
        near = select_a.select_nearest(vecs, catX, e, metric=metric, precision="tf32")
        return pool_t[near]

    rows = []
    base_seeds = nearest_cat(Qg, a.V)
    for eps in a.eps:
        ids, de, ms = run(base_seeds, eps)
        r = recall(ids)
        rows.append(dict(arm="base", eps=eps, c=0, recall=r, dist_per_q=de,
                         search_ms=ms, qps=nq / (ms / 1e3)))
        print(f"  base        eps={eps:<5} recall {r:.4f}  dist/q {de:8.0f}  "
              f"qps {nq / (ms / 1e3):9,.0f}")

    # one first pass, reused by every second-pass setting
    p1_seeds = nearest_cat(Qg, a.v1)
    ids1, de1, ms1 = run(p1_seeds, a.eps1)
    print(f"\n  first pass  V={a.v1} eps={a.eps1}: recall {recall(ids1):.4f}  "
          f"dist/q {de1:.0f}  (charged to every arm below)")
    cand = ids1[:, : a.m].clone()

    for c in a.c:
        per = max(1, a.V // c)
        cat_seeds = torch.empty(nq, per * c, dtype=torch.int32, device="cuda")
        dir_seeds = torch.empty(nq, per * c, dtype=torch.int32, device="cuda")
        gen = torch.Generator(device="cuda"); gen.manual_seed(0)
        for q in range(nq):
            ids_q = cand[q][cand[q] >= 0].long()
            if ids_q.numel() == 0:
                cat_seeds[q] = base_seeds[q][: per * c].cuda()
                dir_seeds[q] = base_seeds[q][: per * c].cuda()
                continue
            Xs = Xg[ids_q]
            C, lab = kmeans(Xs, c, gen=gen)
            cs = nearest_cat(C, per).reshape(-1)
            cat_seeds[q] = pad_to(cs.to(torch.int32), per * c)
            take = []
            for j in range(C.shape[0]):
                mem = ids_q[lab == j]
                if mem.numel() == 0:
                    mem = ids_q[:1]
                d = torch.cdist(Xg[mem], C[j:j + 1]).squeeze(1)
                take.append(mem[d.argsort()[:per]])
            dir_seeds[q] = pad_to(torch.cat(take).to(torch.int32), per * c)
        for name, seeds in (("prf-cat", cat_seeds), ("prf-direct", dir_seeds)):
            for eps in a.eps:
                ids, de, ms = run(seeds.cpu(), eps)
                r = recall(ids)
                rows.append(dict(arm=name, eps=eps, c=c, recall=r,
                                 dist_per_q=de + de1, dist_pass2=de, dist_pass1=de1,
                                 search_ms=ms + ms1, qps=nq / ((ms + ms1) / 1e3)))
                print(f"  {name:11} c={c} eps={eps:<5} recall {r:.4f}  "
                      f"dist/q {de + de1:8.0f} (pass2 {de:.0f})")
    if a.oracle:
        tag = a.tag or Path(a.graph).stem
        frag = np.load(a.data / f"frag{a.h}_{a.split}_{tag}.npy")
        if a.nq:
            frag = frag[: a.nq]
        orc = torch.empty(nq, a.V, dtype=torch.int32, device="cuda")
        for q in range(nq):
            lab = frag[q]
            cents = []
            for f in range(int(lab.max()) + 1):
                slots = np.flatnonzero(lab == f)
                if slots.size:
                    cents.append(Xg[torch.as_tensor(gt[q, slots].astype(np.int64),
                                                    device="cuda")].mean(0))
            C = torch.stack(cents)
            per = max(1, a.V // C.shape[0])
            orc[q] = pad_to(nearest_cat(C, per).reshape(-1).to(torch.int32), a.V)
        for eps in a.eps:
            ids, de, ms = run(orc.cpu(), eps)
            r = recall(ids)
            rows.append(dict(arm="oracle-pieces", eps=eps, c=-1, recall=r,
                             dist_per_q=de, search_ms=ms, qps=nq / (ms / 1e3)))
            print(f"  oracle      eps={eps:<5} recall {r:.4f}  dist/q {de:8.0f}  "
                  f"qps {nq / (ms / 1e3):9,.0f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, V=a.V, L=a.L, B=a.B,
                                         v1=a.v1, eps1=a.eps1, m=a.m,
                                         nq=int(nq), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
