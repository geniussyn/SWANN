"""How much of a batch's vector traffic is the same vector read again.

This is the headroom measurement for an inverted expansion -- one that reads each
distinct node once per round and scores it against every query that wants it.
Measured on MS MARCO 1M at V=32, L=128, B=512: at the high-recall point
(eps=0.30) a batch of 2000 queries touches 972,387 of the million nodes and reads
them 7.33 times over, 14.60 MB per query where 1.99 would do. The reuse grows
with the batch (1.81x at Q=250), so it is a batch-regime result and says nothing
about Q=1.

This counts requests and is therefore an *upper bound* on what an inverted
expansion could collect: the per-branch visited table and the shared distance
table already remove the intra-query duplicates. `bench/reuse_window.py` counts
only the evaluations that survive those, and it is the number a design should be
sized against -- 1.62x if the grouping window is one round, 8.56x if it is the
whole search.

Each (query, branch) CTA reads its candidates' vectors itself, so a node wanted
by several queries of one batch is read once per query. This counts the reuse
that an inverted traversal -- read each distinct node once per round, score it
against every query that wants it -- would be able to collect. The evaluated set
of a query is the union of G[p] over the nodes p it expanded, which the state
reports exactly (the expanded flag travels with the queue entry).
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a
from gpuann.pools import load_pool

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--L", type=int, default=128)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=512)
ap.add_argument("--eps", type=float, nargs="+", default=[0.05, 0.30])
ap.add_argument("--nq", type=int, nargs="+", default=[250, 1000, 2000])
ap.add_argument("--pool", default="random4k")
ap.add_argument("--P", type=int, default=4096)
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Qall = np.load(a.data / "Q_val.npy")
G0 = np.load(a.data / a.graph)
Gg = torch.from_numpy(np.ascontiguousarray(G0)).cuda()
deg = engine.graph_degrees(Gg)
pool = load_pool(a.data, a.pool, a.P); pool_t = torch.from_numpy(pool).cuda()
catX = select_a.pool_vectors(Xg, pool_t)
N, d = Xg.shape
print(f"{man['corpus']} n={N} d={d} graph={a.graph}  V={a.V} L={a.L} B={a.B}\n")
print(f"{'eps':>5s} {'Q':>5s} {'eval/q':>8s} {'sum evals':>11s} {'distinct':>9s} "
      f"{'reuse':>7s} {'hot node':>9s} {'MB/q now':>9s} {'MB/q ideal':>11s}")

for eps in a.eps:
    for nq in a.nq:
        Q = np.ascontiguousarray(Qall[:nq])
        Qg = torch.from_numpy(Q).cuda()
        near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
        seeds = pool_t[near].unsqueeze(1).contiguous()
        ids, dist, st = engine.search(Xg, Gg, Qg, seeds, k=10, L=a.L, B=a.B,
                                      metric=metric, policy=engine.POLICY_ROUND_ROBIN,
                                      stop_eps=eps, shared_cache=True,
                                      shared_bound=True, deg=deg)
        # nodes this query expanded, per query (branches share one state here)
        exp_mask = (st.expanded == 1) & (st.ids >= 0)          # [Q, J, L]
        counts = torch.zeros(N, dtype=torch.int32, device=Xg.device)
        total = 0
        for q in range(nq):
            p = st.ids[q][exp_mask[q]].long()
            if p.numel() == 0:
                continue
            nb = Gg[p].reshape(-1).long()
            nb = nb[nb >= 0]
            u = torch.unique(nb)                               # this query reads each once
            total += int(u.numel())
            counts[u] += 1
        distinct = int((counts > 0).sum())
        reuse = total / max(distinct, 1)
        hot = int(counts.max())
        mb_now = total / nq * d * 4 / 1e6
        mb_ideal = distinct / nq * d * 4 / 1e6
        print(f"{eps:5.2f} {nq:5d} {total/nq:8.0f} {total:11,d} {distinct:9,d} "
              f"{reuse:6.2f}x {hot:9d} {mb_now:9.2f} {mb_ideal:11.2f}")
        del st, ids, dist, counts
        torch.cuda.empty_cache()
