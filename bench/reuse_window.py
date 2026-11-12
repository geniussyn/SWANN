"""Reuse per round among the evaluations that survive the existing coordination.

The raw request count double-counts what the per-branch visited table and the
shared distance table already remove. This tracks, per query, every node that
query has evaluated, so a round's new work is the nodes it has not seen before --
and the reuse that is left is that work over the distinct nodes it names.
"""
import argparse, json
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a
from gpuann.pools import load_pool

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--eps", type=float, default=0.10)
ap.add_argument("--nq", type=int, default=1000)
ap.add_argument("--pool", default="random4k")
ap.add_argument("--P", type=int, default=4096)
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Qg = torch.from_numpy(np.ascontiguousarray(
    np.load(a.data / "Q_val.npy")[: a.nq])).cuda()
Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
dg = engine.graph_degrees(Gg)
pool = load_pool(a.data, a.pool, a.P); pool_t = torch.from_numpy(pool).cuda()
catX = select_a.pool_vectors(Xg, pool_t)
near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
seeds = pool_t[near].unsqueeze(-1).contiguous()
N, d = Xg.shape

st = engine.init_states(Xg, Qg, seeds, L=a.L, B=a.B, D=Gg.shape[1], metric=metric,
                        policy=engine.POLICY_BEST, stop_eps=a.eps, kstop=10,
                        shared_cache=True, shared_bound=True, exclusive=True)
seen = torch.zeros((a.nq, N), dtype=torch.bool, device=Xg.device)
print(f"{man['corpus']} n={N} d={d}  fanout=branch V={a.V} L={a.L} B={a.B} "
      f"eps={a.eps} Q={a.nq}\n")
print(f"{'round':>5s} {'new (q,node)':>13s} {'distinct node':>14s} {'reuse':>7s}")
prev = st.expanded.clone()
tot_new = tot_dist = 0
for rnd in range(a.B):
    engine.advance(Xg, Gg, Qg, st, 1, deg=dg)
    newly = (st.expanded == 1) & (prev == 0) & (st.ids >= 0)
    prev = st.expanded.clone()
    if int(newly.sum()) == 0:
        break
    qi = newly.nonzero(as_tuple=True)[0]                    # query of each parent
    p = st.ids[newly].long()
    nb = Gg[p].long()                                       # [np, D]
    qq = qi.view(-1, 1).expand_as(nb)
    m = nb >= 0
    nb, qq = nb[m], qq[m]
    fresh = ~seen[qq, nb]
    nb, qq = nb[fresh], qq[fresh]
    # one query may reach the same node twice in a round; count it once
    keyed = qq.to(torch.int64) * N + nb
    keyed = torch.unique(keyed)
    qq, nb = keyed // N, keyed % N
    seen[qq, nb] = True
    new = int(nb.numel()); dist = int(torch.unique(nb).numel())
    tot_new += new; tot_dist += dist
    print(f"{rnd:5d} {new:13,d} {dist:14,d} {new/max(dist,1):6.2f}x")
print(f"\nnew evaluations {tot_new:,}  ({tot_new/a.nq:.0f} per query)")
print(f"distinct-per-round summed {tot_dist:,}   round reuse {tot_new/max(tot_dist,1):.2f}x")
print(f"whole-search distinct {int(seen.any(0).sum()):,}   "
      f"whole-search reuse {tot_new/max(int(seen.any(0).sum()),1):.2f}x")
print(f"vector traffic {tot_new*d*4/2**30:.1f} GiB now -> "
      f"{tot_dist*d*4/2**30:.1f} GiB per-round grouped -> "
      f"{int(seen.any(0).sum())*d*4/2**30:.1f} GiB whole-search grouped")
