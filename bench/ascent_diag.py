"""Three numbers bench/ascent_index.py got wrong or did not separate.

That run reported "the top-1 slot takes 90.4% of each visited node's steps" and
then a w=1 narrow walk that reaches the full-width attractor 7.9% of the time.
Both cannot be right, and the suspect is the statistic: it averaged the share
PER NODE, so a node the build stepped through once contributes share 1.0 at w=1
by construction. With a heavy-tailed visit distribution that average says almost
nothing about the steps that matter.

  1. concentration, STEP weighted   sum over nodes of (steps through the top w
                                    slots) / all steps, and the same split by how
                                    often the node was visited.
  2. per-step agreement             on held-out queries: at the nodes a full walk
                                    actually passes through, does the index's top
                                    w contain the slot the full walk chose? This
                                    separates a wrong index from a right one whose
                                    errors compound over t steps.
  3. coverage                        what share of the held-out walk's steps happen
                                    at nodes the build ever saw. The index can only
                                    help there; everywhere else it falls back.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.ascent_index import climb_counts                          # noqa: E402


def walk_record(X, G, starts, Qv, steps, chunk=4096):
    """Every (node, chosen slot) the full-width walk passes through."""
    n, m = starts.shape
    cur = starts.reshape(-1).clone()
    qv = Qv.unsqueeze(1).expand(-1, m, -1).reshape(-1, Qv.shape[1])
    nodes, slots = [], []
    for _ in range(steps):
        nxt = torch.empty_like(cur)
        for s in range(0, cur.numel(), chunk):
            e = min(s + chunk, cur.numel())
            nb = G[cur[s:e]].long()
            sc = (X[nb.clamp(min=0)] * qv[s:e].unsqueeze(1)).sum(2)
            here = (X[cur[s:e]] * qv[s:e]).sum(1)
            sn = torch.where(nb >= 0, sc, torch.full((1,), -1e30, device=X.device))
            bs, bi = sn.max(1)
            moved = bs > here
            nodes.append(cur[s:e][moved].clone())
            slots.append(bi[moved].clone())
            nxt[s:e] = torch.where(moved,
                                   torch.gather(nb, 1, bi.view(-1, 1)).view(-1),
                                   cur[s:e])
        if torch.equal(nxt, cur):
            break
        cur = nxt
    return torch.cat(nodes), torch.cat(slots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--w", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--build-steps", type=int, default=16)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Qall = torch.from_numpy(np.load(a.data / "Q_val.npy")[: a.nq]).cuda()
    half = Qall.shape[0] // 2
    Qtr, Qev = Qall[:half], Qall[half:]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    D = G.shape[1]

    counts = torch.zeros(X.shape[0], D, dtype=torch.int32, device=X.device)
    tr_ent = pt[select_a.select_nearest(Qtr, Vp, a.M, engine.METRIC_IP)]
    climb_counts(X, G, tr_ent, Qtr, counts, a.build_steps)
    tot = counts.sum(1)
    visited = tot > 0
    srt = torch.argsort(counts, dim=1, descending=True)
    topk_cnt = torch.gather(counts, 1, srt).cumsum(1)                # [N, D]
    allsteps = int(tot.sum())
    print(f"{a.data.name} graph {a.graph} width {D}, build on {half} queries "
          f"({a.M} entries, {a.build_steps} steps)")
    print(f"  steps taken: {allsteps:,} over {int(visited.sum()):,} distinct nodes "
          f"({visited.float().mean():.3%} of the corpus)")
    vc = tot[visited].float()
    print(f"  visits per visited node: mean {vc.mean():.1f}  median "
          f"{vc.median():.0f}  p99 {torch.quantile(vc, 0.99):.0f}  max {vc.max():.0f}")
    once = (tot == 1).sum()
    print(f"  nodes visited exactly once: {int(once):,} "
          f"({once.float()/visited.sum():.1%} of visited nodes, "
          f"{once.float()/allsteps:.1%} of steps)\n")

    print("concentration of the ascent's choice:")
    print(f"  {'w':>3} {'per-node mean (what the last run reported)':>44} "
          f"{'STEP weighted':>15} {'steps at hubs (>=10 visits)':>28}")
    hub = tot >= 10
    rows = []
    for ww in a.w:
        if ww > D:
            continue
        pernode = (topk_cnt[visited, ww-1].float()
                   / tot[visited].clamp(min=1).float()).mean()
        stepw = topk_cnt[visited, ww-1].sum().float() / allsteps
        hubw = (topk_cnt[hub, ww-1].sum().float() / tot[hub].sum().float()
                if int(hub.sum()) else float('nan'))
        print(f"  {ww:3d} {pernode:44.4f} {stepw:15.4f} {hubw:28.4f}")
        rows.append(dict(w=ww, per_node=float(pernode), step_weighted=float(stepw),
                         hub=float(hubw)))
    print()

    # ---- held out: per-step agreement and coverage --------------------------
    ev_ent = pt[select_a.select_nearest(Qev, Vp, a.M, engine.METRIC_IP)]
    nodes, slots = walk_record(X, G, ev_ent, Qev, a.steps)
    seen = visited[nodes]
    print(f"held-out walk ({Qev.shape[0]} queries x {a.M} entries x <= {a.steps} "
          f"steps): {nodes.numel():,} steps")
    print(f"  steps at a node the build had seen: {seen.float().mean():.1%}")
    print(f"  distinct nodes on the held-out walk: "
          f"{nodes.unique().numel():,}, of which seen: "
          f"{visited[nodes.unique()].float().mean():.1%}\n")
    print("per-step agreement -- does the index's top w hold the slot the full "
          "walk chose?")
    print(f"  {'w':>3} {'all steps':>12} {'steps at seen nodes':>21} "
          f"{'w nearest edges (no log)':>26} {'compounded over 8':>18}")
    for ww in a.w:
        if ww > D:
            continue
        S = srt[:, :ww]
        inw = (S[nodes] == slots.unsqueeze(1)).any(1)
        at_seen = inw[seen].float().mean() if int(seen.sum()) else float('nan')
        base = (slots < ww).float().mean()        # the first w slots, no log at all
        allm = inw.float().mean()
        print(f"  {ww:3d} {allm:12.4f} {at_seen:21.4f} {base:26.4f} "
              f"{allm.item()**a.steps:18.4f}")
        rows.append(dict(w=ww, step_agree=float(allm), step_agree_seen=float(at_seen),
                         nearest_edges_agree=float(base)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, D=D, M=a.M, steps=a.steps,
            build_queries=half, build_steps=allsteps,
            visited=int(visited.sum()), seen_share=float(seen.float().mean()),
            rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
