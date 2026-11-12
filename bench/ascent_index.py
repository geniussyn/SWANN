"""Cut the cost of knowing where an entry descends to, by narrowing the graph.

bench/basin_predict.py: whether an entry's basin holds an answer is predicted by
its own similarity to the query at AUC 0.5393 -- a coin flip -- and by the
similarity reached after a few best-neighbour steps at 0.6302 (1 step), 0.7353
(2), 0.9328 (4) and 0.9995 (8). bench/triage.py spends those steps and gets the
recall (+0.0167 at V=8), but 79% of its work is the spending, so against the base
frontier swept over V, budget and eps it is a wash.

Why it is expensive, and where the only cheap route is:

  the h-hop BALL is exponential   deg is 51 here, so the 4-hop ball is ~6.7M
                                  nodes. Storing a node's neighbourhood offline
                                  therefore stops at h=1, which is exactly the
                                  max-pool arm of bench/entry_maxpool.py, and
                                  1-step AUC 0.6302 is exactly how weak that was.
  the greedy PATH is linear       it visits t nodes. So the information at 4-8
                                  hops is reachable, just not by enumeration.

So the cost is not the path's length, it is the price of one step: picking the
best neighbour evaluates all deg of them. cV * t * deg.

This asks whether the graph can be narrowed for that one purpose. Greedy ascent
at node p picks argmax_{u in N(p)} <u, q>. Over all directions every neighbour
can win -- in d=1024 all 51 are extreme points of their own convex hull, so the
geometric bound is vacuous -- but over the directions that actually arrive at p,
the distribution may be concentrated. If the top w slots take most of the
choices, an **ascent index** of w edges per node reproduces the walk at w/deg of
the cost, and at w=4 that is 12.75x off the 79%.

Built from a query log, which is a construction this lab already uses
(ngfix, roargraph). Built on the first half of val, measured on the second, so
the index never sees the queries it is scored on.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def climb_counts(X, G, starts, Qv, counts, steps, chunk=4096):
    """Walk, and tally which neighbour SLOT each visited node sends the walk to."""
    n, m = starts.shape
    cur = starts.reshape(-1).clone()
    qv = Qv.unsqueeze(1).expand(-1, m, -1).reshape(-1, Qv.shape[1])
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
            nxt[s:e] = torch.where(moved, torch.gather(nb, 1, bi.view(-1, 1)).view(-1),
                                   cur[s:e])
            # tally only the steps that were actually taken
            counts.view(-1).index_add_(
                0, (cur[s:e][moved] * G.shape[1] + bi[moved]),
                torch.ones(int(moved.sum()), dtype=counts.dtype, device=counts.device))
        if torch.equal(nxt, cur):
            break
        cur = nxt
    return cur.view(n, m)


def climb_narrow(X, G, S, starts, Qv, steps, chunk=4096):
    """The same walk, but only the w slots the ascent index keeps at each node."""
    n, m = starts.shape
    w = S.shape[1]
    cur = starts.reshape(-1).clone()
    qv = Qv.unsqueeze(1).expand(-1, m, -1).reshape(-1, Qv.shape[1])
    for _ in range(steps):
        nxt = torch.empty_like(cur)
        for s in range(0, cur.numel(), chunk):
            e = min(s + chunk, cur.numel())
            nb = torch.gather(G[cur[s:e]].long(), 1, S[cur[s:e]].long())   # [n, w]
            sc = (X[nb.clamp(min=0)] * qv[s:e].unsqueeze(1)).sum(2)
            here = (X[cur[s:e]] * qv[s:e]).sum(1)
            sn = torch.where(nb >= 0, sc, torch.full((1,), -1e30, device=X.device))
            bs, bi = sn.max(1)
            nxt[s:e] = torch.where(bs > here,
                                   torch.gather(nb, 1, bi.view(-1, 1)).view(-1),
                                   cur[s:e])
        if torch.equal(nxt, cur):
            break
        cur = nxt
    return cur.view(n, m)


def auc(score, label):
    out = []
    for s, y in zip(score, label):
        p, n = y.sum(), (~y).sum()
        if p == 0 or n == 0:
            continue
        r = np.argsort(np.argsort(s)) + 1
        out.append((r[y].sum() - p * (p + 1) / 2) / (p * n))
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=64, help="entries per query, as the triage uses")
    ap.add_argument("--w", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--steps", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--build-steps", type=int, default=16)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Qall = torch.from_numpy(np.load(a.data / "Q_val.npy")[: a.nq]).cuda()
    gtall = np.load(a.data / "gt_val_ids.npy")[: a.nq, : a.k]
    half = Qall.shape[0] // 2
    Qtr, Qev = Qall[:half], Qall[half:]
    gtev = gtall[half:]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    D = G.shape[1]
    print(f"{a.data.name}  graph {a.graph} width {D}  build on {half} val queries, "
          f"measure on {Qev.shape[0]}  M={a.M} entries, k={a.k}\n")

    # ---- build the ascent index on the training half ------------------------
    counts = torch.zeros(X.shape[0], D, dtype=torch.int32, device=X.device)
    tr_ent = pt[select_a.select_nearest(Qtr, Vp, a.M, engine.METRIC_IP)]
    climb_counts(X, G, tr_ent, Qtr, counts, a.build_steps)
    tot = counts.sum(1)
    visited = tot > 0
    srt = torch.argsort(counts, dim=1, descending=True)
    cum = torch.gather(counts, 1, srt).cumsum(1).float()
    share = cum / tot.clamp(min=1).unsqueeze(1).float()
    print(f"nodes the build ever stepped through: {int(visited.sum()):,} "
          f"({visited.float().mean():.3%} of the corpus)")
    print("of each visited node's steps, the share taken by its top-w slots:")
    for ww in a.w:
        if ww <= D:
            print(f"  w={ww:<3d} {share[visited, ww-1].mean():.4f}")
    print()

    # ---- the label: the FULL-width walk is the definition -------------------
    ev_ent = pt[select_a.select_nearest(Qev, Vp, a.M, engine.METRIC_IP)]
    att_full = climb(X, G, ev_ent, Qev)
    att_gt = climb(X, G, torch.from_numpy(gtev.astype(np.int64)).cuda(), Qev)
    gtsets = [set(r.tolist()) for r in att_gt.cpu()]
    ae = att_full.cpu().numpy()
    label = np.zeros(ae.shape, bool)
    for q in range(ae.shape[0]):
        label[q] = np.isin(ae[q], list(gtsets[q]))
    print(f"entries whose basin holds an answer: {label.mean():.1%}\n")

    def sim(ids):
        return (X[ids.reshape(-1)].view(ids.shape[0], -1, X.shape[1])
                * Qev.unsqueeze(1)).sum(2).cpu().numpy()

    print(f"  {'ascent width':16} {'steps':>6} {'AUC':>8} {'dist/step/entry':>16} "
          f"{'reaches full attractor':>24}")
    rows = []
    v = auc(sim(ev_ent), label)
    print(f"  {'entry itself':16} {0:6d} {v:8.4f} {0:16d} {'':>24}")
    rows.append(dict(w=0, steps=0, auc=v))
    for t in a.steps:
        full = climb(X, G, ev_ent, Qev, steps=t)
        v = auc(sim(full), label)
        agree = float((full == att_full).float().mean())
        print(f"  {'full (all slots)':16} {t:6d} {v:8.4f} {D:16d} {agree:23.1%}")
        rows.append(dict(w=D, steps=t, auc=v, agree=agree))
    # the fallback for a node the build never saw: its first w slots, which the
    # builder wrote nearest-first, so this is "keep the w nearest edges".
    for ww in a.w:
        S = srt[:, :ww].to(torch.int16).clone()
        fb = torch.arange(ww, device=X.device, dtype=torch.int16)
        S[~visited] = fb
        for t in a.steps:
            nar = climb_narrow(X, G, S, ev_ent, Qev, steps=t)
            v = auc(sim(nar), label)
            agree = float((nar == att_full).float().mean())
            print(f"  {f'index w={ww}':16} {t:6d} {v:8.4f} {ww:16d} {agree:23.1%}")
            rows.append(dict(w=ww, steps=t, auc=v, agree=agree))
        # and the control: w nearest edges, no log, no index at all
        S0 = torch.arange(ww, device=X.device, dtype=torch.int16).expand(
            X.shape[0], ww).contiguous()
        for t in a.steps:
            nar = climb_narrow(X, G, S0, ev_ent, Qev, steps=t)
            v = auc(sim(nar), label)
            agree = float((nar == att_full).float().mean())
            print(f"  {f'w={ww} nearest only':16} {t:6d} {v:8.4f} {ww:16d} "
                  f"{agree:23.1%}")
            rows.append(dict(w=ww, steps=t, auc=v, agree=agree, arm="nearest-edges"))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, D=D, M=a.M, nq=a.nq,
            build_queries=half, visited=int(visited.sum()),
            share={str(ww): float(share[visited, ww-1].mean()) for ww in a.w if ww <= D},
            positive_rate=float(label.mean()), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
