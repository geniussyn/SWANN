"""Do branches that climb to the same attractor collect the same piece?

The decomposition in the paper says recall inside a reached piece is 1.0000: the
first branch to arrive mines all of it. If an attractor is a piece -- and the
attractor count tracks phi_1 at 0.984 across corpus-graph pairs -- then a second
branch landing under the same attractor should contribute nothing new, and its
budget is free to move. That is an assumption worth checking before building a
policy on it, because branches do not run a pure climb: a queue of width L
carries candidates worse than the current best, so two branches under one
attractor can still end up in different places.

Per branch: where its entry climbs to, and which of the query's true answers its
own best-k holds. Then, within each attractor, what the second and later branches
add over the first.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--no-exclusive", action="store_true",
                    help="turn off the claims, to see whether they are what "
                         "keeps branches under one attractor from duplicating")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)]     # [NQ, V]

    _, _, st = engine.search(X, G, Q, seeds.to(torch.int32), k=a.k, L=a.L, B=a.B,
                             metric=metric, team_size=a.team, stop_eps=a.stop_eps,
                             shared_cache=True, shared_bound=True,
                             exclusive=not a.no_exclusive,
                             elide_visited=not a.no_exclusive)
    per = st.ids[:, :, : a.k].cpu().numpy()                            # [NQ, V, k]
    exp = st.expansions.cpu().numpy()                                  # [NQ, V]

    # where each entry climbs to, batched over the V entries of every query
    cur = seeds.clone()
    for _ in range(a.steps):
        nb = G[cur.reshape(-1)].long()                                 # [NQ*V, D]
        qv = Q.unsqueeze(1).expand(-1, a.V, -1).reshape(-1, Q.shape[1])
        sn = torch.where(nb >= 0,
                         (X[nb.clamp(min=0)] * qv.unsqueeze(1)).sum(2),
                         torch.full((1,), -1e30, device=X.device))
        bs, bi = sn.max(1)
        here = (X[cur.reshape(-1)] * qv).sum(1)
        nxt = torch.where(bs > here,
                          torch.gather(nb, 1, bi.view(-1, 1)).view(-1),
                          cur.reshape(-1))
        if torch.equal(nxt, cur.reshape(-1)):
            break
        cur = nxt.view(NQ, a.V)
    att = cur.cpu().numpy()

    gts = [set(int(t) for t in r) for r in gt]
    first_new, later_new, later_n, first_n = [], [], 0, 0
    exp_first, exp_later = [], []
    groups = []
    for q in range(NQ):
        order = np.argsort(-exp[q])                   # the branch that worked most first
        seen_att, got = {}, set()
        for j in order:
            hits = gts[q] & {int(v) for v in per[q, j] if v >= 0}
            new = len(hits - got)
            got |= hits
            A = int(att[q, j])
            if A not in seen_att:
                seen_att[A] = 1
                first_new.append(new); exp_first.append(exp[q, j]); first_n += 1
            else:
                seen_att[A] += 1
                later_new.append(new); exp_later.append(exp[q, j]); later_n += 1
        groups.append(len(seen_att))

    g = np.array(groups)
    print(f"{a.data.name} {a.graph}  V={a.V} eps={a.stop_eps}  {NQ} queries\n")
    print(f"distinct attractors among the {a.V} entries: "
          f"mean {g.mean():.1f}, p10 {np.percentile(g,10):.0f}, "
          f"p90 {np.percentile(g,90):.0f}")
    print(f"branches sharing an attractor with an earlier one: "
          f"{later_n:,} of {later_n+first_n:,}  "
          f"({100*later_n/(later_n+first_n):.1f}%)\n")
    print(f"{'branch':44} {'n':>8} {'new answers':>12} {'expansions':>11}")
    print(f"{'first under its attractor':44} {first_n:8,} "
          f"{np.mean(first_new):12.4f} {np.mean(exp_first):11.1f}")
    print(f"{'a later one under the same attractor':44} {later_n:8,} "
          f"{np.mean(later_new):12.4f} {np.mean(exp_later):11.1f}")
    share = sum(later_new) / max(sum(first_new) + sum(later_new), 1)
    print(f"\nthe duplicates carry {100*share:.1f}% of the answers and "
          f"{100*sum(exp_later)/max(sum(exp_first)+sum(exp_later),1):.1f}% "
          f"of the expansions")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, V=a.V, nq=NQ, attractors_mean=float(g.mean()),
            later_frac=later_n / (later_n + first_n),
            first_new=float(np.mean(first_new)), later_new=float(np.mean(later_new)),
            first_exp=float(np.mean(exp_first)), later_exp=float(np.mean(exp_later)),
            answer_share_later=float(share)), indent=2))


if __name__ == "__main__":
    main()
