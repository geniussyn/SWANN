"""When the best possible entry fails to reach an answer, what is in the way?

Two stories fit the 46.4% arrival rate measured by handing each missed answer the
catalogue node nearest to it. Either the walk reaches the answer's neighbourhood
and is pulled out of it, or it never enters: the entry sits in some other basin
and a greedy climb ends at that basin's own local maximum. The allentown query
says the second -- its four Allentown-the-city documents all climb to one of
themselves and stop, 0.6164 against the answer's 0.7009, because their
neighbours are each other at mutual rank 1-3 and the route out is rank 878.

One query is not a mechanism. This runs the same pure L=1 climb from the best
entry for every missed, alone-in-its-piece answer and asks where it stops: on the
answer, on something nearer the query than the answer, or on a local maximum
that is worse than the answer and simply has nowhere better adjacent.
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
    ap.add_argument("--cap", type=int, default=400, help="max greedy steps")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")[: a.nq]

    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)]
    ids, _, _ = engine.search(X, G, Q, seeds.to(torch.int32), k=a.k, L=a.L, B=a.B,
                              metric=metric, team_size=a.team, stop_eps=a.stop_eps,
                              shared_cache=True, shared_bound=True,
                              exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])
    sizes = np.zeros_like(frag)
    for qq in range(len(frag)):
        c = np.bincount(frag[qq]); sizes[qq] = c[frag[qq]]
    sel = (~hit) & (sizes == 1)
    qi, ki = np.nonzero(sel)
    tgt = gt[qi, ki]
    print(f"{a.data.name} {a.graph}  pool {a.pool}(P={a.P})  V={a.V}\n"
          f"{len(qi)} missed, alone-in-piece answers in {a.nq} queries\n")

    # the entry each answer would most like to be given
    with torch.no_grad():
        tv = X[torch.from_numpy(tgt.astype(np.int64)).cuda()]
        gp = (1.0 - tv @ Vp.T) if ip else torch.cdist(tv, Vp)
        ent = pt[gp.argmin(1)]
        del gp, tv

    def climb(start, qv, cap):
        u = int(start)
        for _ in range(cap):
            nb = G[u].long(); nb = nb[nb >= 0]
            if nb.numel() == 0:
                break
            s = nb_score = (X[nb] @ qv) if ip else -torch.cdist(X[nb], qv.view(1, -1)).view(-1)
            j = int(s.argmax())
            cur = float(X[u] @ qv) if ip else -float(torch.dist(X[u], qv))
            if float(s[j]) <= cur:
                break
            u = int(nb[j])
        return u

    landed_on_answer = better_than_answer = stuck_worse = 0
    gaps = []
    for n in range(len(qi)):
        qv = Q[int(qi[n])]
        end = climb(ent[n], qv, a.cap)
        st = float(X[end] @ qv) if ip else -float(torch.dist(X[end], qv))
        at = float(X[int(tgt[n])] @ qv) if ip else -float(torch.dist(X[int(tgt[n])], qv))
        if end == int(tgt[n]):
            landed_on_answer += 1
        elif st >= at:
            better_than_answer += 1
        else:
            stuck_worse += 1
            gaps.append(at - st)
    n = len(qi)
    print("a pure best-neighbour climb from that entry ends:")
    print(f"  on the answer itself                      {landed_on_answer:5} "
          f"{100*landed_on_answer/n:5.1f}%")
    print(f"  somewhere nearer the query than the answer{better_than_answer:5} "
          f"{100*better_than_answer/n:5.1f}%")
    print(f"  on a local maximum worse than the answer  {stuck_worse:5} "
          f"{100*stuck_worse/n:5.1f}%")
    if gaps:
        g = np.array(gaps)
        print(f"\n  for that last group, how far short it stops "
              f"(similarity to the query):\n"
              f"    p25 {np.percentile(g,25):.4f}   p50 {np.percentile(g,50):.4f}   "
              f"p90 {np.percentile(g,90):.4f}")
    print("\nA climb that stops below the answer stopped because nothing adjacent\n"
          "was better: it is a basin that does not contain the answer, not a walk\n"
          "that found the answer and left it.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, pool=a.pool, P=a.P, n=n,
            on_answer=landed_on_answer, nearer_than_answer=better_than_answer,
            stuck_worse=stuck_worse,
            gap_p50=float(np.percentile(gaps, 50)) if gaps else None), indent=2))


if __name__ == "__main__":
    main()
