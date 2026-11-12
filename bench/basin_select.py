"""Decompose the query over the catalogue, and spend one entry per mode.

The scan already computes the query's similarity to every catalogue entry and
then throws all but the top V away. Those discarded scores are the decomposition
we want: a query like "what is the weather in allentown?" does not have one peak
over the catalogue but several, one per attractor its answers sit under, and the
entries that matter are the best of each peak rather than the best overall.
Distance ordering cannot see peaks -- it fills V from whichever one is highest,
which is how a catalogue holding 120 entries under the Allentown-the-city
attractor contributes none of the 32 selected.

Separating the peaks needs the entries' relation to each other, and the pool
graph already holds it. Climbing inside the pool graph by the same scores groups
the catalogue into pool-level basins for free, and the question measured here is
whether those agree with the corpus basins the real search will fall into: how
many distinct corpus attractors each selector's V entries land under, and whether
that is what recovers a lost answer.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_map import basins                                   # noqa: E402


def pool_basins(score, poolG, steps=64):
    """Attractor of every catalogue entry under a climb inside the pool graph."""
    nb = poolG.long()
    valid = nb >= 0
    sn = torch.where(valid, score[nb.clamp(min=0)],
                     torch.full_like(score[nb.clamp(min=0)], -1e30))
    best = sn.argmax(1)
    bid = torch.gather(nb, 1, best.view(-1, 1)).view(-1)
    bsc = torch.gather(sn, 1, best.view(-1, 1)).view(-1)
    here = torch.arange(score.numel(), device=score.device)
    nxt = torch.where(bsc > score, bid, here)
    att = nxt.clone()
    for _ in range(steps):
        nx = nxt[att]
        if torch.equal(nx, att):
            break
        att = nx
    return att


def pick_per_basin(score, patt, V, M):
    """Best entry of each pool basin, basins ordered by their best score, then
    the next-nearest entries if the top M held fewer than V basins."""
    order = torch.argsort(score, descending=True)
    cand = order[:M]
    seen, out = set(), []
    for c in cand.tolist():
        b = int(patt[c])
        if b not in seen:
            seen.add(b); out.append(c)
            if len(out) == V:
                return torch.tensor(out, device=score.device), len(seen)
    for c in cand.tolist():                       # top up, nearest first
        if c not in out:
            out.append(c)
            if len(out) == V:
                break
    return torch.tensor(out[:V], device=score.device), len(seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--M", type=int, default=512, help="catalogue entries considered")
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--max-cases", type=int, default=120)
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
    poolG = torch.from_numpy(
        np.load(a.data / "pools" / f"poolG_{a.pool}.npy")).cuda()
    near_all = select_a.select_nearest(Q, Vp, a.V, metric=metric)
    ids, _, _ = engine.search(X, G, Q, pt[near_all].to(torch.int32), k=a.k, L=a.L,
                              B=a.B, metric=metric, team_size=a.team,
                              stop_eps=a.stop_eps, shared_cache=True,
                              shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])
    sizes = np.zeros_like(frag)
    for qq in range(len(frag)):
        c = np.bincount(frag[qq]); sizes[qq] = c[frag[qq]]
    qi, ki = np.nonzero((~hit) & (sizes == 1))
    m = min(a.max_cases, len(qi))
    print(f"{a.data.name} {a.graph}  pool {a.pool}(P={a.P}) V={a.V} M={a.M}\n"
          f"baseline recall {hit.mean():.4f}; {len(qi)} missed alone-in-piece, "
          f"working {m}\n")

    cov_n = []; cov_b = []; rec_n = 0; rec_b = 0; in_n = 0; in_b = 0; nbas = []
    for n in range(m):
        qq, t = int(qi[n]), int(gt[qi[n], ki[n]])
        qv = Q[qq].view(1, -1).contiguous()
        att, _ = basins(X, G, qv.view(-1), ip)
        tgt_basin = int(att[t])
        pool_corpus_att = att[pt]                              # per entry
        sc = (Vp @ qv.view(-1)) if ip else -torch.cdist(Vp, qv).view(-1)
        patt = pool_basins(sc, poolG)
        sel_b, nb = pick_per_basin(sc, patt, a.V, a.M)
        nbas.append(nb)
        sel_n = near_all[qq]
        cov_n.append(len(set(pool_corpus_att[sel_n].tolist())))
        cov_b.append(len(set(pool_corpus_att[sel_b].tolist())))
        in_n += int((pool_corpus_att[sel_n] == tgt_basin).any())
        in_b += int((pool_corpus_att[sel_b] == tgt_basin).any())
        out, _, _ = engine.search(X, G, qv, pt[sel_b].view(1, -1, 1).to(torch.int32),
                                  k=a.k, L=a.L, B=a.B, metric=metric,
                                  team_size=a.team, stop_eps=a.stop_eps,
                                  shared_cache=True, shared_bound=True,
                                  exclusive=True, elide_visited=True)
        rec_b += int(t in set(int(v) for v in out[0].cpu().numpy() if v >= 0))
        del att

    print(f"{'':44} {'nearest V':>10} {'per-basin':>10}")
    print(f"{'distinct corpus attractors the V entries cover':44} "
          f"{np.mean(cov_n):10.2f} {np.mean(cov_b):10.2f}")
    print(f"{'the lost answer’s own attractor is covered':44} "
          f"{100*in_n/m:9.1f}% {100*in_b/m:9.1f}%")
    print(f"{'the lost answer is recovered':44} {0:9.1f}% {100*rec_b/m:9.1f}%")
    print(f"\npool-level basins found among the top {a.M} entries: "
          f"mean {np.mean(nbas):.1f}, min {min(nbas)}, max {max(nbas)}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, pool=a.pool, P=a.P, V=a.V, M=a.M, cases=m,
            cover_nearest=float(np.mean(cov_n)), cover_basin=float(np.mean(cov_b)),
            target_covered_nearest=in_n, target_covered_basin=in_b,
            recovered_basin=rec_b, pool_basins_mean=float(np.mean(nbas))), indent=2))


if __name__ == "__main__":
    main()
