"""Pick entries that are near the query AND near EACH OTHER. The mirror of MMR.

The law (docs/SUBSPACE-ENTRIES.md 12): over 14 cells -- five graph builders,
three encoder widths on the same documents, four corpus sizes, four domains --
the share of the nearest V entries landing in a basin that holds an answer is

    in_mass  =  0.9993 - 1.0299 * (att/entry),     r = -0.972, rms 0.033

where att/entry is the number of distinct basins the V entries reach, over V. It
holds across selection strategies too: piece_dedup drives att/entry to 0.982 and
in_mass to 0.101, and the oracle mass allocation reaches in_mass 0.950 not by
escaping the relation but by driving att/entry down to 0.141.

So the lever is **concentration**, and every entry-side intervention this lab has
tried pushed the other way: coverage over a reach sketch, near-but-spread max-min,
MMR with the trade-off swept, a pool graph (4.5, 4.7). All ten are diversity, all
ten raise att/entry, all ten lost.

The untried mirror image needs no basin labels and no climb -- entries that are
mutually close tend to share a basin, so mutual similarity is a free surrogate.
Greedy, with lam >= 0 REWARDING closeness to what is already picked:

    pick argmax_p  <q, x_p> + lam * max_{p' in picked} <x_p, x_p'>

lam = 0 is nearest-V. lam < 0 is MMR, which is here as the control that should
lose, because it is the arm 4.7 already measured losing.
"""
import argparse, json, sys
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def greedy(S, Vp, V, lam, cand):
    """S [NQ,P] query scores; pick V of the top `cand` by score + lam*max-sim."""
    NQ = S.shape[0]
    top = torch.topk(S, cand, dim=1).indices                         # [NQ, cand]
    sq = torch.gather(S, 1, top)                                     # [NQ, cand]
    Vc = Vp[top]                                                     # [NQ, cand, d]
    sim = Vc @ Vc.transpose(1, 2)                                     # [NQ, cand, cand]
    sim.diagonal(dim1=1, dim2=2).fill_(-1e30)
    picked = torch.zeros(NQ, V, dtype=torch.long, device=S.device)
    taken = torch.zeros(NQ, cand, dtype=torch.bool, device=S.device)
    best = torch.full((NQ, cand), -1e30, device=S.device)
    first = sq.argmax(1)
    picked[:, 0] = first
    taken.scatter_(1, first.unsqueeze(1), True)
    ar = torch.arange(NQ, device=S.device)
    for j in range(1, V):
        best = torch.maximum(best, sim[ar, picked[:, j - 1]])
        obj = sq + lam * best
        obj = obj.masked_fill(taken, -1e30)
        nxt = obj.argmax(1)
        picked[:, j] = nxt
        taken.scatter_(1, nxt.unsqueeze(1), True)
    return torch.gather(top, 1, picked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--lam", type=float, nargs="+",
                    default=[-0.5, 0.0, 0.25, 0.5, 1.0, 2.0])
    ap.add_argument("--cand", type=int, default=256)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
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
    pl = torch.from_numpy(load_pool(a.data, a.pool, a.P).astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pl)
    with select_a.reduced_precision(True):
        S = (Q @ Vp.T).float()
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q).cpu().numpy()
    gs = [set(r.tolist()) for r in att_gt]

    print(f"{a.data.name} {a.split} {NQ} queries, pool {a.pool} P={a.P:,}, "
          f"candidates {a.cand}, k={a.k}, total_B={a.total_B}\n")

    def run(sel, V, B):
        ids, _, st = engine.search(X, G, Q, pl[sel].to(torch.int32), k=a.k, L=a.L,
                                   B=B, metric=metric, team_size=a.team,
                                   stop_eps=a.stop_eps, shared_cache=True,
                                   shared_bound=True, exclusive=True,
                                   elide_visited=True)
        got = ids.cpu().numpy()
        pq = np.array([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        att = climb(X, G, pl[sel], Q).cpu().numpy()
        ape = np.mean([len(set(att[q].tolist())) / sel.shape[1] for q in range(NQ)])
        imm = np.mean([np.mean([x in gs[q] for x in att[q]]) for q in range(NQ)])
        return pq, float(ape), float(imm), float(st.dist_evals.sum(1).float().mean())

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        kk = min(w, l)
        return (1.0 if n == 0 else
                min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n))

    rows = []
    for V in a.V:
        B = max(1, a.total_B // V)
        print(f"V={V}, B={B}")
        print(f"  {'lambda':>8} {'what it is':22} {'recall':>8} {'att/entry':>10} "
              f"{'in_mass':>8} {'1-ape':>7} {'dist/q':>8} {'vs near':>9} {'p':>9}")
        base = None
        for lam in sorted(a.lam):
            sel = (torch.topk(S, V, dim=1).indices if lam == 0.0
                   else greedy(S, Vp, V, lam, a.cand))
            pq, ape, imm, de = run(sel, V, B)
            what = ("MMR (spread)" if lam < 0 else
                    "nearest-V" if lam == 0 else "anti-MMR (cluster)")
            if lam == 0.0:
                base = pq
            p = 1.0 if base is None or lam == 0.0 else sg(pq, base)
            dv = 0.0 if base is None else pq.mean() - base.mean()
            print(f"  {lam:8.2f} {what:22} {pq.mean():8.4f} {ape:10.3f} "
                  f"{imm:8.3f} {1-ape:7.3f} {de:8.0f} {dv:+9.4f} {p:9.1e}")
            rows.append(dict(V=V, lam=lam, recall=float(pq.mean()), att_per_entry=ape,
                             in_mass=imm, dist=de, delta=dv, p=p))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, cand=a.cand, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
