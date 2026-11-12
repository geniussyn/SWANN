"""Which entries climb to an answer-bearing basin, and what predicts it cheaply?

bench/mass_alloc.py: allocating V over the ANSWER-HOLDING pieces is worth
+0.0196 at V=8 (p=2e-47), +0.0053 at V=32, and it takes the share of entries
landing in an answer-holding piece from 60-65% to 96-99%. Equal and mass-
proportional quotas tie, so the lever is the piece's identity, not its weight.
That arm is an oracle. This asks what can stand in for it.

The pieces are not an offline structure: `climb` is the query's own greedy walk,
so an entry's piece is where ITS descent ends for THIS query. The natural cheap
signal is therefore the walk's own output, and the question is how early it is
readable:

  attractor similarity  <q, x_A> for the node the full climb ends at. The
                        ceiling for any walk-based signal.
  t-step similarity     <q, .> after only t best-neighbour steps, t small. If
                        this already ranks entries the way the attractor does,
                        the method is oversubscribe-then-prune: start 4V entries
                        on a small budget, keep the V whose walks look best,
                        spend the rest there. The engine is resumable
                        (advance(a) then advance(b) == advance(a+b)), so that
                        costs nothing to express.
  entry similarity      <q, x_p>, the entry's own score -- what nearest-V uses.
                        The baseline to beat, and 4.5 already warns it is weak:
                        entry similarity predicts fragment disjointness at AUC
                        0.606.

Reported as AUC for "this entry's piece holds an answer", over the nearest M.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def auc(score, label):
    """Per query, then averaged. Queries with no positive or no negative skipped."""
    out = []
    for s, y in zip(score, label):
        p, n = y.sum(), (~y).sum()
        if p == 0 or n == 0:
            continue
        r = np.argsort(np.argsort(s)) + 1
        out.append((r[y].sum() - p * (p + 1) / 2) / (p * n))
    return float(np.mean(out)), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=256)
    ap.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    nearQ = select_a.select_nearest(Q, Vp, a.M, engine.METRIC_IP)
    ent = pt[nearQ]                                                  # [NQ, M] node ids
    att_en = climb(X, G, ent, Q)
    att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q)

    gtset = [set(r.tolist()) for r in att_gt.cpu()]
    label = np.zeros((NQ, a.M), bool)
    ae = att_en.cpu().numpy()
    for q in range(NQ):
        label[q] = np.isin(ae[q], list(gtset[q]))
    print(f"{a.data.name} {a.split} {NQ} queries, nearest M={a.M} entries, k={a.k}\n")
    print(f"entries whose basin holds an answer: {label.mean():.1%} of the "
          f"nearest {a.M}")
    print(f"  among the nearest 8:  {label[:, :8].mean():.1%}")
    print(f"  among the nearest 32: {label[:, :32].mean():.1%}\n")

    def sim(ids):
        return (X[ids.reshape(-1)].view(NQ, -1, X.shape[1])
                * Q.unsqueeze(1)).sum(2).cpu().numpy()

    print(f"  {'signal':34} {'AUC':>8} {'queries scored':>16}")
    rows = []
    s_ent = sim(ent)
    v, n = auc(s_ent, label)
    print(f"  {'entry similarity <q,x_p> (baseline)':34} {v:8.4f} {n:16d}")
    rows.append(dict(signal="entry", auc=v, n=n))
    for t in a.steps:
        st = climb(X, G, ent, Q, steps=t)
        v, n = auc(sim(st), label)
        print(f"  {f'{t}-step climb similarity':34} {v:8.4f} {n:16d}")
        rows.append(dict(signal=f"climb{t}", auc=v, n=n))
    v, n = auc(sim(att_en), label)
    print(f"  {'attractor similarity (ceiling)':34} {v:8.4f} {n:16d}")
    rows.append(dict(signal="attractor", auc=v, n=n))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, M=a.M,
                                         positive_rate=float(label.mean()),
                                         rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
