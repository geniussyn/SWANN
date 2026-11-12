"""How wide does the *rescore* corpus have to be?

The narrow arm is two corpora, not one: a packed one the walk reads, and a
full-precision one the final queue is rescored against. The second is what makes
the first usable -- 4-bit alone returns 0.92 where the rescore takes it to 0.995
-- and it is also where the footprint goes, because the packed copy is an
addition to it and not a replacement.

So the question for memory is how much precision the *ranking* step needs, which
is a different question from how much the *walk* needs. The walk has to order
thousands of candidates well enough to keep going in the right direction; the
rescore has to order sixty-four well enough to pick ten.

Search is always at 4 bits here. Only the rescore corpus changes.

    python bench/rerank_precision.py
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


@torch.no_grad()
def dequantised(X, bits, rot, chunk=200_000):
    """X at `bits` per dimension, returned in fp32 so the rescore can read it.

    A stand-in for storing it at that width: the ranking error is the same, the
    bytes are not, and the bytes are reported from the width rather than from
    this tensor.
    """
    out = torch.empty_like(X)
    lvl = float(2 ** (bits - 1) - 1)
    for s in range(0, X.shape[0], chunk):
        e = min(s + chunk, X.shape[0])
        v = X[s:e] @ rot
        if bits == 1:
            q = torch.sign(v) * v.abs().mean(1, keepdim=True)
        else:
            sc = (engine._best_clip(X, rot, lvl=lvl) *
                  v.std(1, keepdim=True).clamp(min=1e-12)) / lvl
            q = torch.round(v / sc).clamp(-lvl - 1, lvl) * sc
        out[s:e] = q @ rot.T
        del v, q
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_ptf32.npy")
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--epss", type=float, nargs="+", default=[0.04, 0.12, 0.30])
    ap.add_argument("--depth", type=int, default=64)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Q = np.ascontiguousarray(np.load(a.data / "Q_val.npy")[: a.nq])
    gt = np.load(a.data / "gt_val_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
    dg = engine.graph_degrees(Gg)
    pool = np.load(a.data / "pools" / "pool_auto.npy").astype(np.int32)
    pool_t = torch.from_numpy(pool).cuda()
    catX = select_a.pool_vectors(Xg, pool_t)
    near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
    seeds = pool_t[near].unsqueeze(-1).contiguous()
    N, d = Xg.shape
    P4, sc4, rot = engine.pack_narrow(Xg, 4)
    Qr = engine.rotate_queries(Qg, rot)
    deg = float(dg[0].float().mean()) if dg else Gg.shape[1]

    ARMS = [("fp32", 4, Xg), ("fp16", 2, Xg.half().contiguous())]
    for b in (8, 6, 4):
        ARMS.append((f"{b}-bit", b / 8, dequantised(Xg, b, rot)))
    ARMS.append(("none (4-bit raw)", 0, None))

    print(f"{man['corpus']} n={N:,} d={d}  search always 4-bit, V={a.V} L={a.L} "
          f"B={a.B}, rescore depth {a.depth}, {len(Q)} val queries")
    print(f"packed corpus {N*d/2/2**30:.2f} GiB, graph {N*deg*4/2**30:.2f} GiB\n")
    print(f"{'rescore corpus':>18s} {'B/vec':>6s} {'footprint':>10s}  " +
          "  ".join(f"{'r@'+str(e):>8s}" for e in a.epss) + f"  {'ms@'+str(a.epss[1]):>9s}")
    base = {}
    for tag, bpe, Xf in ARMS:
        rcs, ms_at = [], None
        for eps in a.epss:
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, _, st = engine.search(
                    P4, Gg, Qr, seeds, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                    shared_cache=True, exclusive=True, shared_bound=True,
                    xscale=sc4, xbits=4)
                fin = (engine.rerank(st, Xf, Qg, k=10, depth=a.depth, metric=metric)
                       if Xf is not None else ids)
                torch.cuda.synchronize(); el = (time.perf_counter() - t) * 1e3
                if r and (best is None or el < best[0]):
                    best = (el, fin)
            rcs.append(recall_at_k(best[1].cpu().numpy(), gt))
            if eps == a.epss[1]:
                ms_at = best[0]
        foot = (N * d / 2 + N * deg * 4 + (N * d * bpe if bpe else 0)) / 2**30
        if tag == "fp32":
            base = dict(zip(a.epss, rcs))
        delta = "" if tag == "fp32" else "   d " + " ".join(
            f"{r - base[e]:+.4f}" for e, r in zip(a.epss, rcs))
        print(f"{tag:>18s} {int(d*bpe) if bpe else 0:6d} {foot:8.2f} GiB  " +
              "  ".join(f"{r:8.4f}" for r in rcs) + f"  {ms_at:9.2f}" + delta)
        if Xf is not None and tag not in ("fp32", "fp16"):
            del Xf
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
