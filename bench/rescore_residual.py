"""Can the rescore corpus be the traversal corpus plus a residual?

The narrow arm stores two copies: 4-bit codes to walk with (0.48 GiB at
d = 1024, N = 1e6) and an fp16 copy to rank with (1.91 GiB). The second
dominates the footprint, and it is why a 4-bit index is *larger* than an fp16
one -- 2.62 GiB against 2.15 -- which is the one claim the bit-width figure
cannot make.

The base codes are already resident, so a second 4-bit code of the *residual*
costs 0.48 GiB and not 1.91, and its scale is fitted to the residual's own
spread rather than the vector's. That is the difference from a flat 8-bit
quantiser, which was measured at -0.0061 recall and rejected: the flat one
spends its extra bits at the vector's dynamic range, the residual one at the
error's.

    python bench/rescore_residual.py --data /dev/shm/gpuann/msmarco-1m
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402


def unpack(packed, scale, bits):
    """Codes back to floats, in the rotated basis."""
    per, off = 8 // bits, 2 ** (bits - 1)
    mask = (1 << bits) - 1
    N, nb = packed.shape
    out = torch.empty((N, nb * per), dtype=torch.float32, device=packed.device)
    for i in range(per):
        out[:, i::per] = ((packed >> (bits * i)) & mask).float() - off
    return out * scale[:, None]


def quantise(V, bits, clip):
    """Per-row scalar quantiser at `bits`, clip in units of the row's sigma."""
    lvl = float(2 ** (bits - 1) - 1)
    sc = (clip * V.std(1, keepdim=True).clamp(min=1e-12)) / lvl
    q = torch.round(V / sc).clamp(-lvl - 1, lvl)
    return q * sc, sc


def best_clip(V, bits, grid=(0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)):
    best, bc = float("inf"), grid[0]
    for c in grid:
        e = float(((quantise(V, bits, c)[0] - V) ** 2).mean())
        if e < best:
            best, bc = e, c
    return bc, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_ptf32.npy")
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--depths", type=int, nargs="+", default=[16, 32, 64, 128])
    ap.add_argument("--epss", type=float, nargs="+", default=[0.02, 0.07, 0.16])
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Q = np.ascontiguousarray(np.load(a.data / f"Q_{a.split}.npy")[: a.nq])
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
    dg = engine.graph_degrees(Gg)
    pool = torch.from_numpy(np.load(a.data / "pools" / "pool_auto.npy")
                            .astype(np.int32)).cuda()
    catX = select_a.pool_vectors(Xg, pool)
    N, d = Xg.shape
    nr = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
    seeds = pool[nr].unsqueeze(-1).contiguous()

    P4, s4, rot = engine.pack_narrow(Xg, 4)
    Qr = engine.rotate_queries(Qg, rot)
    Xr = Xg @ rot                                   # the basis everything lives in
    base = unpack(P4, s4, 4)                        # what the 4-bit corpus holds
    resid = Xr - base
    del Xr
    torch.cuda.empty_cache()

    print(f"{man['corpus']} n={N:,} d={d} {a.split} q={len(Q)}  V={a.V}\n")
    print("reconstruction error of each candidate rescore corpus "
          "(mean squared, in the rotated basis)")
    err0 = float((resid ** 2).mean())
    print(f"  4-bit base only                        {err0:.3e}   0.48 GiB")

    cands = {}
    Xh = Xg.half()
    cands["fp16"] = (Xh, N * d * 2 / 2**30)
    # flat 8-bit, the arm that was already rejected
    c8, e8 = best_clip(Xg[:200000] @ rot, 8)
    q8, _ = quantise(Xg @ rot, 8, c8)
    print(f"  flat 8-bit (clip {c8:.2f} sigma)                   "
          f"{float(((q8 - (base + resid)) ** 2).mean()):.3e}   {N*d/2**30:.2f} GiB")
    cands["flat8"] = ((q8 @ rot.T).half(), N * d / 2**30)
    del q8
    torch.cuda.empty_cache()
    # 4-bit base + 4-bit residual: the base is already resident
    for rb in (2, 4):
        cr, _ = best_clip(resid[:200000], rb)
        dq, _ = quantise(resid, rb, cr)
        e = float(((dq - resid) ** 2).mean())
        extra = N * d * rb / 8 / 2**30
        print(f"  4-bit base + {rb}-bit residual (clip {cr:.2f})     "
              f"{e:.3e}   +{extra:.2f} GiB  (total {0.48+extra:.2f})")
        cands[f"resid{rb}"] = (((base + dq) @ rot.T).half(), 0.48 + extra)
        del dq
        torch.cuda.empty_cache()
    del resid, base
    torch.cuda.empty_cache()

    def rc(got):
        return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / 10
                              for g, t in zip(got, gt)]))

    print(f"\n{'eps':>5s} {'depth':>6s} " +
          "  ".join(f"{k:>9s}" for k in cands) + f"   {'no rescore':>11s}")
    rows = []
    for eps in a.epss:
        ids, _, st = engine.search(P4, Gg, Qr, seeds, k=10, L=16, B=16, metric=metric,
                                   policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                                   shared_cache=True, exclusive=True,
                                   shared_bound=True, xscale=s4, xbits=4)
        raw = rc(ids.cpu().numpy())
        for dep in a.depths:
            cells = []
            for k, (Xc, gib) in cands.items():
                fin = engine.rerank(st, Xc, Qg, k=10, depth=dep, metric=metric)
                v = rc(fin.cpu().numpy())
                cells.append(v)
                rows.append(dict(eps=eps, depth=dep, corpus=k, recall=v,
                                 rescore_gib=gib, raw=raw))
            print(f"{eps:5.2f} {dep:6d} " +
                  "  ".join(f"{v:9.4f}" for v in cells) + f"   {raw:11.4f}")
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
