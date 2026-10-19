"""IVF + RaBitQ on the GPU: build the index.

RaBitQ (Gao & Long, SIGMOD 2024) quantises a vector's residual from its IVF
centroid to one bit per dimension after a random rotation, and recovers an
unbiased inner product from the sign pattern alone. Per vector it keeps:

    code    D bits, sign(P (x - c)) packed into uint32 words
    rnorm   ||x - c||
    err     <o_bar, o_res>, the cosine between the sign vector and the true
            residual direction, which is what makes the estimator unbiased

and the estimate of <q, x> is

    <q, c> + rnorm * ( <P q_res, o_bar> / err )

with `o_bar` the sign vector scaled by 1/sqrt(D). The query is never quantised.
That asymmetry is the whole reason the scheme reaches useful accuracy at one bit
-- a symmetric version is a strawman -- and it costs nothing at search time
because the query is rotated once per probe list, not once per candidate.

The random rotation is a fixed orthogonal matrix drawn once and stored with the
index: it spreads whatever structure the residuals have across all coordinates,
which is what lets a per-coordinate sign carry information.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def kmeans(Xg, n_lists, iters=25, seed=0, chunk=100_000):
    g = torch.Generator(device="cpu").manual_seed(seed)
    idx = torch.randperm(Xg.shape[0], generator=g)[:n_lists].to(Xg.device)
    C = Xg[idx].clone()
    for _ in range(iters):
        assign = torch.empty(Xg.shape[0], dtype=torch.long, device=Xg.device)
        for s in range(0, Xg.shape[0], chunk):
            e = min(s + chunk, Xg.shape[0])
            assign[s:e] = torch.cdist(Xg[s:e], C).argmin(1)
        newC = torch.zeros_like(C)
        newC.index_add_(0, assign, Xg)
        cnt = torch.bincount(assign, minlength=n_lists).unsqueeze(1).clamp(min=1)
        newC = newC / cnt
        empty = torch.bincount(assign, minlength=n_lists) == 0
        newC[empty] = C[empty]
        C = newC
    return C, assign


@torch.no_grad()
def random_rotation(D, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(D, D, generator=g)
    Qm, _ = torch.linalg.qr(A)
    return Qm.to(device).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--lists", type=int, default=0, help="0 = 2*sqrt(n)")
    ap.add_argument("--iters", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="rabitq")
    a = ap.parse_args()

    X = np.load(a.data / "X.npy")
    n, D = X.shape
    pad = (-D) % 32            # the codes pack 32 dimensions to a word
    if pad:
        X = np.ascontiguousarray(np.pad(X, ((0, 0), (0, pad))))
        D += pad
    lists = a.lists or int(2 * np.sqrt(n))
    Xg = torch.from_numpy(X).cuda()

    t = time.perf_counter()
    C, assign = kmeans(Xg, lists, a.iters, a.seed)
    R = random_rotation(D, Xg.device, a.seed)

    W = D // 32
    codes = torch.zeros((n, W), dtype=torch.int32, device=Xg.device)
    rnorm = torch.zeros(n, device=Xg.device)
    err = torch.zeros(n, device=Xg.device)
    bitpos = (1 << torch.arange(32, device=Xg.device, dtype=torch.int64))
    for s in range(0, n, 50_000):
        e = min(s + 50_000, n)
        res = Xg[s:e] - C[assign[s:e]]
        nr = res.norm(dim=1).clamp(min=1e-12)
        rot = (res / nr.unsqueeze(1)) @ R.T                      # unit residual, rotated
        sign = (rot > 0)
        obar = torch.where(sign, 1.0, -1.0) / np.sqrt(D)
        err[s:e] = (obar * rot).sum(1).clamp(min=1e-6)           # <o_bar, o_res>
        rnorm[s:e] = nr
        packed = (sign.view(e - s, W, 32).long() * bitpos).sum(-1)
        codes[s:e] = packed.to(torch.int32)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t

    order = torch.argsort(assign)                                 # group by list
    starts = torch.searchsorted(assign[order], torch.arange(lists, device=Xg.device))
    counts = torch.bincount(assign, minlength=lists)

    out = a.data / a.out
    out.mkdir(parents=True, exist_ok=True)
    for name, t_ in (("centroids", C), ("rotation", R), ("codes", codes),
                     ("rnorm", rnorm), ("err", err), ("order", order.to(torch.int32)),
                     ("starts", starts.to(torch.int32)), ("counts", counts.to(torch.int32))):
        np.save(out / f"{name}.npy", t_.cpu().numpy())
    (out / "manifest.json").write_text(json.dumps(
        dict(n=n, d=D, words=W, n_lists=lists, iters=a.iters, seed=a.seed,
             build_seconds=round(build_s, 2),
             bytes_per_vector=dict(code=W * 4, scalars=8, fp32=D * 4)), indent=2))
    print(f"IVF-RaBitQ: n={n:,} d={D} lists={lists} -> {build_s:.1f}s")
    print(f"  per vector: {W*4} bytes of code + 8 of scalars, against {D*4} fp32 "
          f"({D*4/(W*4+8):.0f}x smaller)")
    print(f"  list sizes: min/med/max {int(counts.min())}/{int(counts.median())}/{int(counts.max())}")
    print(f"  mean <o_bar,o_res> = {float(err.mean()):.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
