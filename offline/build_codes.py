"""A small code per node, so an expansion can decide without reading.

Expansion is bandwidth bound -- 1511 GB/s measured, and time tracks bytes moved
almost exactly -- and 87% of the bytes go to neighbours that fail the queue's
test: 96 vectors read, 12.5 kept. A sign code answers the only question the
kernel asks of those bytes. Measured on MS MARCO at the shipped operating point,
scoring the 96 neighbours from their codes and passing the best 16 keeps 100% of
the candidates the exact test would have admitted, at 0.20x the traffic.

The code stands for x - mu and is scored asymmetrically: the query stays fp32
and only the stored side is quantised, so the estimate is <q, code> and the
<q, mu> term it omits is the same for every candidate of a query and cannot
change the order. The query must *not* be centred as well -- that would add
-<mu, x - mu>, which does vary by candidate.

Which code to use depends on the metric, and the difference is large:

  normalised inner product   one bit per dimension is enough, because every
                             vector has the same length and only its direction
                             is in question. Measured on MS MARCO: passing the
                             best 16 of 96 neighbours keeps 100% of the
                             candidates the exact test admits, at 0.20x the bytes.
  unnormalised L2            one bit is not enough -- it throws away the length,
                             and on GIST it keeps only 57% at the same m. int8
                             keeps 100% at m = 8, for 0.33x the bytes.

A sign code also needs one scale per node. The bits give the direction of
x - mu but not its length, so <q, sign(c)> is not in the units of <q, c> and
cannot be compared against a distance. Projecting b onto c fixes it:
<q, c> ~ <q, b> * <c, b> / <b, b> = <q, b> * ||c||_1 / d, so one float per node
puts the estimate back in the corpus's own units -- which is what lets the
kernel compare it against the queue's worst entry instead of having to rank all
D candidates against each other.

An L2 corpus also needs |x|^2, which is one exact scalar per node, so the
distance is reconstructed as |x|^2 - 2<q, code> with only the inner product
approximated.

Writes codes_<name>.npy -- uint32 [n, ceil(d/32)] for 1 bit, int8 [n, d] for
int8 -- the mean as mu_<name>.npy, and sqnorm_<name>.npy when the corpus is not
an inner-product one. At n = 1e6, d = 1024: 128 MB for 1 bit, 1.0 GB for int8,
against 4.1 GB of fp32 vectors.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def quantise_int8(Xg, mu, chunk=16384):
    """x - mu to int8 on one shared scale, so the kernel needs no per-node scale.

    A global scale rather than a per-node one because the kernel reads the code
    of every neighbour and a per-node multiply would cost it another load; the
    accuracy that buys is not needed, since 100% of the admitted candidates
    survive at m = 8 already.
    """
    n, d = Xg.shape
    lo = hi = None
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = Xg[s:e] - mu
        bl, bh = b.aminmax()
        lo = bl if lo is None else torch.minimum(lo, bl)
        hi = bh if hi is None else torch.maximum(hi, bh)
    scale = torch.maximum(hi.abs(), lo.abs()) / 127.0
    out = torch.empty((n, d), dtype=torch.int8, device=Xg.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = torch.clamp(torch.round((Xg[s:e] - mu) / scale),
                               -127, 127).to(torch.int8)
    return out, float(scale)


@torch.no_grad()
def pack_signs(Xg, mu, chunk=16384):
    """sign(x - mu) > 0 as packed bits, 32 dimensions to a word."""
    n, d = Xg.shape
    words = (d + 31) // 32
    out = torch.empty((n, words), dtype=torch.int32, device=Xg.device)
    bit = (torch.arange(32, device=Xg.device, dtype=torch.int32))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        b = (Xg[s:e] - mu) > 0                                   # [m, d]
        if d % 32:
            b = torch.cat([b, torch.zeros((e - s, words * 32 - d),
                                          dtype=torch.bool, device=Xg.device)], 1)
        b = b.view(e - s, words, 32).to(torch.int32)
        out[s:e] = (b << bit).sum(2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--kind", default=None, choices=["1bit", "int8"],
                    help="default: 1bit for a normalised inner-product corpus, "
                         "int8 otherwise, which is what the measurements support")
    ap.add_argument("--name", default=None, help="suffix; defaults to the kind")
    ap.add_argument("--check", type=int, default=4096,
                    help="nodes to verify the packing round-trips on")
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    kind = a.kind or ("1bit" if ip else "int8")
    name = a.name or kind
    X = np.load(a.data / "X.npy", mmap_mode="r")
    n, d = X.shape
    Xg = torch.from_numpy(np.ascontiguousarray(X)).cuda()
    t = time.perf_counter()
    mu = Xg.mean(0)
    scale = None
    alpha = None
    if kind == "1bit":
        codes = pack_signs(Xg, mu)
        alpha = torch.empty(n, device=Xg.device)
        for s in range(0, n, 16384):
            e = min(s + 16384, n)
            alpha[s:e] = (Xg[s:e] - mu).abs().sum(1) / d
    else:
        codes, scale = quantise_int8(Xg, mu)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t

    # the packing is the only place a silent bug would not show up in recall,
    # so it is checked against the unquantised values directly
    g = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(0, n, (min(a.check, n),), generator=g, device="cuda")
    if kind == "1bit":
        want = (Xg[idx] - mu) > 0
        bits = ((codes[idx].unsqueeze(2) >> torch.arange(32, device=Xg.device,
                                                         dtype=torch.int32)) & 1)
        got = bits.view(len(idx), -1)[:, :d].bool()
        assert torch.equal(want, got), "packed bits do not match the signs"
        checked = "bit for bit"
    else:
        err = ((codes[idx].float() * scale) - (Xg[idx] - mu)).abs().max()
        assert float(err) <= scale, f"int8 error {float(err):.3e} over one step"
        checked = f"within one quantisation step ({float(err):.3e} <= {scale:.3e})"

    np.save(a.data / f"codes_{name}.npy", codes.cpu().numpy())
    np.save(a.data / f"mu_{name}.npy", mu.cpu().numpy())
    if alpha is not None:
        np.save(a.data / f"alpha_{name}.npy", alpha.cpu().numpy())
    if not ip:
        sq = torch.empty(n, device=Xg.device)
        for s in range(0, n, 16384):
            e = min(s + 16384, n)
            sq[s:e] = (Xg[s:e] * Xg[s:e]).sum(1)
        np.save(a.data / f"sqnorm_{name}.npy", sq.cpu().numpy())

    mb = codes.numel() * codes.element_size() / 1e6
    print(f"{a.data.name}: n={n:,} d={d} metric={man['metric']}  kind={kind}")
    print(f"  codes {tuple(codes.shape)} {codes.dtype}, {mb:,.0f} MB "
          f"({100*mb/(n*d*4/1e6):.1f}% of the fp32 corpus)"
          + (f" plus {4*n/1e6:.0f} MB of per-node scales" if alpha is not None else "")
          + ("" if ip else f" plus {4*n/1e6:.0f} MB of squared norms"))
    print(f"  built in {secs:.1f}s; {len(idx):,} nodes verified {checked}")
    (a.data / f"codes_{name}.json").write_text(json.dumps(
        dict(n=n, d=d, kind=kind, metric=man["metric"],
             words=int(codes.shape[1]) if kind == "1bit" else None,
             scale=scale, has_sqnorm=not ip, has_alpha=alpha is not None,
             scoring="asymmetric: query fp32, stored side quantised",
             seconds=round(secs, 1), megabytes=round(mb)), indent=2))


if __name__ == "__main__":
    main()
