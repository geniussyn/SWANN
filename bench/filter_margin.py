"""The one constant a threshold filter needs: how far the code can under-estimate.

Selecting the best m of D needs a block-wide ranking; a threshold needs nothing
but the queue's worst distance, which the kernel already has in hand, so the
filter becomes a per-team early-out and no second pass. The cost is a constant:
a candidate is skipped when its approximate distance exceeds worst + margin, so
margin has to cover how much the code can over-state the distance of a candidate
that the exact test would have admitted.

margin >= approx - true, over the candidates that pass the exact test. Measured
as a distribution rather than a maximum, because the maximum over a sample is
not a bound either and a quantile says what recall a given margin risks.
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
    ap.add_argument("--codes", default=None, help="default: 1bit or int8 by metric")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=400)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    kind = a.codes or ("1bit" if ip else "int8")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")[: a.nq]).cuda()
    dim = X.shape[1]
    D = G.shape[1]
    mu = torch.from_numpy(np.load(a.data / f"mu_{kind}.npy")).cuda()
    meta = json.loads((a.data / f"codes_{kind}.json").read_text())
    cd = np.load(a.data / f"codes_{kind}.npy")
    if kind == "1bit":
        words = torch.from_numpy(cd).cuda()
        bits = torch.arange(32, device=X.device, dtype=torch.int32)

        alpha = torch.from_numpy(np.load(a.data / f"alpha_{kind}.npy")).cuda()

        def decode(ids):
            # +-1 bits scaled by the node's own ||c||_1 / d, which puts the
            # estimate in the corpus's units instead of the bits' units
            w = words[ids]
            b = ((w.unsqueeze(-1) >> bits) & 1).view(*ids.shape, -1)[..., :dim]
            return (b.float() * 2.0 - 1.0) * alpha[ids].unsqueeze(-1)
        scale = 1.0
    else:
        codes = torch.from_numpy(cd).cuda()
        scale = float(meta["scale"])

        def decode(ids):
            return codes[ids].float() * scale
    sq = (torch.from_numpy(np.load(a.data / f"sqnorm_{kind}.npy")).cuda()
          if not ip else None)

    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)]
    ids, dist, st = engine.search(X, G, Q, seeds.to(torch.int32), k=a.k, L=a.L,
                                  B=a.B, metric=metric, team_size=a.team,
                                  stop_eps=a.stop_eps, shared_cache=True,
                                  shared_bound=True, exclusive=True,
                                  elide_visited=True)
    worst = dist[:, a.k - 1].clone()
    held = st.ids[:, :, : a.L].reshape(a.nq, -1)
    g = torch.Generator(device="cuda").manual_seed(0)
    pick = torch.randint(0, held.shape[1], (a.nq, 8), generator=g, device="cuda")
    src = torch.gather(held, 1, pick).clamp(min=0)

    gaps, tds = [], []
    for s in range(0, a.nq, 32):
        e = min(s + 32, a.nq)
        nodes = src[s:e].reshape(-1)
        nb = G[nodes].long()
        valid = nb >= 0
        nbc = nb.clamp(min=0)
        qv = Q[s:e].repeat_interleave(8, dim=0)
        NV = X[nbc]
        if ip:
            true_d = 1.0 - torch.einsum("md,mkd->mk", qv, NV)
            approx = 1.0 - (torch.einsum("md,mkd->mk", qv, decode(nbc))
                            + (qv @ mu).view(-1, 1))
        else:
            true_d = ((NV - qv.unsqueeze(1)) ** 2).sum(2)
            approx = (sq[nbc] - 2.0 * (torch.einsum("md,mkd->mk", qv, decode(nbc))
                                       + (qv @ mu).view(-1, 1))
                      + (qv * qv).sum(1).view(-1, 1))
        w = worst[s:e].repeat_interleave(8).view(-1, 1)
        adm = valid & (true_d <= w)
        gaps.append((approx - true_d)[adm].cpu())
        tds.append(true_d[valid].cpu())
        del nb, NV, true_d, approx
    gaps = torch.cat(gaps).numpy()
    tds = torch.cat(tds).numpy()
    print(f"{a.data.name}  metric={man['metric']}  codes={kind}  d={dim}, D={D}")
    print(f"{len(gaps):,} candidates the exact test admits\n")
    print(f"the code's over-statement, approx - true, on those candidates:")
    for qq in (50, 90, 99, 99.9, 100):
        print(f"  p{qq:<5} {np.percentile(gaps, qq):+.5f}")
    print(f"\nfor scale, the true distances a search sees: "
          f"p1 {np.percentile(tds,1):.4f}  p50 {np.percentile(tds,50):.4f}")
    print(f"\nwhat a margin costs and buys")
    print(f"  {'margin':>9} {'admitted kept':>14} {'of all D skipped':>18}")
    for mg in (0.0, 0.01, 0.02, 0.05, 0.10, 0.20, 0.40):
        kept = float((gaps <= mg).mean())
        print(f"  {mg:9.2f} {kept:13.2%} {'':>18}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, kind=kind,
                                         n=len(gaps),
                                         p99=float(np.percentile(gaps, 99)),
                                         p999=float(np.percentile(gaps, 99.9)),
                                         pmax=float(gaps.max())), indent=2))


if __name__ == "__main__":
    main()
