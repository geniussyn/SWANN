"""Could a cheap code decide which neighbours are worth reading in full?

Expansion is bandwidth bound at 1511 GB/s and 87% of the bytes it reads are for
neighbours that fail the queue's test. The fix is not to read faster but to
decide without reading: score the D neighbours from a small per-node code, keep
the ones that might pass, and load full vectors only for those. Whether that
works is one question -- does the code preserve the decision -- and it can be
answered before any kernel is written.

The kernel's actual test is the radius: a candidate enters if its distance is at
most the queue's kstop-th best. So for sampled (query, node) pairs this takes the
true set of neighbours under that radius, ranks the same neighbours by each
code's approximate score, and reports how much of the true set survives when
only the best m are passed through. The read volume is then m/D plus the code
itself, which is what decides whether the trade is worth making.
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
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest16k")
    ap.add_argument("--P", type=int, default=16384)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=400)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")[: a.nq]).cuda()
    n, dim = X.shape
    D = G.shape[1]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)]
    ids, dist, st = engine.search(X, G, Q, seeds.to(torch.int32), k=a.k, L=a.L,
                                  B=a.B, metric=metric,
                                  team_size=a.team, stop_eps=a.stop_eps,
                                  shared_cache=True, shared_bound=True,
                                  exclusive=True, elide_visited=True)
    # the radius the kernel compares against: the query's k-th best distance
    radius = dist[:, a.k - 1].clone()
    # nodes the search had in hand, as stand-ins for the ones it expanded
    held = st.ids[:, :, : a.L].reshape(a.nq, -1)

    mu = X.mean(0)                                        # the code's origin
    print(f"{a.data.name}  d={dim}, D={D}, radius = the query's {a.k}-th best\n")

    def codes_for(nb, kind):
        V = X[nb]                                         # [m, D, d]
        if kind == "1bit":
            return torch.sign(V - mu), None
        if kind == "int8":
            lo, hi = (V - mu).aminmax()
            sc = (hi - lo) / 255.0
            return torch.round((V - mu - lo) / sc) * sc + lo, None
        if kind == "fp16":
            return (V - mu).half().float(), None
        raise ValueError(kind)

    print(f"{'code':>7} {'bytes/dim':>10} {'pass m':>7} {'true kept':>10} "
          f"{'read vs now':>12}")
    rows = []
    g = torch.Generator(device="cuda").manual_seed(0)
    pick = torch.randint(0, held.shape[1], (a.nq, 8), generator=g, device="cuda")
    src = torch.gather(held, 1, pick).clamp(min=0)         # [nq, 8] nodes
    for kind, bpd in (("1bit", 1 / 8), ("int8", 1.0), ("fp16", 2.0)):
        for m in (8, 16, 24, 32):
            kept = tot = 0
            for s in range(0, a.nq, 64):
                e = min(s + 64, a.nq)
                nodes = src[s:e].reshape(-1)
                nb = G[nodes].long()                       # [b*8, D]
                valid = nb >= 0
                nbc = nb.clamp(min=0)
                qv = Q[s:e].repeat_interleave(8, dim=0)    # [b*8, d]
                NV = X[nbc]                                # [m, D, d]
                if ip:
                    true_d = 1.0 - torch.einsum("md,mkd->mk", qv, NV)
                else:
                    true_d = ((NV - qv.unsqueeze(1)) ** 2).sum(2)
                rad = radius[s:e].repeat_interleave(8).view(-1, 1)
                is_true = valid & (true_d <= rad)
                C, _ = codes_for(nbc, kind)
                # The code stands for v - mu, so the query is used as it is:
                # <q, v> = <q, v - mu> + <q, mu> and the second term is the same
                # for every candidate of a query, so it cannot change the order.
                # Centring the query as well would add -<mu, v - mu>, which does
                # vary with v -- harmless on a normalised corpus where |mu| is
                # small, wrong on one where it is not.
                inner = torch.einsum("md,mkd->mk", qv, C)
                if ip:
                    approx = -inner
                else:
                    # d^2 = |v|^2 - 2<q, v> + |q|^2; |v|^2 is one scalar a node
                    # can store exactly beside its code, |q|^2 is a constant.
                    approx = (NV * NV).sum(2) - 2.0 * inner
                approx = torch.where(valid, approx, torch.full_like(approx, 1e30))
                sel = torch.topk(approx, min(m, D), dim=1, largest=False).indices
                keptmask = torch.zeros_like(is_true)
                keptmask.scatter_(1, sel, True)
                kept += int((is_true & keptmask).sum())
                tot += int(is_true.sum())
                del nb, C, approx, true_d
            frac = kept / max(tot, 1)
            read = (bpd * D + 4.0 * m) / (4.0 * D)
            print(f"{kind:>7} {bpd:10.3f} {m:7} {frac:9.1%} {read:11.2f}x")
            rows.append(dict(code=kind, bytes_per_dim=bpd, m=m, kept=frac,
                             read_ratio=read))
        print()
    print(f"'read vs now' is the bytes an expansion would move against reading all\n"
          f"{D} neighbours in fp32: the code for every neighbour plus full vectors\n"
          f"for the m that pass. The fan-out actually needed is 12.5.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, D=D, dim=dim,
                                         rows=rows), indent=2))


if __name__ == "__main__":
    main()
