"""Place entries along the axes the query's own neighbourhood spreads on.

The query is a mixture and its answers sit in different parts of the space, each
scoring on its own dimensions -- which is why two pieces can be equally close to
the query and far from each other. That structure is visible without an encoder
and without the query text. Take the entries the catalogue scan already ranked,
remove the query direction from them, and the residual is not isotropic: on the
allentown query the leading axis carries 22x the variance isotropy would give,
and it separates the city passages (+0.307) from the weather passages (-0.223,
-0.276) with no overlap.

So the decomposition is a local PCA, and an entry "on the city side" is the best
catalogue entry for the probe normalize(q + lambda * axis). Both signs of each
axis are probed, because the query sits between the pieces and neither direction
is privileged.
"""
import argparse, json, sys, time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def local_axes(Qv, Vp, top, c):
    """The c leading directions of the top-M entries once the query is removed.

    Done through the M x M Gram matrix rather than an SVD of M x d, because M is
    small and d is not.
    """
    R = Vp[top]                                                     # [n, M, d]
    R = R - (R @ Qv.unsqueeze(2)) * Qv.unsqueeze(1)                 # drop q
    R = R - R.mean(1, keepdim=True)
    g = R @ R.transpose(1, 2)                                       # [n, M, M]
    w, U = torch.linalg.eigh(g)
    U = U[:, :, -c:].flip(-1)                                       # top c
    A = R.transpose(1, 2) @ U                                       # [n, d, c]
    return torch.nn.functional.normalize(A, dim=1), w.flip(-1)[:, :c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--M", type=int, default=64, help="entries the axes are fitted on")
    ap.add_argument("--axes", type=int, default=3)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.6, 1.0])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=512)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    with select_a.reduced_precision(True):
        sc = Q @ Vp.T if ip else -torch.cdist(Q, Vp)
    nearQ = torch.topk(sc, 64, dim=1).indices
    with torch.no_grad():
        tv = X[torch.from_numpy(gt.astype(np.int64)).cuda().reshape(-1)]
        dp = (1.0 - tv @ Vp.T) if ip else torch.cdist(tv, Vp)
        nearT = dp.argmin(1).view(NQ, a.k); del dp, tv

    # the probes: both signs of each local axis, tilted off the query
    t0 = time.perf_counter()
    topM = torch.topk(sc, a.M, dim=1).indices
    probes = []
    for s in range(0, NQ, a.chunk):
        e = min(s + a.chunk, NQ)
        A, _ = local_axes(Q[s:e], Vp, topM[s:e], a.axes)            # [n, d, c]
        qs = Q[s:e].unsqueeze(2)
        probes.append(torch.cat([qs + A, qs - A], dim=2))           # lam folded in later
    P_ax = torch.cat(probes)                                        # [NQ, d, 2c]
    torch.cuda.synchronize(); axes_ms = (time.perf_counter() - t0) * 1e3
    print(f"{a.data.name}  {NQ} queries, catalogue {a.P}, M={a.M}, {a.axes} axes\n"
          f"fitting the axes costs {axes_ms:.1f} ms for the batch "
          f"({1000*axes_ms/NQ:.1f} us a query)\n")

    def probe_entries(lam, nper):
        qs = Q.unsqueeze(2)
        A = (P_ax - qs) if lam == 1.0 else qs + lam * (P_ax - qs)
        A = torch.nn.functional.normalize(A if lam == 1.0 else A, dim=1)
        if lam != 1.0:
            A = torch.nn.functional.normalize(qs + lam * (P_ax - qs), dim=1)
        else:
            A = torch.nn.functional.normalize(P_ax, dim=1)
        out = []
        for j in range(A.shape[2]):
            s2 = A[:, :, j] @ Vp.T if ip else -torch.cdist(A[:, :, j], Vp)
            out.append(torch.topk(s2, nper, dim=1).indices)
        # round-robin the axes so each contributes before any repeats
        inter = torch.stack(out, dim=2).reshape(NQ, -1)
        return inter

    def fill(primary, V):
        out = torch.cat([primary, nearQ], 1)
        keep = out >= 0
        for j in range(1, out.shape[1]):
            keep[:, j] &= (out[:, j:j+1] != out[:, :j]).all(1)
        rank = torch.where(keep, torch.arange(out.shape[1], device=out.device).expand_as(out),
                           torch.full_like(out, 10**6))
        return torch.gather(out, 1, torch.argsort(rank, 1)[:, :V])

    def run(seeds):
        ids, _, st = engine.search(X, G, Q, pt[seeds.clamp(min=0)].to(torch.int32),
                                   k=a.k, L=a.L, B=a.B, metric=metric,
                                   team_size=a.team, stop_eps=a.stop_eps,
                                   shared_cache=True, shared_bound=True,
                                   exclusive=True, elide_visited=True)
        got = ids.cpu().numpy()
        return (np.array([len(gts[q] & set(int(v) for v in got[q] if v >= 0)) / a.k
                          for q in range(NQ)]),
                float(st.dist_evals.sum(1).float().mean()))

    def sg(x, y):
        d = x - y; w = int((d > 0).sum()); l = int((d < 0).sum()); n = w + l
        if n == 0: return w, l, 1.0
        kk = min(w, l)
        return w, l, min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n)

    rows = []
    for V in a.V:
        base, de0 = run(nearQ[:, :V])
        orc, deo = run(fill(nearT, V))
        print(f"V={V}")
        print(f"  {'selector':44} {'recall':>8} {'dist/q':>8} {'vs base':>9} {'p':>9}")
        print(f"  {'nearest to the query':44} {base.mean():8.4f} {de0:8.0f}")
        for lam in a.lam:
            for nper in (1, 2):
                pq, de = run(fill(probe_entries(lam, nper), V))
                w, l, p = sg(pq, base)
                print(f"  {f'axes, lambda={lam}, {nper} entry per direction':44} "
                      f"{pq.mean():8.4f} {de:8.0f} {pq.mean()-base.mean():+9.4f} {p:9.1e}")
                rows.append(dict(V=V, lam=lam, nper=nper, recall=float(pq.mean()),
                                 delta=float(pq.mean()-base.mean()), p=float(p)))
        w, l, p = sg(orc, base)
        print(f"  {'beside each true answer (ceiling)':44} {orc.mean():8.4f} {deo:8.0f} "
              f"{orc.mean()-base.mean():+9.4f} {p:9.1e}\n")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, M=a.M, axes=a.axes,
                                         axes_ms=axes_ms, rows=rows), indent=2))


if __name__ == "__main__":
    main()
