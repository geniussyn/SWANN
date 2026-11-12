"""Does the local subspace signal exist at all, and is it RESOLUTION that kills it?

Two arms of the same mechanism were measured and disagree:

  PRF pass-1 (bench/subspace_seed.py)   axes fitted on the top-m=50 CORPUS
                                        documents a pass-1 search returned.
                                        V=2: recall 0.9212 -> 0.9585.
  local PCA (bench/subspace_select.py)  axes fitted on the top-M=64 POOL
                                        entries the catalogue scan already has.
                                        Null at every V in {4,8,16,32}.

The mechanism is identical -- drop q, PCA the residual, probe q +- lambda*axis,
take the nearest pool entry -- so the difference is what the axes are fitted on.
At P=4096 over 1M nodes the nearest pool node to an answer is that answer's
~170th neighbour; at P=131072 it is the ~7th. So the pool is a coarse sample of
the neighbourhood and its leading axes may be estimating the spread of the POOL
around q rather than the spread of the ANSWERS.

This holds everything else fixed and moves only the fitting set, using the exact
top-m corpus neighbours -- no index, no pass-1 search, free of any budget
argument because it is a bound and not a method. Entries are still chosen from
the pool, so the output is comparable row for row with subspace_select.py.

If corpus resolution recovers the PRF gain, the signal is real and the question
becomes how to estimate the axes cheaply -- a denser catalogue (the one entry-side
intervention that ever worked) or the 1-bit RaBitQ scan already in the tree.
If it does not, the axes are not the mechanism and PRF is winning on its extra
distance evaluations alone.
"""
import argparse, json, sys, time
from math import comb
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.subspace_select import local_axes                         # noqa: E402


def exact_topm(X, Q, m, ip, chunk=256):
    """The m nearest corpus documents of each query, exactly. A bound, not a method."""
    out = []
    for s in range(0, Q.shape[0], chunk):
        sc = (Q[s:s + chunk] @ X.T) if ip else -torch.cdist(Q[s:s + chunk], X)
        out.append(torch.topk(sc, m, dim=1).indices)
        del sc
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[4, 8, 16, 32])
    ap.add_argument("--m", type=int, nargs="+", default=[10, 50],
                    help="corpus neighbours the axes are fitted on")
    ap.add_argument("--M", type=int, default=64, help="pool entries, for the pool arm")
    ap.add_argument("--axes", type=int, default=3)
    ap.add_argument("--lam", type=float, nargs="+", default=[0.3, 0.6])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
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
        sc = (Q @ Vp.T) if ip else -torch.cdist(Q, Vp)
    nearQ = torch.topk(sc, 64, dim=1).indices
    with torch.no_grad():
        tv = X[torch.from_numpy(gt.astype(np.int64)).cuda().reshape(-1)]
        dp = (1.0 - tv @ Vp.T) if ip else torch.cdist(tv, Vp)
        nearT = dp.argmin(1).view(NQ, a.k); del dp, tv

    print(f"{a.data.name} {a.split} {NQ} queries, pool {a.pool} P={a.P}, "
          f"{a.axes} axes, k={a.k}\n")

    # the fitting sets: the pool entries the scan has, and the exact corpus top-m
    fits = {}
    t0 = time.perf_counter()
    fits["pool-%d" % a.M] = (torch.topk(sc, a.M, dim=1).indices, Vp)
    wide = exact_topm(X, Q, max(a.m) + a.k, ip)
    for m in a.m:
        # top-m exactly. At m = k this IS the ground truth -- circular, a bound
        # on the geometry and nothing more.
        fits["corpus-%d" % m] = (wide[:, :m], X)
        # the same m neighbours with the k answers removed. Not circular: ranks
        # k..k+m hold no answer, so a gain here is the neighbourhood's structure
        # and is estimable by any approximate retrieval.
        fits["corpus-%d-nogt" % m] = (wide[:, a.k:a.k + m], X)
    torch.cuda.synchronize()
    print(f"fitting sets built in {time.perf_counter()-t0:.1f}s "
          f"(the corpus ones are a bound, not a method)\n")

    def axes_for(name, chunk=512):
        idx, src = fits[name]
        out = []
        for s in range(0, NQ, chunk):
            e = min(s + chunk, NQ)
            A, _ = local_axes(Q[s:e], src, idx[s:e], a.axes)
            qs = Q[s:e].unsqueeze(2)
            out.append(torch.cat([qs + A, qs - A], dim=2))
        return torch.cat(out)

    def probe_entries(P_ax, lam, nper):
        qs = Q.unsqueeze(2)
        A = torch.nn.functional.normalize(qs + lam * (P_ax - qs), dim=1)
        out = []
        for j in range(A.shape[2]):
            s2 = (A[:, :, j] @ Vp.T) if ip else -torch.cdist(A[:, :, j], Vp)
            out.append(torch.topk(s2, nper, dim=1).indices)
        return torch.stack(out, dim=2).reshape(NQ, -1)

    def fill(primary, V):
        out = torch.cat([primary, nearQ], 1)
        keep = out >= 0
        for j in range(1, out.shape[1]):
            keep[:, j] &= (out[:, j:j + 1] != out[:, :j]).all(1)
        rank = torch.where(keep, torch.arange(out.shape[1], device=out.device)
                           .expand_as(out), torch.full_like(out, 10 ** 6))
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
        if n == 0:
            return w, l, 1.0
        kk = min(w, l)
        return w, l, min(1.0, 2 * sum(comb(n, i) for i in range(kk + 1)) / 2 ** n)

    ax = {n: axes_for(n) for n in fits}
    rows = []
    for V in a.V:
        base, de0 = run(nearQ[:, :V])
        orc, deo = run(fill(nearT, V))
        print(f"V={V}")
        print(f"  {'axes fitted on':24} {'lam':>5} {'n/dir':>5} {'recall':>8} "
              f"{'dist/q':>8} {'vs base':>9} {'p':>9}")
        print(f"  {'nearest to the query':24} {'':>5} {'':>5} {base.mean():8.4f} {de0:8.0f}")
        rows.append(dict(V=V, fit="base", lam=0, nper=0, recall=float(base.mean()),
                         dist=de0, delta=0.0, p=1.0))
        for name in fits:
            for lam in a.lam:
                for nper in (1, 2):
                    pq, de = run(fill(probe_entries(ax[name], lam, nper), V))
                    w, l, p = sg(pq, base)
                    print(f"  {name:24} {lam:5} {nper:5} {pq.mean():8.4f} {de:8.0f} "
                          f"{pq.mean()-base.mean():+9.4f} {p:9.1e}")
                    rows.append(dict(V=V, fit=name, lam=lam, nper=nper,
                                     recall=float(pq.mean()), dist=de,
                                     delta=float(pq.mean()-base.mean()), p=float(p)))
        w, l, p = sg(orc, base)
        print(f"  {'beside each answer (ceil)':24} {'':>5} {'':>5} {orc.mean():8.4f} "
              f"{deo:8.0f} {orc.mean()-base.mean():+9.4f} {p:9.1e}\n")
        rows.append(dict(V=V, fit="ceiling", lam=0, nper=0, recall=float(orc.mean()),
                         dist=deo, delta=float(orc.mean()-base.mean()), p=float(p)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         pool=a.pool, P=a.P, axes=a.axes, M=a.M,
                                         m=a.m, L=a.L, B=a.B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
