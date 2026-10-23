"""cuVS IVF-FLAT and IVF-PQ on our vectors and our ground truth.

These are the other two things a GPU index can be, and they bracket what
quantisation costs: IVF-FLAT keeps the vectors and pays for them, IVF-PQ
compresses them and pays in precision, and IVF-PQ with refinement buys the
precision back by re-ranking a longer candidate list against the real vectors.
Reporting PQ without that distinction is reporting whichever half flatters the
point being made, so both are swept.

Runs under PYTHONPATH=/opt/cuvsenv in its own process: cuVS ships for CUDA 12.9
with its own numpy, torch here is 12.8, and letting the two shadow each other in
one process is how a comparison turns into a bug report.
"""
import argparse, json, time
from pathlib import Path

import numpy as np


def recall_at_k(got, gt, k):
    return float(np.mean([len(set(g[:k].tolist()) & set(t[:k].tolist())) / k
                          for g, t in zip(got, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--pq-bits", type=int, default=8)
    ap.add_argument("--pq-dim", type=int, default=0, help="0 = cuVS decides")
    ap.add_argument("--refine", type=float, nargs="+", default=[2.0, 4.0],
                    help="candidate multiples re-ranked against the real vectors")
    ap.add_argument("--arms", nargs="+", default=["ivf_flat", "ivf_pq"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    # device_ndarray rather than cupy: cuVS ships pylibraft and this tree has
    # no cupy, and the harness beside this one already moves arrays this way.
    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_flat, ivf_pq, refine as cuvs_refine

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = "inner_product" if ip else "sqeuclidean"
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    nq = Q.shape[0]
    Xd, Qd = device_ndarray(X), device_ndarray(Q)
    print(f"{man['corpus']} n={man['n']:,} d={X.shape[1]} metric={metric} "
          f"q={nq:,} k={a.k}\n")

    rows = []
    n, dim = X.shape
    per_list = n / a.lists
    print(f"{'arm':22} {'probes':>7} {'recall':>8} {'ms':>9} {'QPS':>10} "
          f"{'GB/s*':>7}")
    for arm in a.arms:
        t = time.perf_counter()
        if arm == "ivf_flat":
            idx = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.lists, metric=metric), Xd)
        else:
            kw = dict(n_lists=a.lists, metric=metric, pq_bits=a.pq_bits)
            if a.pq_dim:
                kw["pq_dim"] = a.pq_dim
            idx = ivf_pq.build(ivf_pq.IndexParams(**kw), Xd)
        build_s = time.perf_counter() - t
        print(f"  {arm} built in {build_s:.1f}s")

        for pr in a.probes:
            sp = (ivf_flat.SearchParams(n_probes=pr) if arm == "ivf_flat"
                  else ivf_pq.SearchParams(n_probes=pr))
            variants = [(arm, 0.0)]
            if arm == "ivf_pq":
                variants = [(f"{arm} (wo/ref)", 0.0)] + \
                           [(f"{arm} (w/ref x{r:g})", r) for r in a.refine]
            for label, mult in variants:
                kk = a.k if mult == 0 else int(a.k * mult)
                best = None
                for _ in range(a.repeats + 1):
                    t0 = time.perf_counter()
                    if arm == "ivf_flat":
                        d, i = ivf_flat.search(sp, idx, Qd, kk)
                    else:
                        d, i = ivf_pq.search(sp, idx, Qd, kk)
                    if mult:
                        d, i = cuvs_refine(Xd, Qd, i, a.k, metric=metric)
                    ms = (time.perf_counter() - t0) * 1e3
                    best = ms if best is None else min(best, ms)
                got = np.asarray(i.copy_to_host())[:, : a.k]
                rec = recall_at_k(got, gt, a.k)
                # Modelled, not counted: cuVS exposes no traffic counter and
                # the profiler is unavailable on this machine, so the bytes are
                # what the structure forces. A probed list is scanned whole --
                # full vectors for IVF-FLAT, one byte per PQ subspace for
                # IVF-PQ -- plus the centroid scan every query pays, plus the
                # real vectors a refinement pass re-reads.
                scanned = pr * per_list * nq
                bytes_ = (a.lists * dim * 4 * nq                  # centroids
                          + scanned * (dim * 4 if arm == "ivf_flat"
                                       else (a.pq_dim or dim // 4)))
                if mult:
                    bytes_ += nq * kk * dim * 4                   # the re-read
                gbs = bytes_ / 1e9 / (best / 1e3)
                print(f"{label:22} {pr:7} {rec:8.4f} {best:9.2f} "
                      f"{nq/(best/1e3):10,.0f} {gbs:7.0f}")
                rows.append(dict(algo=label, n_probes=pr, lists=a.lists,
                                 refine=mult, recall=rec, batch_ms=best,
                                 qps=nq / (best / 1e3), build_seconds=build_s,
                                 modelled_gb_per_s=gbs))
        del idx
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k,
                                         nq=int(nq), rows=rows), indent=2))
        print(f"wrote {a.out}")
    print("\n* modelled from the index structure, not read from a counter")


if __name__ == "__main__":
    main()
