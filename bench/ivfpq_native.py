"""cuVS IVF-PQ with exact refinement, as the quantized-GPU baseline.

cuVS 26.10 has no RaBitQ, so this is the nearest thing it does have: an inverted
file over coarse centroids, product-quantized residuals for the scan, and an
exact fp32 refinement of the shortlist. The shape of the pipeline -- coarse
probe, cheap approximate scan, exact rerank -- is the one RaBitQ pipelines use;
the code is PQ, and this file says so rather than borrowing the name.

Two knobs are swept together because they trade against each other: n_probes
decides how much of the corpus is scanned, and the refinement ratio decides how
much of that scan survives into an exact comparison.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from cuvs.neighbors import ivf_pq, refine

try:
    from cuvs.common import Resources
except ImportError:
    from pylibraft.common import DeviceResources as Resources
from pylibraft.common import device_ndarray


def per_query_recall(got, gt):
    k = gt.shape[1]
    return np.array([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, gt)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lists", type=int, default=0, help="0 = ~sqrt(n)")
    ap.add_argument("--pq-dim", type=int, default=0, help="0 = cuVS default")
    ap.add_argument("--probes", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256])
    ap.add_argument("--ratios", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    nq, n = Q.shape[0], X.shape[0]
    lists = a.lists or int(np.sqrt(n)) * 2
    res = Resources()
    Xd, Qd = device_ndarray(X), device_ndarray(Q)

    kw = dict(n_lists=lists, metric="inner_product")
    if a.pq_dim:
        kw["pq_dim"] = a.pq_dim
    t = time.perf_counter()
    idx = ivf_pq.build(ivf_pq.IndexParams(**kw), Xd, resources=res)
    res.sync()
    build_s = time.perf_counter() - t
    print(f"{man['corpus']} n={n} d={man['d']}  {a.split} q={nq} k={a.k}\n"
          f"IVF-PQ build: n_lists={lists} -> {build_s:.2f}s\n")

    print(f"{'probes':>7s} {'refine':>7s} {'recall@10':>10s} {'ms(batch)':>10s} {'QPS':>9s}")
    rows = []
    for ratio in a.ratios:
        kk = a.k * ratio
        for p in a.probes:
            sp = ivf_pq.SearchParams(n_probes=p)
            best, out = None, None
            for r in range(a.repeats + 1):
                t0 = time.perf_counter()
                _, cand = ivf_pq.search(sp, idx, Qd, kk, resources=res)
                if ratio > 1:
                    out = refine(Xd, Qd, cand, a.k, metric="inner_product", resources=res)[1]
                else:
                    out = cand
                res.sync()
                ms = (time.perf_counter() - t0) * 1e3
                if r:
                    best = ms if best is None else min(best, ms)
            nb = np.asarray(out.copy_to_host())[:, : a.k].astype(np.int64)
            rec = float(per_query_recall(nb, gt).mean())
            rows.append(dict(algo="ivf_pq", itopk=p, width=ratio, recall=rec,
                             recall_hard=None, batch_ms=best, qps=nq / (best / 1e3)))
            print(f"{p:7d} {ratio:7d} {rec:10.4f} {best:10.2f} {nq/(best/1e3):9.0f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, nq=int(nq),
                                         build_seconds=round(build_s, 2), n_lists=lists,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
