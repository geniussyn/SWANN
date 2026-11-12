"""What each index costs to build and to keep.

The search comparison hides two costs a deployment cannot ignore: the time
before the first query can be answered, and the bytes the index occupies beside
the vectors. They are reported against the raw vectors, which every method
needs unless it discards them -- and discarding them is exactly the trade
IVF-PQ without refinement makes, which is why it appears in both tables.

Build time is wall clock on an idle device, measured once (these are minutes,
not microseconds, and the variance does not change a conclusion). Storage is
computed from the structure each method actually stores, not from a process's
resident set, which would charge every method for the same copy of the vectors.
"""
import argparse, json, time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = "inner_product" if ip else "sqeuclidean"
    X = np.load(a.data / "X.npy")
    n, d = X.shape
    rows = []

    raw = n * d * 4
    rows.append(dict(system="Raw vectors (fp32)", build_s=0.0,
                     index_bytes=0, total_bytes=raw, keeps_vectors=True))
    rows.append(dict(system="Raw vectors (fp16)", build_s=0.0,
                     index_bytes=0, total_bytes=n * d * 2, keeps_vectors=True))

    G = np.load(a.data / a.graph, mmap_mode="r")
    deg = G.shape[1]
    # the graph is int32 neighbour ids; SWANN keeps the vectors as they are
    rows.append(dict(system="SWANN", build_s=None, index_bytes=n * deg * 4,
                     total_bytes=raw + n * deg * 4, keeps_vectors=True,
                     note=f"degree {deg}"))

    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_flat, ivf_pq, cagra
    Xd = device_ndarray(X)

    t = time.perf_counter()
    ci = cagra.build(cagra.IndexParams(metric=metric,
                                       graph_degree=man.get("deg", 64)), Xd)
    cagra_s = time.perf_counter() - t
    cdeg = man.get("deg", 64)
    rows.append(dict(system="CAGRA", build_s=cagra_s, index_bytes=n * cdeg * 4,
                     total_bytes=raw + n * cdeg * 4, keeps_vectors=True,
                     note=f"degree {cdeg}"))
    del ci

    t = time.perf_counter()
    fi = ivf_flat.build(ivf_flat.IndexParams(n_lists=a.lists, metric=metric), Xd)
    flat_s = time.perf_counter() - t
    # a copy of the vectors arranged by list, plus the centroids and the ids
    flat_idx = n * d * 4 + a.lists * d * 4 + n * 8
    rows.append(dict(system="IVF-FLAT", build_s=flat_s, index_bytes=flat_idx,
                     total_bytes=flat_idx, keeps_vectors=True))
    del fi

    t = time.perf_counter()
    pi = ivf_pq.build(ivf_pq.IndexParams(n_lists=a.lists, metric=metric,
                                         pq_bits=8), Xd)
    pq_s = time.perf_counter() - t
    pq_dim = max(1, d // 4)
    pq_idx = n * pq_dim + a.lists * d * 4 + n * 8
    rows.append(dict(system="IVF-PQ (wo/ref)", build_s=pq_s, index_bytes=pq_idx,
                     total_bytes=pq_idx, keeps_vectors=False,
                     note=f"pq_dim {pq_dim}"))
    rows.append(dict(system="IVF-PQ (w/ref)", build_s=pq_s, index_bytes=pq_idx,
                     total_bytes=pq_idx + raw, keeps_vectors=True,
                     note="refinement re-reads the vectors"))
    del pi

    for r in rows:
        r["n"], r["d"], r["corpus"] = n, d, a.corpus
        r["bytes_per_vector"] = r["total_bytes"] / n
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(corpus=a.corpus, n=n, d=d, rows=rows), indent=2))
    print(f"{a.corpus}: n={n:,} d={d}")
    print(f"{'system':22} {'build (s)':>10} {'index GB':>10} {'total GB':>10} "
          f"{'B / vector':>11}")
    for r in rows:
        b = f"{r['build_s']:10.1f}" if r["build_s"] is not None else f"{'--':>10}"
        print(f"{r['system']:22} {b} {r['index_bytes']/1e9:10.2f} "
              f"{r['total_bytes']/1e9:10.2f} {r['bytes_per_vector']:11,.0f}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
