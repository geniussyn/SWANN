"""Native cuVS CAGRA on the same vectors, as the thing to be faster than.

Two arms, because "faster than CAGRA" is two different questions:

  from_graph   CAGRA's search over *our* graph and our vectors. Graph held
               fixed, so the difference is the search engine and nothing else.
  built        CAGRA's own graph, built by CAGRA. This is the honest end-to-end
               comparison, and it is not an engine comparison: a better graph is
               part of what CAGRA is.

Runs in its own process. cuVS is installed under /opt/cuvsenv with its own numpy
and its own CUDA 12.9 libraries; torch here uses 12.8, and letting the two shadow
each other in one process is how a comparison becomes a bug report. The arms
therefore exchange numbers through .npy files and a shared GT, never a shared
process.

    PYTHONPATH=/opt/cuvsenv python bench/cagra_native.py
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from cuvs.neighbors import cagra

try:
    from cuvs.common import Resources
except ImportError:                                    # older layouts
    from pylibraft.common import DeviceResources as Resources
from pylibraft.common import device_ndarray


def per_query_recall(got, gt):
    k = gt.shape[1]
    return np.array([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, gt)])


def recall_at_k(got, gt):
    return float(per_query_recall(got, gt).mean())


def timed_search(sp, idx, qd, k, res, repeats):
    """Best of `repeats` after a warm-up. cuVS returns (distances, neighbors) --
    in that order, which is the opposite of the argument order of `search`."""
    best, out = None, None
    for r in range(repeats + 1):
        t = time.perf_counter()
        out = cagra.search(sp, idx, qd, k, resources=res)
        res.sync()
        ms = (time.perf_counter() - t) * 1e3
        if r:
            best = ms if best is None else min(best, ms)
    return best, out[1].copy_to_host().astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m-100k"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--graph", default="G.npy", help="graph file inside --data")
    ap.add_argument("--metric", default="inner_product")
    ap.add_argument("--half", action="store_true",
                    help="give CAGRA the same fp16 corpus our search reads")
    ap.add_argument("--arms", nargs="+", default=["from_graph", "built"])
    ap.add_argument("--itopk", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--width", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--algo", nargs="+", default=["single_cta", "multi_cta"])
    ap.add_argument("--subset", default=None, help="npy of query indices")
    ap.add_argument("--hard", default=None,
                    help="npy of indices to report recall on, with the batch kept whole")
    ap.add_argument("--save-perquery", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.subset:
        sel = np.load(a.subset)
        Q, gt = np.ascontiguousarray(Q[sel]), gt[sel]
    nq = Q.shape[0]
    hard_idx = np.load(a.hard) if a.hard else None
    res = Resources()
    if a.half:
        X = X.astype(np.float16)
        Q = Q.astype(np.float16)
    Xd, Qd = device_ndarray(X), device_ndarray(Q)

    print(f"{man['corpus']} n={man['n']} d={man['d']} deg={man['deg']}  "
          f"{a.split} q={nq} k={a.k} metric={a.metric} dtype={X.dtype}\n")

    indexes = {}
    keepalive = []          # from_graph and build do not copy: the caller owns the
                            # lifetime of the device arrays, and letting one go is an
                            # illegal access inside the next search, not an error here.
    if "from_graph" in a.arms:
        t = time.perf_counter()
        Gd = device_ndarray(np.ascontiguousarray(G.astype(np.uint32)))
        keepalive.append(Gd)
        indexes["from_graph"] = cagra.from_graph(Gd, Xd, metric=a.metric, resources=res)
        res.sync()
        print(f"from_graph: wrapped our deg={G.shape[1]} knn graph in {time.perf_counter()-t:.2f}s")
    if "built" in a.arms:
        t = time.perf_counter()
        ip = cagra.IndexParams(metric=a.metric,
                               graph_degree=man.get("deg", G.shape[1]),
                               intermediate_graph_degree=2 * man["deg"])
        indexes["built"] = cagra.build(ip, Xd, resources=res)
        res.sync()
        print(f"built:      CAGRA's own graph, degree {man['deg']}, in {time.perf_counter()-t:.2f}s")

    rows, perq = [], []
    print(f"\n{'arm':11s} {'algo':11s} {'itopk':>6s} {'width':>6s} {'recall@10':>10s} "
          f"{'ms(batch)':>10s} {'QPS':>9s}")
    for arm, idx in indexes.items():
        for algo in a.algo:
            for it in a.itopk:
                for w in a.width:
                    if algo == "multi_cta" and w == 1:
                        continue                        # multi_cta's width is its CTA count
                    try:
                        sp = cagra.SearchParams(itopk_size=it, search_width=w, algo=algo)
                        ms, nbrs = timed_search(sp, idx, Qd, a.k, res, a.repeats)
                    except Exception as e:              # noqa: BLE001
                        print(f"{arm:11s} {algo:11s} {it:6d} {w:6d}   skipped: {type(e).__name__}")
                        continue
                    pq = per_query_recall(nbrs, gt)
                    rec = float(pq.mean())
                    if a.save_perquery:
                        perq.append(pq)
                    rec_hard = (recall_at_k(nbrs[hard_idx], gt[hard_idx])
                                if hard_idx is not None else None)
                    rows.append(dict(arm=arm, algo=algo, itopk=it, width=w, recall=rec,
                                     recall_hard=rec_hard, batch_ms=ms, qps=nq / (ms / 1e3)))
                    print(f"{arm:11s} {algo:11s} {it:6d} {w:6d} {rec:10.4f} {ms:10.2f} "
                          f"{nq/(ms/1e3):9.0f}"
                          + (f" {rec_hard:9.4f}" if rec_hard is not None else ""))

    if a.save_perquery:
        np.savez(a.save_perquery, perq=np.array(perq),
                 labels=np.array([f"{r['algo']}_it{r['itopk']}_w{r['width']}" for r in rows]))
        print(f"\nwrote {a.save_perquery}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, nq=int(nq),
                                         metric=a.metric, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
