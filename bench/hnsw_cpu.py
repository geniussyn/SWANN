"""CPU HNSW, as the reference line every GPU number should be read against.

Not a like-for-like comparison and not pretending to be one: this box has 128
cores and one GPU, and the point of putting hnswlib on the same axes is to show
what the GPU is buying, in QPS, at the recall the application actually wants.
Everything else is held the same -- our vectors, our fp32 ground truth, the whole
query batch timed at once, recall@10.
"""
import argparse, json, time
from pathlib import Path

import hnswlib
import numpy as np


def per_query_recall(got, gt):
    k = gt.shape[1]
    return np.array([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, gt)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m-100k"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=32)
    ap.add_argument("--efc", type=int, default=200)
    ap.add_argument("--efs", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--threads", type=int, default=0, help="0 = all cores")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--hard", default=None)
    ap.add_argument("--save-perquery", type=Path, default=None,
                    help="npz of per-query recall for every efSearch row")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    import os
    threads = a.threads or os.cpu_count()
    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    hard_idx = np.load(a.hard) if a.hard else None
    n, d = X.shape
    nq = Q.shape[0]

    space = "ip" if man["metric"].startswith("ip") else "l2"
    idx = hnswlib.Index(space=space, dim=d)
    idx.init_index(max_elements=n, ef_construction=a.efc, M=a.M)
    idx.set_num_threads(threads)
    t = time.perf_counter()
    idx.add_items(X, np.arange(n))
    build_s = time.perf_counter() - t
    print(f"{man['corpus']} n={n} d={d}  {a.split} q={nq} k={a.k}  space={space}\n"
          f"hnswlib build: M={a.M} efc={a.efc} threads={threads} -> {build_s:.1f}s\n")

    print(f"{'efSearch':>9s} {'recall@10':>10s} {'ms(batch)':>10s} {'QPS':>9s}"
          + (f" {'hard@10':>9s}" if hard_idx is not None else ""))
    rows, perq = [], []
    for ef in a.efs:
        idx.set_ef(ef)
        best, lab = None, None
        for r in range(a.repeats + 1):
            t = time.perf_counter()
            lab, _ = idx.knn_query(Q, k=a.k, num_threads=threads)
            ms = (time.perf_counter() - t) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        pq = per_query_recall(lab.astype(np.int64), gt)
        rec = float(pq.mean())
        rh = float(pq[hard_idx].mean()) if hard_idx is not None else None
        perq.append(pq)
        rows.append(dict(algo="hnsw_cpu", itopk=ef, width=0, recall=rec, recall_hard=rh,
                         batch_ms=best, qps=nq / (best / 1e3)))
        print(f"{ef:9d} {rec:10.4f} {best:10.2f} {nq/(best/1e3):9.0f}"
              + (f" {rh:9.4f}" if rh is not None else ""))

    if a.save_perquery:
        Path(a.save_perquery).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.save_perquery, perq=np.array(perq),
                 labels=np.array([f"ef{r['itopk']}" for r in rows]))
        print(f"wrote {a.save_perquery}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, nq=int(nq),
                                         build_seconds=round(build_s, 1), M=a.M, efc=a.efc,
                                         threads=threads, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
