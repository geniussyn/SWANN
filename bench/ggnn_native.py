"""GGNN on the same corpus, the same ground truth and the same protocol.

GGNN reads fvecs from disk and nothing else, so this writes the corpus out once
(cached beside the .npy it came from) and then measures exactly what the CAGRA
bench measures: build time once, then a search-effort sweep timed over the whole
query batch, recall against our own fp32 ground truth, and the per-query recall
vector the own-decile protocol needs.

The fvecs writer is the part that will be reused for GANNS and SONG, which take
the same format.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import ggnn


def write_fvecs(path: Path, X: np.ndarray):
    """int32 dimension then d float32, per vector. Written once and cached."""
    if path.exists() and path.stat().st_size == X.shape[0] * (4 + 4 * X.shape[1]):
        return path
    n, d = X.shape
    with open(path, "wb") as f:
        buf = np.empty((n, d + 1), dtype=np.float32)
        buf[:, 0] = np.frombuffer(np.int32(d).tobytes(), dtype=np.float32)[0]
        buf[:, 1:] = X
        buf.tofile(f)
    return path


def per_query_recall(got, gt):
    k = gt.shape[1]
    return np.array([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, gt)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m-100k"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--k-build", type=int, default=24,
                    help="GGNN's per-node neighbour count; its graph is k_build*2 wide")
    ap.add_argument("--tau-build", type=float, default=0.5)
    ap.add_argument("--refine", type=int, default=2)
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    ap.add_argument("--measure", default="cosine", choices=["cosine", "euclidean"])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--hard", default=None)
    ap.add_argument("--save-perquery", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    hard_idx = np.load(a.hard) if a.hard else None
    nq = Q.shape[0]

    xf = write_fvecs(a.data / "X.fvecs", X)
    qf = write_fvecs(a.data / f"Q_{a.split}.fvecs", Q)
    base = ggnn.FloatDataset.load(str(xf))
    query = ggnn.FloatDataset.load(str(qf))
    measure = (ggnn.DistanceMeasure.Cosine if a.measure == "cosine"
               else ggnn.DistanceMeasure.Euclidean)

    g = ggnn.GGNN()
    g.set_base(base)
    g.set_return_results_on_gpu(False)
    torch.cuda.synchronize(); t = time.perf_counter()
    g.build(a.k_build, a.tau_build, a.refine, measure)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t
    print(f"{man['corpus']} n={man['n']} d={man['d']}  {a.split} q={nq} k={a.k}\n"
          f"GGNN build: k_build={a.k_build} tau_build={a.tau_build} "
          f"refine={a.refine} -> {build_s:.2f}s\n")

    print(f"{'tau_query':>10s} {'recall@10':>10s} {'ms(batch)':>10s} {'QPS':>9s}"
          + (f" {'hard@10':>9s}" if hard_idx is not None else ""))
    rows, perq = [], []
    for tau in a.taus:
        best, out = None, None
        for r in range(a.repeats + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            out = g.query(query, a.k, tau, 400, measure)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        nb = np.asarray(out[0])[:, : a.k].astype(np.int64)
        pq = per_query_recall(nb, gt)
        rec, rh = float(pq.mean()), (float(pq[hard_idx].mean()) if hard_idx is not None else None)
        perq.append(pq)
        rows.append(dict(algo="ggnn", itopk=tau, width=0, recall=rec, recall_hard=rh,
                         batch_ms=best, qps=nq / (best / 1e3)))
        print(f"{tau:10.2f} {rec:10.4f} {best:10.2f} {nq/(best/1e3):9.0f}"
              + (f" {rh:9.4f}" if rh is not None else ""))

    if a.save_perquery:
        np.savez(a.save_perquery, perq=np.array(perq),
                 labels=np.array([f"tau{r['itopk']}" for r in rows]))
        print(f"\nwrote {a.save_perquery}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, nq=int(nq),
                                         build_seconds=round(build_s, 2),
                                         k_build=a.k_build, tau_build=a.tau_build,
                                         measure=a.measure, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
