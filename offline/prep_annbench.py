"""Prepare an ann-benchmarks HDF5 corpus the way prep_data.py prepares the lab's.

Same contract as the MS MARCO path: fp32 vectors that every arm reads, a query
split fixed by seed, exact fp32 ground truth recomputed here rather than taken
from the file, and a manifest. The file's own `neighbors` are not used -- they
were computed by somebody else's code at somebody else's precision, and the
ruler has to be ours.

Two details the kernel cares about:

  dimension   padded up to a multiple of 8 with zeros. The distance loop reads
              16 bytes at a time -- 4 floats or 8 halves -- and a row whose
              length is not a multiple of 8 would have its last load reach past
              the row. Zeros contribute nothing to an inner product and nothing
              to a squared difference against a zero-padded query.
  metric      angular corpora are L2-normalised and searched by inner product,
              which is the same ordering. Euclidean corpora keep their scale and
              are searched by squared L2.
"""
import argparse, json
from pathlib import Path

import h5py
import numpy as np
import torch

from prep_data import exact_topk, exact_topk_l2, knn_graph   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", type=Path, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--metric", default=None, choices=["ip", "l2"],
                    help="default: read it off the file name (angular -> ip)")
    ap.add_argument("--n", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-degenerate", action="store_true",
                    help="keep zero-length rows; they are dropped by default "
                         "because cosine does not define their neighbours")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    name = a.name or a.h5.stem
    metric = a.metric or ("ip" if "angular" in a.h5.name else "l2")
    out = a.out or Path("/dev/shm/swann") / name
    out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.h5, "r") as f:
        X = np.asarray(f["train"], dtype=np.float32)
        Qall = np.asarray(f["test"], dtype=np.float32)
    if a.n:
        X = X[: a.n]
    d0 = X.shape[1]
    pad = (-d0) % 8
    if pad:
        X = np.pad(X, ((0, 0), (0, pad)))
        Qall = np.pad(Qall, ((0, 0), (0, pad)))
    if metric == "ip":
        xn = np.linalg.norm(X, axis=1, keepdims=True)
        qn = np.linalg.norm(Qall, axis=1, keepdims=True)
        if not a.keep_degenerate:
            # A zero-length row has no direction, so cosine says nothing about
            # it: its similarity to everything is 0, its place in a ground-truth
            # list is arbitrary, and a clustering index collects all of them in
            # one cell that then pollutes every probe that reaches it. On
            # ann-benchmarks' nytimes-256 that is 239 documents and 7 queries,
            # enough to make IVF recall fall as the probe count rises. They are
            # dropped rather than kept at zero.
            keep_x = (xn[:, 0] > 1e-12)
            keep_q = (qn[:, 0] > 1e-12)
            nd, nq = int((~keep_x).sum()), int((~keep_q).sum())
            if nd or nq:
                print(f"  dropped {nd:,} zero-length documents and {nq:,} queries")
            X, xn = X[keep_x], xn[keep_x]
            Qall, qn = Qall[keep_q], qn[keep_q]
        X /= np.maximum(xn, 1e-12)
        Qall /= np.maximum(qn, 1e-12)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Qall = np.ascontiguousarray(Qall, dtype=np.float32)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(Qall.shape[0])
    half = Qall.shape[0] // 2
    val, test = np.sort(perm[:half]), np.sort(perm[half:])
    print(f"{name}: X {X.shape} ({d0} -> {X.shape[1]} padded), metric {metric}, "
          f"queries val/test {val.size}/{test.size}")

    np.save(out / "X.npy", X)
    np.save(out / "Q_val.npy", Qall[val]); np.save(out / "Q_test.npy", Qall[test])
    np.save(out / "qidx_val.npy", val);    np.save(out / "qidx_test.npy", test)
    topk = exact_topk if metric == "ip" else exact_topk_l2
    for split, Q in (("val", Qall[val]), ("test", Qall[test])):
        gi, gv = topk(np.ascontiguousarray(Q), X, a.k)
        np.save(out / f"gt_{split}_ids.npy", gi); np.save(out / f"gt_{split}_sim.npy", gv)
        print(f"  gt_{split}: {gi.shape}")
    if metric == "ip":
        G = knn_graph(X, a.deg)
    else:
        # A kNN graph by L2; the builder replaces it, this is only the stand-in.
        import torch as _t
        n = X.shape[0]
        G = np.empty((n, a.deg), dtype=np.int32)
        Xt = _t.from_numpy(X).cuda(); xn = (Xt * Xt).sum(1)
        for s2 in range(0, n, 10000):
            e2 = min(s2 + 10000, n)
            sc = xn.view(1, -1) - 2.0 * (Xt[s2:e2] @ Xt.T)
            sc[_t.arange(e2 - s2, device=sc.device), _t.arange(s2, e2, device=sc.device)] = float("inf")
            G[s2:e2] = _t.topk(sc, a.deg, dim=1, largest=False).indices.cpu().numpy().astype(np.int32)
            del sc
        del Xt; _t.cuda.empty_cache()
    np.save(out / "G_knn.npy", G)
    np.save(out / "G.npy", G)
    print(f"  G {G.shape} (exact kNN stand-in; build_graph.py writes the real one)")

    (out / "manifest.json").write_text(json.dumps({
        "corpus": name, "n": int(X.shape[0]), "d": int(X.shape[1]), "d_original": int(d0),
        "deg": int(a.deg), "k": int(a.k), "seed": int(a.seed),
        "metric": "ip_on_l2_normalised" if metric == "ip" else "sqeuclidean",
        "graph": "exact_knn", "source": str(a.h5),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
