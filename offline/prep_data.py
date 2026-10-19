"""Build the 100K integration corpus: vectors, held-out queries, exact GT, graph.

Vectors come from the SYN lab's MS MARCO 1M embeddings (fp16, L2-normalised,
d=1024) and are cast to fp32 once, here, because every arm must search the same
bits: the engine's exact distance, the GT, and any native baseline all read this
one file. Nothing downstream re-normalises.

Queries are split into a validation half and a test half with a fixed seed and
written separately. tau, the candidate pool, and any other tuned quantity may
only ever see the validation half -- the design document says so and this is
where the separation is made physical.

The graph is an exact kNN graph at this stage. That is a deliberate stand-in: it
makes the engine measurable on real data today, and it is NOT a navigable graph
of the kind CAGRA builds, so no recall number from it should be read as a CAGRA
comparison. Swapping in a built graph changes this file and nothing else.
"""
import argparse, json, os
from pathlib import Path

import numpy as np
import torch

DEFAULT_OUT = Path(os.environ.get("GPUANN_DATA", "/dev/shm/gpuann"))
SYN_EMB = Path(os.environ.get("SYN_EMB_DIR", "/workspace/SYN/emb"))


def exact_topk(Qv, X, k, device="cuda", qchunk=1024, dchunk=200_000):
    """fp32 exhaustive inner-product top-k. The ruler is at least as precise as
    the thing it measures, so it is never computed in fp16."""
    nq, n = Qv.shape[0], X.shape[0]
    Qg = torch.from_numpy(Qv).to(device)
    bv = torch.full((nq, k), -3.4e38, device=device)
    bi = torch.zeros((nq, k), dtype=torch.int64, device=device)
    for s in range(0, n, dchunk):
        e = min(s + dchunk, n)
        Xg = torch.from_numpy(np.ascontiguousarray(X[s:e])).to(device)
        for qs in range(0, nq, qchunk):
            qe = min(qs + qchunk, nq)
            sim = Qg[qs:qe] @ Xg.T
            v, i = torch.topk(sim, min(k, e - s), dim=1)
            cv = torch.cat([bv[qs:qe], v], 1)
            ci = torch.cat([bi[qs:qe], i + s], 1)
            v2, p = torch.topk(cv, k, dim=1)
            bv[qs:qe], bi[qs:qe] = v2, torch.gather(ci, 1, p)
        del Xg
    return bi.cpu().numpy().astype(np.int32), bv.cpu().numpy()


def knn_graph(X, deg, device="cuda", chunk=20_000, dchunk=200_000):
    """Exact kNN out-edges, self excluded, ordered nearest first."""
    n = X.shape[0]
    G = np.empty((n, deg), dtype=np.int32)
    Xt = torch.from_numpy(X).to(device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = Xt[s:e] @ Xt.T
        sim[torch.arange(e - s, device=device), torch.arange(s, e, device=device)] = -3.4e38
        G[s:e] = torch.topk(sim, deg, dim=1).indices.cpu().numpy().astype(np.int32)
        del sim
    del Xt
    torch.cuda.empty_cache()
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--arm", default="vanilla")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = a.out or (DEFAULT_OUT / f"{a.corpus}-{a.n // 1000}k")
    out.mkdir(parents=True, exist_ok=True)

    docs = np.load(SYN_EMB / f"{a.corpus}__{a.arm}_docs.npy", mmap_mode="r")
    qrys = np.load(SYN_EMB / f"{a.corpus}__{a.arm}_queries.npy", mmap_mode="r")
    X = np.ascontiguousarray(docs[: a.n], dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)          # fp16 norms are ~1, not 1
    Qall = np.ascontiguousarray(qrys[:], dtype=np.float32)
    Qall /= np.linalg.norm(Qall, axis=1, keepdims=True)

    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(Qall.shape[0])
    half = Qall.shape[0] // 2
    val, test = np.sort(perm[:half]), np.sort(perm[half:])

    print(f"X {X.shape} {X.dtype}   queries val/test {val.size}/{test.size}")
    np.save(out / "X.npy", X)
    np.save(out / "Q_val.npy", Qall[val]); np.save(out / "Q_test.npy", Qall[test])
    np.save(out / "qidx_val.npy", val);    np.save(out / "qidx_test.npy", test)

    for name, Q in (("val", Qall[val]), ("test", Qall[test])):
        gi, gv = exact_topk(np.ascontiguousarray(Q), X, a.k)
        np.save(out / f"gt_{name}_ids.npy", gi)
        np.save(out / f"gt_{name}_sim.npy", gv)
        print(f"  gt_{name}: {gi.shape}")

    G = knn_graph(X, a.deg)
    np.save(out / "G.npy", G)
    print(f"  G {G.shape}")

    (out / "manifest.json").write_text(json.dumps({
        "corpus": a.corpus, "arm": a.arm, "n": int(a.n), "d": int(X.shape[1]),
        "deg": int(a.deg), "k": int(a.k), "seed": int(a.seed),
        "metric": "ip_on_l2_normalised", "graph": "exact_knn",
        "graph_note": "stand-in: not a navigable built graph, not a CAGRA comparison",
        "source": str(SYN_EMB / f"{a.corpus}__{a.arm}_docs.npy"),
    }, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
