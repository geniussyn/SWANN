"""Prepare an fvecs corpus the way prep_annbench.py prepares an HDF5 one.

Same contract: fp32 vectors, a fixed query split, exact fp32 ground truth
computed here rather than taken from anybody else's file, a manifest. The
difference is the container -- fvecs stores each row as a 4-byte dimension
followed by that many floats -- and the size: SPCL's arxiv corpus is 2.74M rows
of 4096 dimensions, 44.8 GB, which is read in blocks and written straight into a
memory-mapped output rather than held whole.

`--n` takes a prefix of the corpus. A prefix rather than a sample because the
file is read sequentially and the rows carry no order we would be destroying.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


def fvecs_shape(path):
    d = int(np.fromfile(path, dtype=np.int32, count=1)[0])
    n = path.stat().st_size // (4 + 4 * d)
    return n, d


def fvecs_to_npy(src, dst, n_want, d, pad, chunk=50_000):
    """Stream rows into a memmap, padding the dimension up to a multiple of 8."""
    out = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32,
                                    shape=(n_want, d + pad))
    row = 1 + d                                   # int32 header + d floats
    done = 0
    with open(src, "rb") as f:
        while done < n_want:
            m = min(chunk, n_want - done)
            buf = np.fromfile(f, dtype=np.float32, count=m * row)
            if buf.size < m * row:
                m = buf.size // row
                if m == 0:
                    break
                buf = buf[: m * row]
            out[done:done + m, :d] = buf.reshape(m, row)[:, 1:]
            done += m
    out.flush()
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--queries", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--metric", default="ip", choices=["ip", "l2"])
    ap.add_argument("--n", type=int, default=0, help="0 = the whole corpus")
    ap.add_argument("--nq", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    out = a.out or Path("/dev/shm/swann") / a.name
    out.mkdir(parents=True, exist_ok=True)
    n_all, d0 = fvecs_shape(a.base)
    nq_all, dq = fvecs_shape(a.queries)
    assert dq == d0, f"queries are {dq}-dimensional, base is {d0}"
    n = min(a.n, n_all) if a.n else n_all
    pad = (-d0) % 8
    print(f"{a.name}: {n:,} of {n_all:,} rows, d={d0}"
          + (f" padded to {d0 + pad}" if pad else ""))

    t = time.perf_counter()
    got = fvecs_to_npy(a.base, out / "X.npy", n, d0, pad)
    assert got == n, f"read {got} rows, wanted {n}"
    nq = min(a.nq, nq_all)
    fvecs_to_npy(a.queries, out / "Q_test.npy", nq, d0, pad)
    print(f"  converted in {time.perf_counter() - t:.0f}s")

    X = np.load(out / "X.npy", mmap_mode="r")
    Q = np.load(out / "Q_test.npy")
    if a.metric == "ip":
        # normalised so an inner product is a cosine, as everywhere else here
        Xg = torch.from_numpy(np.ascontiguousarray(X)).cuda()
        Xg /= Xg.norm(dim=1, keepdim=True).clamp(min=1e-12)
        np.save(out / "X.npy", Xg.cpu().numpy())
        Qg = torch.from_numpy(Q).cuda()
        Qg /= Qg.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Q = Qg.cpu().numpy()
        np.save(out / "Q_test.npy", Q)
    else:
        Xg = torch.from_numpy(np.ascontiguousarray(X)).cuda()
        Qg = torch.from_numpy(Q).cuda()

    # exact ground truth, in blocks so the score matrix never exists whole
    t = time.perf_counter()
    ids = np.empty((nq, a.k), np.int32)
    sim = np.empty((nq, a.k), np.float32)
    for s in range(0, nq, 256):
        e = min(s + 256, nq)
        if a.metric == "ip":
            sc = Qg[s:e] @ Xg.T
            v, i = torch.topk(sc, a.k, dim=1, largest=True)
        else:
            sc = torch.cdist(Qg[s:e], Xg)
            v, i = torch.topk(sc, a.k, dim=1, largest=False)
            v = -v
        ids[s:e], sim[s:e] = i.cpu().numpy(), v.cpu().numpy()
        del sc
    np.save(out / "gt_test_ids.npy", ids)
    np.save(out / "gt_test_sim.npy", sim)
    print(f"  ground truth in {time.perf_counter() - t:.0f}s")
    (out / "manifest.json").write_text(json.dumps(dict(
        corpus=a.name, n=int(n), d=int(d0 + pad), d_original=int(d0),
        k=a.k, seed=a.seed, source=str(a.base),
        metric="ip_on_l2_normalised" if a.metric == "ip" else "sqeuclidean"),
        indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
