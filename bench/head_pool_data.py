"""Labels for predicting ENTRIES directly: which of the nearest M reach an answer.

bench/head_train.py measured the two losses in the vector-generating route
(test, V=8, 2000 queries):

    nearest-V              recall 0.9658   in_mass 0.655
    head (predicted)       recall 0.9620   in_mass 0.706
    TRUE piece centroids   recall 0.9725   in_mass 0.795
    TRUE attractors (8.2)  recall 0.9830   in_mass 0.986

So aiming at a piece centroid and taking the nearest pool entry throws away 58%
of the prize even when the centroids are exact, and the head's own prediction
error (cosine ~0.876 to its target) eats the remaining 42%. Basin boundaries are
finer than that: at att/entry ~0.5 they sit below the spacing of the entries
themselves (12).

The oracle's target was never a vector -- it is a **set of pool entries**. So
predict that instead, and the embedding-space detour disappears. Scoring the pool
is also the shape of the catalogue scan the search already runs, so it is one
matmul of a width the system already pays for.

This caches the supervision: for each of the 59,242 training queries, its nearest
M pool entries and, per entry, whether its basin holds one of the query's
answers -- the in_mass=1 set, which is exactly what the oracle maximises.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--cache", type=Path, default=Path("/dev/shm/gpuann/head"))
    ap.add_argument("--trainq", type=Path,
                    default=Path("/workspace/SYN/emb/"
                                 "msmarco-1m-trainpool__vanilla_queries.npy"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--chunk", type=int, default=4096)
    a = ap.parse_args()

    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    pl = torch.from_numpy(load_pool(a.data, a.pool, a.P).astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pl)
    gt = torch.from_numpy(np.load(a.cache / "gt_train.npy")).cuda()
    att_gt = torch.from_numpy(np.load(a.cache / "att_train.npy")).cuda()
    Q = torch.from_numpy(np.load(a.trainq).astype(np.float32)).cuda()
    Q = torch.nn.functional.normalize(Q, dim=1)[: gt.shape[0]]
    NQ = Q.shape[0]
    print(f"{NQ:,} training queries, pool {a.pool} P={a.P:,}, M={a.M}, k={a.k}")

    nearM = select_a.select_nearest(Q, Vp, a.M, engine.METRIC_IP)
    t0 = time.perf_counter()
    att_en = torch.empty(NQ, a.M, dtype=torch.int32, device=X.device)
    step = max(1, a.chunk // a.M)
    for s in range(0, NQ, step):
        e = min(s + step, NQ)
        att_en[s:e] = climb(X, G, pl[nearM[s:e]], Q[s:e]).to(torch.int32)
    torch.cuda.synchronize()
    print(f"basins of {NQ * a.M:,} entries in "
          f"{time.perf_counter()-t0:.0f}s")

    ae = att_en.cpu().numpy()
    ag = att_gt.cpu().numpy()
    lab = np.zeros((NQ, a.M), bool)
    for q in range(NQ):
        lab[q] = np.isin(ae[q], ag[q])
    print(f"\npositives (entry's basin holds an answer): {lab.mean():.1%} of the "
          f"nearest {a.M}")
    print(f"  among the nearest 8:  {lab[:, :8].mean():.1%}")
    print(f"  queries with no positive at all: {(~lab.any(1)).mean():.2%}")
    print("  (val/test nearest-8 in_mass is 0.649/0.655 -- docs 12, so the "
          "training labels line up with the evaluation)")

    np.save(a.cache / f"nearM{a.M}_train.npy", nearM.cpu().numpy().astype(np.int32))
    np.save(a.cache / f"att_en{a.M}_train.npy", ae)
    np.save(a.cache / f"lab{a.M}_train.npy", lab)
    (a.cache / f"poollab{a.M}_manifest.json").write_text(json.dumps(dict(
        nq=NQ, M=a.M, k=a.k, pool=a.pool, P=a.P, graph=a.graph,
        positive_rate=float(lab.mean()),
        positive_rate_top8=float(lab[:, :8].mean())), indent=2))
    print(f"\nwrote {a.cache}/nearM{a.M}_train.npy, att_en{a.M}_train.npy, "
          f"lab{a.M}_train.npy")


if __name__ == "__main__":
    main()
