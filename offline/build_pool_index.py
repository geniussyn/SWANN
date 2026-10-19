"""A small index over the entry pool, so entries can be dense without being scanned.

Entry quality is set by how close the nearest pool node is to the query, and that
is a function of pool density: at P=4096 the nearest entry sits at corpus rank 18
on a 100K corpus and rank 202 on a 1.18M one. Raising P fixes the rank and
destroys the budget, because finding the nearest of P entries by a flat scan
costs P distances -- 16ms at P=131072, against a 20ms search.

So the pool gets its own graph and its own search. Finding an entry becomes a few
hundred distances instead of P, and P is free to track N. This is the usual
hierarchy, built out of the parts already here: the same builder, the same
engine, one level up.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

from build_graph import knn_candidates, robust_prune, add_reverse   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/glove-100"))
    ap.add_argument("--pool", default="random128k")
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--cand", type=int, default=64)
    ap.add_argument("--alpha", type=float, default=1.2)
    a = ap.parse_args()

    X = np.load(a.data / "X.npy")
    pool = np.load(a.data / "pools" / f"pool_{a.pool}.npy").astype(np.int64)
    Xp = np.ascontiguousarray(X[pool])
    Xg = torch.from_numpy(Xp).cuda()
    t = time.perf_counter()
    cand, cdist = knn_candidates(Xg, min(a.cand, len(pool) - 1))
    Gp = robust_prune(Xg, cand, cdist, a.deg, alpha=a.alpha)
    Gp = add_reverse(Xg, Gp, a.deg, alpha=a.alpha)
    torch.cuda.synchronize()
    secs = time.perf_counter() - t

    out = a.data / "pools"
    np.save(out / f"poolX_{a.pool}.npy", Xp)
    np.save(out / f"poolG_{a.pool}.npy", Gp.cpu().numpy().astype(np.int32))
    (out / f"poolindex_{a.pool}.json").write_text(json.dumps(
        dict(pool=a.pool, P=int(len(pool)), deg=a.deg, cand=a.cand, alpha=a.alpha,
             seconds=round(secs, 2)), indent=2))
    print(f"pool index over {len(pool):,} entries, degree {a.deg}, built in {secs:.2f}s")
    print(f"wrote {out}/poolG_{a.pool}.npy")


if __name__ == "__main__":
    main()
