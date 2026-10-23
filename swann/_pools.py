"""Loading an entry pool, and the one rule about it: it never sees a query.

A pool is offline metadata -- node ids and nothing else -- so every arm and the
oracle read it the same way, and swapping a placement changes one flag rather
than a code path.
"""
from pathlib import Path

import numpy as np


def load_pool(data, name="random", P=512, seed=0):
    """A named placement from `offline/place_entries.py`, or an inline random
    pool when none has been written yet (identical to placement 'random')."""
    p = Path(data) / "pools" / f"pool_{name}.npy"
    if p.exists():
        pool = np.load(p).astype(np.int32)
        if P and len(pool) != P:
            pool = pool[:P]
        return pool
    if name != "random":
        raise FileNotFoundError(f"no pool '{name}' under {p.parent}; run offline/place_entries.py")
    n = np.load(Path(data) / "X.npy", mmap_mode="r").shape[0]
    return np.sort(np.random.default_rng(seed).choice(n, size=P, replace=False)).astype(np.int32)
