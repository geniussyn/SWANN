"""The difficulty stratum, by the lab's definition rather than a new one.

`run_cell.py:hard_decile` fixes it as: the queries with the lowest mean ANN
recall@10 across a search-effort sweep, measured on a *selector* index built
independently of every index that is later compared, and then held fixed for all
arms. Nothing about a query's geometry is consulted; difficulty is defined by
what a search actually recovers.

The selector here is built by CAGRA, the competitor, at a degree neither measured
graph uses (32/64 against 64/128 and our own 64). Using the other side's builder
for the stratum is the conservative choice: if it biased the decile at all, it
would bias it against us.

    PYTHONPATH=/opt/cuvsenv python offline/hard_decile.py
"""
import argparse, json
from pathlib import Path

import numpy as np
from cuvs.neighbors import cagra

try:
    from cuvs.common import Resources
except ImportError:
    from pylibraft.common import DeviceResources as Resources
from pylibraft.common import device_ndarray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--deg", type=int, default=32)
    ap.add_argument("--intermediate", type=int, default=64)
    ap.add_argument("--efs", type=int, nargs="+", default=[32, 64, 128, 256, 512])
    ap.add_argument("--frac", type=float, default=0.10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    res = Resources()
    Xd, Qd = device_ndarray(X), device_ndarray(Q)
    idx = cagra.build(cagra.IndexParams(metric="inner_product", graph_degree=a.deg,
                                        intermediate_graph_degree=a.intermediate),
                      Xd, resources=res)
    res.sync()
    print(f"selector index: CAGRA degree {a.deg}/{a.intermediate} "
          f"(neither measured graph)")

    acc = np.zeros(Q.shape[0])
    for ef in a.efs:
        out = cagra.search(cagra.SearchParams(itopk_size=ef), idx, Qd, a.k, resources=res)
        res.sync()
        nb = out[1].copy_to_host().astype(np.int64)
        per = np.array([len(set(x.tolist()) & set(g.tolist())) / a.k for x, g in zip(nb, gt)])
        acc += per
        print(f"  itopk {ef:4d}: mean recall {per.mean():.4f}")
    mean_recall = acc / len(a.efs)

    n_hard = max(1, int(round(a.frac * Q.shape[0])))
    order = np.lexsort((np.arange(Q.shape[0]), mean_recall))     # ties by index
    hard = np.sort(order[:n_hard])
    out = Path(a.out) if a.out else a.data / f"harddecile_{a.split}.npz"
    np.savez(out, hard=hard.astype(np.int64), mean_recall=mean_recall,
             deg=a.deg, intermediate=a.intermediate, efs=np.array(a.efs))
    np.save(a.data / f"hard10_{a.split}.npy", hard.astype(np.int64))
    print(f"\nhardest {100*a.frac:.0f}%: {n_hard} queries, selector mean recall "
          f"{mean_recall[hard].mean():.4f} against {mean_recall.mean():.4f} overall "
          f"(worst {mean_recall[hard].max():.4f} at the cut)")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
