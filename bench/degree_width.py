"""What a padded adjacency matrix costs, and which half of it the kernel can get.

A pruned or degree-allocated build does not fill its rows: the matrix is as wide
as its *widest* node and every other node leaves a tail of -1. Measured here on
GIST that is 37.5 edges in a 96-wide row. Two different costs hide in that:

  bounds   the kernel walking slots that hold no edge -- the graph read, the
           cache probes, the expansion rounds, the candidate sort. `deg` fixes
           this and is free: the slots it skips are the ones already rejected,
           so the answer is bitwise identical (tests/test_engine.py).
  stride   the row being 384 bytes instead of 240 whatever the kernel reads from
           it -- HBM traffic and L2 footprint. Only true compaction (CSR) reaches
           this, and this bench is what says whether that is worth writing.

It separates them by storing one real graph at several strides and running each
stride both ways. With a dense degree-64 graph on GloVe:

    stride 64            3.16 ms    1.000x
    stride 96  bounds    3.26       1.030x
    stride 96  degree    3.18       1.007x
    stride 128 bounds    3.63       1.149x
    stride 128 degree    3.43       1.085x

so at 1.5x over-width `deg` recovers 2.3 of the 3.0 points and leaves 0.7, and at
2x it recovers 6.4 of 14.9. The real graphs are at 1.5x, which is why the gain on
them is about 1%: the padding tax was mostly never the bounds.

    python bench/degree_width.py --data /dev/shm/gpuann/glove-100 \
        --graph G_cagra_d64.npy --strides 64 96 128
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/glove-100"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--strides", type=int, nargs="+", default=[0],
                    help="widths to store the graph at; 0 means its own width. A "
                         "stride narrower than the graph is skipped.")
    ap.add_argument("--L", type=int, default=64)
    ap.add_argument("--B", type=int, default=128)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Q = np.ascontiguousarray(np.load(a.data / f"Q_{a.split}.npy")[: a.nq])
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pool_t = torch.from_numpy(pool).cuda()
    catX = select_a.pool_vectors(Xg, pool_t)
    near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
    seeds = pool_t[near].unsqueeze(1).contiguous()

    G0 = np.load(a.data / a.graph)
    real = (G0 >= 0).sum(1)
    strides = [w if w else G0.shape[1] for w in a.strides]
    print(f"{man['corpus']} n={man['n']} d={man['d']} graph={a.graph}  "
          f"degree mean {real.mean():.1f} max {real.max()}  "
          f"{a.split} q={len(Q)} V={a.V} L={a.L} B={a.B} eps={a.eps}\n")
    print(f"{'stride':>7s} {'fill%':>6s} {'bounds':>7s} {'recall@10':>10s} "
          f"{'ms':>7s} {'vs narrowest':>13s}")
    rows, base = [], None
    for W in strides:
        if W < G0.shape[1]:
            print(f"{W:7d}  (narrower than the graph's own {G0.shape[1]}, skipped)")
            continue
        Gw = np.full((G0.shape[0], W), -1, dtype=np.int32)
        Gw[:, : G0.shape[1]] = G0
        Gg = torch.from_numpy(Gw).cuda()
        dg = engine.graph_degrees(Gg)
        for tag, d in (("stride", None), ("degree", dg)):
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(
                    Xg, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_ROUND_ROBIN, stop_eps=a.eps,
                    shared_cache=True, shared_bound=True, deg=d)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (best is None or ms < best[0]):
                    best = (ms, ids)
            ms, ids = best
            rc = recall_at_k(ids.cpu().numpy(), gt)
            if base is None:
                base = ms
            rows.append(dict(stride=W, bounds=tag, recall=rc, ms=ms, rel=ms / base))
            print(f"{W:7d} {100*real.mean()/W:6.1f} {tag:>7s} {rc:10.4f} "
                  f"{ms:7.2f} {ms/base:12.3f}x")
        del Gg, Gw
        torch.cuda.empty_cache()
    # the shared bound makes which branch retires CTA-order dependent, so recall
    # is reported rather than asserted bitwise; the width must not move it.
    span = max(r["recall"] for r in rows) - min(r["recall"] for r in rows)
    print(f"\nrecall spread across every row: {span:.4f} "
          f"(the width is a cost, not an answer)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, graph=a.graph, split=a.split,
                                         nq=len(Q), V=a.V, L=a.L, B=a.B, eps=a.eps,
                                         deg_mean=float(real.mean()),
                                         deg_max=int(real.max()), rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
