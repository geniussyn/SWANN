"""CAGRA's multi_cta against our search, with the machine held fixed.

multi_cta is the one published mode that gives a single query several thread
blocks, so it is the comparison that isolates what coordination buys rather
than what parallelism buys. Three things are held equal and nothing else is:

  the graph        both search the graph this file is pointed at
  the blocks       multi_cta runs one CTA per search path, so search_width = V
  the expansions   min_iterations = max_iterations = T fixes its iteration
                   count, and a CTA expands one node per iteration, so the
                   query costs V*T expansions -- which is what our per-branch
                   budget B = T spends at V branches

What is left over is how the blocks treat each other: multi_cta keeps a visited
hashmap per query, so a node already inserted is not re-inserted, while nothing
stops two CTAs descending the same way. Our claim map stops the second branch
from taking the node at all.

cuVS exposes no counters, so the work is not read out of it -- it is fixed from
outside and the recall at that fixed work is what gets compared.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
from pylibraft.common import device_ndarray
from cuvs.neighbors import cagra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--metric", default="inner_product")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--widths", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--iters", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    ap.add_argument("--algo", default="multi_cta",
                    choices=["multi_cta", "single_cta"],
                    help="single_cta expands search_width nodes per iteration "
                         "from one block, so its work is also iters x width x "
                         "degree and can be fixed from outside the same way")
    ap.add_argument("--itopk", type=int, nargs="+", default=None,
                    help="itopk values; default is itopk-per-cta * width")
    ap.add_argument("--itopk-per-cta", type=int, default=32,
                    help="multi_cta derives its CTA count from itopk/32 as well as "
                         "search_width; itopk = this * width keeps the two agreeing")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    if a.half:
        X = X.astype(np.float16)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    gts = [set(int(t) for t in r) for r in gt]
    G = np.load(a.data / a.graph)
    NQ = Q.shape[0]

    Xd = device_ndarray(np.ascontiguousarray(X))
    Qd = device_ndarray(np.ascontiguousarray(Q.astype(X.dtype)))
    Gd = device_ndarray(np.ascontiguousarray(G.astype(np.uint32)))
    idx = cagra.from_graph(Gd, Xd, metric=a.metric)

    print(f"{man['corpus']} n={man['n']:,} d={X.shape[1]} graph={a.graph} "
          f"deg={G.shape[1]} {'fp16' if a.half else 'fp32'} nq={NQ}\n")
    print(f"{'blocks':>7}{'iters':>7}{'expansions':>12}{'recall@'+str(a.k):>12}"
          f"{'ms':>9}{'QPS':>11}{'itopk':>9}")
    rows = []
    for w in a.widths:
        for t in a.iters:
            for it in (a.itopk or [a.itopk_per_cta * w]):
              sp = cagra.SearchParams(itopk_size=it, search_width=w,
                                      algo=a.algo, min_iterations=t,
                                      max_iterations=t)
              try:
                best = None
                for r in range(a.repeats + 1):
                    t0 = time.perf_counter()
                    d, n = cagra.search(sp, idx, Qd, a.k)
                    ms = (time.perf_counter() - t0) * 1e3
                    if r:
                        best = ms if best is None else min(best, ms)
                got = np.asarray(n.copy_to_host())
              except Exception as e:
                print(f"{w:7d}{t:7d}   skipped: {type(e).__name__}: {e}")
                continue
              rec = float(np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                                   for q in range(NQ)]))
              print(f"{w:7d}{t:7d}{w*t:12d}{rec:12.4f}{best:9.2f}{NQ/(best/1e3):11,.0f}"
                    f"{it:9d}")
              rows.append(dict(system=f"CAGRA {a.algo}", blocks=w, iters=t,
                               expansions=w * t, itopk=it,
                               recall=rec, batch_ms=best, qps=NQ / (best / 1e3),
                               half=bool(a.half), graph=a.graph))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k,
                                         nq=int(NQ), graph=a.graph,
                                         half=bool(a.half), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
