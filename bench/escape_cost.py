"""What it costs to reach a second answer fragment from a single starting point.

Every graph index in the comparison starts one query from one node: HNSW
descends its upper layers to a single `entry_point`, NSG and Vamana start from a
single medoid, and CAGRA samples uniformly at random. A descent from one node
converges on one attractor, so an answer fragment that is not in that attractor's
basin can only be reached by walking out of it -- and the only knob any of them
exposes for that is budget (`efSearch`, `itopk_size`, iterations).

This measures the price of that walk. One arm is a single branch with its queue
and its stopping rule opened step by step, which is exactly what tuning
`efSearch` does; the other is SWANN's coordinated branches. Both are read on the
same axis: distance evaluations against the fraction of the query's answer
fragments the search touched at all.

The question is not whether budget works. It does, slowly. The question is what
fraction of the evaluations it buys are spent on territory that holds no answer,
and that is the gap between the two curves.

    python bench/escape_cost.py --data /dev/shm/gpuann/msmarco-1m
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402


def reached(got, frag, k):
    """(fragments present, fragments reached) per query, over all fragments."""
    nq = frag.shape[0]
    out = np.zeros((nq, 2))
    for q in range(nq):
        ret = {int(v) for v in got[q] if v >= 0}
        pieces = {}
        for slot in range(k):
            pieces.setdefault(int(frag[q, slot]), []).append(slot)
        hit = sum(1 for p, slots in pieces.items()
                  if any(int(gtq[s]) in ret for s in slots))
        out[q] = (len(pieces), hit)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--pool", default="pool_auto.npy")
    # One branch, queue and rule opened step by step: what efSearch tuning is.
    # L up to MAXL = 512. Capping the single arm at L = 128 produces a false
    # wall: it stops at 0.954 of fragments and no budget moves it, which reads
    # as a property of the single trajectory and is a property of the queue.
    # At L = 512 the same arm reaches 0.988.
    ap.add_argument("--single", nargs="+",
                    default=["1:16:16:0.02", "1:32:32:0.05", "1:64:64:0.10",
                             "1:128:128:0.20", "1:128:512:0.50", "1:128:2048:2.0",
                             "1:128:8192:-1"])
    ap.add_argument("--swann", nargs="+",
                    default=["8:16:16:0.02", "16:16:16:0.04", "32:16:16:0.07",
                             "32:20:20:0.16", "64:24:24:0.24", "128:32:32:0.40"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy")
    G = np.ascontiguousarray(np.load(a.data / a.graph))
    Q = np.ascontiguousarray(np.load(a.data / f"Q_{a.split}.npy")[: a.nq])
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")[: a.nq]
    phi = np.load(a.data / f"phi1_{a.split}_{tag}.npy")[: a.nq]
    global gtq
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    dg = engine.graph_degrees(Gg)
    pool = torch.from_numpy(np.load(a.data / "pools" / a.pool).astype(np.int32)).cuda()
    catX = select_a.pool_vectors(Xg, pool)
    print(f"{man['corpus']} n={len(X):,} d={X.shape[1]} {a.split} q={len(Q)} "
          f"graph={a.graph} P={len(pool):,}")
    print(f"mean phi1={phi.mean():.3f}  one fragment {100*(phi==1).mean():.1f}% "
          f"of queries\n")

    seeds_for = {}
    rows = []
    print(f"{'arm':8s} {'V:L:B:eps':>18s} {'ms':>7s} {'dist/q':>9s} {'recall':>8s} "
          f"{'frag reached':>13s}")
    for armname, cfgs in (("single", a.single), ("SWANN", a.swann)):
        for cfg in cfgs:
            V, L, B = (int(x) for x in cfg.split(":")[:3])
            eps = float(cfg.split(":")[3])
            # Selection is timed with the search: this table is read against
            # CAGRA, which starts from a random sample and pays nothing for it.
            if V not in seeds_for:
                sb = None
                for r in range(3):
                    torch.cuda.synchronize(); t = time.perf_counter()
                    nr = select_a.select_nearest(Qg, catX, V, metric=metric,
                                                 precision="tf32")
                    torch.cuda.synchronize(); sms = (time.perf_counter() - t) * 1e3
                    if r and (sb is None or sms < sb):
                        sb = sms
                seeds_for[V] = (pool[nr].unsqueeze(-1).contiguous(), sb)
            seeds, sel_ms = seeds_for[V]
            coord = armname == "SWANN"
            best = None
            for r in range(3):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, _, st = engine.search(
                    Xg, Gg, Qg, seeds, k=a.k, L=L, B=B, metric=metric,
                    policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                    shared_cache=coord, exclusive=coord, shared_bound=coord,
                    team_size=a.team)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (best is None or ms < best[0]):
                    best = (ms, ids, st)
            ms, ids, st = best
            search_ms, ms = ms, ms + sel_ms
            got = ids.cpu().numpy()
            gtq = gt                                   # read by reached()
            rc = float(np.mean([len({int(v) for v in got[q]} & {int(t) for t in gt[q]})
                                / a.k for q in range(len(gt))]))
            pr = np.zeros((len(gt), 2))
            for q in range(len(gt)):
                ret = {int(v) for v in got[q] if v >= 0}
                pieces = {}
                for slot in range(a.k):
                    pieces.setdefault(int(frag[q, slot]), []).append(slot)
                pr[q] = (len(pieces),
                         sum(1 for p, sl in pieces.items()
                             if any(int(gt[q, s]) in ret for s in sl)))
            fr = float(pr[:, 1].sum() / pr[:, 0].sum())
            nd = float(st.dist_evals.sum()) / len(Q)
            rows.append(dict(arm=armname, V=V, L=L, B=B, eps=eps, ms=ms,
                             search_ms=search_ms, select_ms=sel_ms,
                             dist_per_q=nd, recall=rc, frag_reached=fr))
            print(f"{armname:8s} {cfg:>18s} {ms:7.2f} {nd:9.0f} {rc:8.4f} "
                  f"{fr:13.4f}")
        print()

    # The price of a fragment, read off both curves at matched coverage.
    print("marginal distance evaluations per additional 1% of fragments reached")
    for armname in ("single", "SWANN"):
        rs = sorted((r for r in rows if r["arm"] == armname),
                    key=lambda r: r["frag_reached"])
        for p, q in zip(rs, rs[1:]):
            dfr = q["frag_reached"] - p["frag_reached"]
            if dfr <= 0:
                continue
            print(f"  {armname:8s} {p['frag_reached']:.4f} -> {q['frag_reached']:.4f}  "
                  f"{(q['dist_per_q'] - p['dist_per_q']) / (dfr * 100):9.0f} dist/q per 1%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, nq=len(Q),
                                         graph=a.graph, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
