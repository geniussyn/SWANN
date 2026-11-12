"""How much of the expansion's memory traffic is spent on candidates it discards.

Expansion is 91% of the wall clock and already runs near the memory roof, so it
gets faster by reading fewer bytes, not by reading faster. Each expansion loads
all D neighbour vectors of a node to compute their distances, but only the ones
that beat the queue's worst entry are kept -- the instrumented fan-out is 10 to
16 out of D, which would mean most of the bytes are read to decide against the
candidate. This measures that share directly, alongside the bandwidth actually
achieved, so the headroom is a number rather than an impression.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest16k")
    ap.add_argument("--P", type=int, default=16384)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    Xf = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    gt = np.load(a.data / "gt_test_ids.npy")[:, : a.k]
    NQ, dim = Q.shape
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(Xf, pt)
    seeds = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)].to(torch.int32)
    D = G.shape[1]

    print(f"{a.data.name} {a.graph}  V={a.V} L={a.L} B={a.B} eps={a.stop_eps}  "
          f"d={dim}, D={D}, {NQ:,} queries\n")
    print(f"{'vectors':>8} {'recall':>8} {'expand ms':>10} {'dist/q':>8} "
          f"{'fan-out':>8} {'GB read':>8} {'GB/s':>8}")
    rows = []
    for lab, X in (("fp32", Xf), ("fp16", Xf.half().contiguous())):
        best = None
        for r in range(3):
            st = engine.init_states(X, Q, seeds, L=a.L, B=a.B, D=D, metric=metric,
                                    team_size=a.team, stop_eps=a.stop_eps,
                                    kstop=a.k, shared_cache=True,
                                    shared_bound=True, exclusive=True,
                                    elide_visited=True)
            torch.cuda.synchronize(); t = time.perf_counter()
            engine.advance(X, G, Q, st, a.B)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        bi, bd = engine.branch_topk(st, a.k)
        ids, _ = engine.merge_branches(bi, bd, a.k)
        got = ids.cpu().numpy()
        rec = np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        de = float(st.dist_evals.sum(1).float().mean())
        fan = float(st.counters[..., 8].sum().item() /
                    max(int(st.counters[..., 9].sum().item()), 1))
        nbytes = de * NQ * dim * X.element_size()
        print(f"{lab:>8} {rec:8.4f} {best:10.2f} {de:8.0f} {fan:8.2f} "
              f"{nbytes/1e9:8.2f} {nbytes/1e9/(best/1e3):8.0f}")
        rows.append(dict(vectors=lab, recall=float(rec), expand_ms=best,
                         dist_per_q=de, fanout=fan, gb=nbytes / 1e9,
                         gbs=nbytes / 1e9 / (best / 1e3)))
        if lab == "fp16":
            del X
    f = rows[0]["fanout"]
    print(f"\nof the {D} neighbour vectors an expansion reads, {f:.1f} clear the\n"
          f"queue's worst entry and the other {D - f:.1f} are read only to be\n"
          f"rejected: {100*(D - f)/D:.0f}% of the traffic. Halving the bytes with\n"
          f"fp16 costs {rows[0]['recall'] - rows[1]['recall']:+.4f} recall and is "
          f"worth {rows[0]['expand_ms']/rows[1]['expand_ms']:.2f}x here.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, D=D, dim=dim,
                                         rows=rows), indent=2))


if __name__ == "__main__":
    main()
