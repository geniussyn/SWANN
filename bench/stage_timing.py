"""Where a search's wall clock actually goes, stage by stage.

Entry selection is easy to forget because it is not the search, but it scales
with the catalogue and the catalogue is the one lever that worked, so its share
decides how dense a catalogue is affordable. The stages are timed with CUDA
events around each call rather than inferred from totals, and the allocation of
state is separated from the expansion loop because only the second scales with
the budget.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def timed(fn, reps=3):
    """Best of `reps` after one warm-up, in milliseconds."""
    fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        torch.cuda.synchronize(); t = time.perf_counter()
        r = fn()
        torch.cuda.synchronize(); best = min(best, (time.perf_counter() - t) * 1e3)
    return best, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pools", nargs="+",
                    default=["nest4k:4096", "nest16k:16384", "nest128k:131072"])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 32, 512, 3490])
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q0 = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    print(f"{a.data.name} {a.graph}  V={a.V} L={a.L} B={a.B} eps={a.stop_eps} "
          f"d={X.shape[1]}\n")
    rows = []
    for spec in a.pools:
        name, P = spec.split(":"); P = int(P)
        pool = load_pool(a.data, name, P)
        pt = torch.from_numpy(pool.astype(np.int64)).cuda()
        Vp = select_a.pool_vectors(X, pt)             # held for the index's life
        print(f"catalogue {P:,}")
        print(f"  {'batch':>6} {'select':>8} {'init':>8} {'expand':>9} "
              f"{'branch-k':>9} {'merge':>8} {'total':>9} {'select %':>9}")
        for nq in a.batches:
            Q = Q0[:nq].contiguous() if nq <= Q0.shape[0] else Q0.repeat(
                (nq + Q0.shape[0] - 1) // Q0.shape[0], 1)[:nq].contiguous()
            sel_ms, near = timed(lambda: select_a.select_nearest(
                Q, Vp, a.V, metric=metric))
            seeds = pt[near].to(torch.int32)
            init_ms, st = timed(lambda: engine.init_states(
                X, Q, seeds, L=a.L, B=a.B, D=G.shape[1], metric=metric,
                team_size=a.team, stop_eps=a.stop_eps, kstop=a.k,
                shared_cache=True, shared_bound=True, exclusive=True,
                elide_visited=True))

            def adv():
                s = engine.init_states(X, Q, seeds, L=a.L, B=a.B, D=G.shape[1],
                                       metric=metric, team_size=a.team,
                                       stop_eps=a.stop_eps, kstop=a.k,
                                       shared_cache=True, shared_bound=True,
                                       exclusive=True, elide_visited=True)
                engine.advance(X, G, Q, s, a.B)
                return s
            both_ms, st2 = timed(adv)
            exp_ms = max(both_ms - init_ms, 0.0)
            bk_ms, (bi, bd) = timed(lambda: engine.branch_topk(st2, a.k))
            mg_ms, _ = timed(lambda: engine.merge_branches(bi, bd, a.k))
            tot = sel_ms + init_ms + exp_ms + bk_ms + mg_ms
            print(f"  {nq:6,} {sel_ms:8.2f} {init_ms:8.2f} {exp_ms:9.2f} "
                  f"{bk_ms:9.2f} {mg_ms:8.2f} {tot:9.2f} {100*sel_ms/tot:8.1f}%")
            rows.append(dict(P=P, nq=nq, select=sel_ms, init=init_ms, expand=exp_ms,
                             branch_topk=bk_ms, merge=mg_ms, total=tot))
            del Q, st, st2, bi, bd
            torch.cuda.empty_cache()
        print()
        del Vp, pt
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
