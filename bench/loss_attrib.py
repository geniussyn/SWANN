"""Where the recall that is missing goes, and how it is spread over queries.

Two separate questions that get confused. The distribution says whether the loss
is a thin tail of broken queries or a little taken from everywhere -- an average
of 0.985 is one bad query in seventy returning nothing, or every query dropping
its tenth answer, and the two call for different work. The attribution says which
knob is holding each missing answer: relaxing one at a time and counting what
comes back assigns the loss to the stopping rule, the queue, the budget, the
entries, or to nothing we can reach.

The knobs overlap, so the shares do not sum to one; each is "what this alone
recovers", which is the number that matters when choosing what to change.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402

S = "/tmp/claude-0/-root/5bf4d04d-280c-4ff8-a5aa-31202385739e/scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--judged", action="store_true",
                    help="also attribute the loss of the human-labelled passages")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    gt = np.load(a.data / "gt_test_ids.npy")[:, : a.k]
    NQ = Q.shape[0]
    gts = [set(int(t) for t in r) for r in gt]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    order = select_a.select_nearest(Q, Vp, max(a.P, 4 * a.V), metric=metric)

    def run(V, L, B, eps, excl=True):
        ids, _, _ = engine.search(X, G, Q, pt[order[:, :V]].to(torch.int32), k=a.k,
                                  L=L, B=B, metric=metric, team_size=a.team,
                                  stop_eps=eps, shared_cache=True,
                                  shared_bound=True, exclusive=excl,
                                  elide_visited=excl)
        got = ids.cpu().numpy()
        return np.array([[int(gt[q, i]) in {int(v) for v in got[q] if v >= 0}
                          for i in range(a.k)] for q in range(NQ)])

    base = run(a.V, a.L, a.B, a.stop_eps)
    pq = base.mean(1)
    nmiss = int((~base).sum())
    print(f"{a.data.name} {a.graph}  V={a.V} L={a.L} B={a.B} eps={a.stop_eps} "
          f"P={a.P:,}\nrecall@{a.k} {pq.mean():.4f}; "
          f"{nmiss:,} of {NQ * a.k:,} answers missing\n")

    print(f"per-query recall@{a.k}, from 1 down to 0")
    print(f"  {'recall':>7} {'queries':>8} {'share':>7} {'of all missing':>15}")
    for v in range(a.k, -1, -1):
        r = v / a.k
        m = np.isclose(pq, r)
        n = int(m.sum())
        lost = int((~base[m]).sum())
        print(f"  {r:7.1f} {n:8,} {100*n/NQ:6.1f}% {100*lost/max(nmiss,1):14.1f}%"
              + ("  " + "#" * int(50 * n / NQ) if n else ""))
    print(f"  queries at 1.0: {100*np.isclose(pq,1.0).mean():.1f}%   "
          f"below 0.5: {100*(pq < 0.5).mean():.2f}%")

    print(f"\nwhat each knob alone recovers, of the {nmiss:,} missing answers")
    trials = [("the stopping rule: eps 0.05 -> 0.30", a.V, a.L, a.B, 0.30, True),
              ("the stopping rule, off entirely",     a.V, a.L, a.B, -1.0, True),
              ("the queue: L 12 -> 64",               a.V, 64, a.B, a.stop_eps, True),
              ("the budget: B 64 -> 512",             a.V, a.L, 512, a.stop_eps, True),
              ("the entries: V 32 -> 128",            128, a.L, a.B, a.stop_eps, True),
              ("all of them at once",                 128, 64, 512, -1.0, True)]
    rows = []
    for lab, V, L, B, eps, excl in trials:
        h = run(V, L, B, eps, excl)
        rec = int((h & ~base).sum())
        print(f"  {lab:40} {rec:8,} {100*rec/max(nmiss,1):6.1f}%")
        rows.append(dict(knob=lab, recovered=rec, share=rec / max(nmiss, 1)))
    never = nmiss - rows[-1]["recovered"]
    print(f"  {'recovered by nothing above':40} {never:8,} "
          f"{100*never/max(nmiss,1):6.1f}%")

    if a.judged:
        recs = [json.loads(l) for l in open(
            "/workspace/SYN/corpora/msmarco-1m_queries.jsonl")]
        idx = np.load(f"{S}/test2syn.npy")
        rel = [set(int(d) for d, v in (recs[i].get("rel") or {}).items() if int(v) > 0)
               for i in idx]
        inside = np.array([[int(gt[q, i]) in rel[q] for i in range(a.k)]
                           for q in range(NQ)])
        jm = int((inside & ~base).sum())
        print(f"\nof those missing answers, {jm} are passages a person judged "
              f"relevant ({100*jm/max(nmiss,1):.1f}% of the loss)")
        for r, lab in zip(rows, [t[0] for t in trials]):
            pass
        for lab, V, L, B, eps, excl in trials:
            h = run(V, L, B, eps, excl)
            rec = int((h & ~base & inside).sum())
            print(f"  {lab:40} {rec:8,} {100*rec/max(jm,1):6.1f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, recall=float(pq.mean()),
                                         missing=nmiss, rows=rows,
                                         hist=[int(np.isclose(pq, v / a.k).sum())
                                               for v in range(a.k, -1, -1)]), indent=2))


if __name__ == "__main__":
    main()
