"""Does the recall we lose cost the end task anything?

The queries that drag the mean down turn out to have ground truth that is
semantic noise: for "define marmite" the exact top-10 contains a passage about
marble, and the catalogue entry that recovers it is about granite. These are the
embedding's surface neighbours for a rare token, not a coherent set of answers,
which is why they scatter and why nothing derived from the query's meaning can
point at them.

MS MARCO ships human relevance judgements, so the question can be settled rather
than argued. Two recalls over the same runs: against the exact top-$k$ of the
encoder, which is what an ANN index is normally graded on, and against the
passages a person marked relevant. If the answers we lose are the ones no person
called relevant, the loss is in the metric and not in the application.
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
    ap.add_argument("--queries", type=Path,
                    default=Path("/workspace/SYN/corpora/msmarco-1m_queries.jsonl"))
    ap.add_argument("--map", type=Path, default=Path(f"{S}/test2syn.npy"),
                    help="row i of the test split is this SYN query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32, 64])
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pools", nargs="+", default=["nest4k:4096", "nest128k:131072"])
    ap.add_argument("--team", type=int, default=16)
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
    recs = [json.loads(l) for l in open(a.queries)]
    idx = np.load(a.map)
    rel = [set(int(d) for d, v in (recs[i].get("rel") or {}).items() if int(v) > 0)
           for i in idx]
    nrel = np.array([len(r) for r in rel])
    gts = [set(int(t) for t in r) for r in gt]
    inside = np.array([len(rel[q] & gts[q]) for q in range(NQ)])
    print(f"{NQ} test queries; judged-relevant passages per query: "
          f"mean {nrel.mean():.2f}, with none {int((nrel==0).sum())}")
    print(f"the judged passage is inside the encoder's exact top-{a.k} for "
          f"{int((inside>0).sum()):,} of them ({100*(inside>0).mean():.1f}%)\n")
    ok = nrel > 0

    print(f"{'catalogue':12} {'V':>4} {'recall@10 vs exact kNN':>24} "
          f"{'recall of the judged passage':>29}")
    rows = []
    for spec in a.pools:
        name, P = spec.split(":"); P = int(P)
        pool = load_pool(a.data, name, P)
        pt = torch.from_numpy(pool.astype(np.int64)).cuda()
        Vp = select_a.pool_vectors(X, pt)
        nearQ = select_a.select_nearest(Q, Vp, max(a.V), metric=metric)
        for V in a.V:
            ids, _, _ = engine.search(X, G, Q, pt[nearQ[:, :V]].to(torch.int32),
                                      k=a.k, L=a.L, B=a.B, metric=metric,
                                      team_size=a.team, stop_eps=a.stop_eps,
                                      shared_cache=True, shared_bound=True,
                                      exclusive=True, elide_visited=True)
            got = ids.cpu().numpy()
            sets = [{int(v) for v in got[q] if v >= 0} for q in range(NQ)]
            r_knn = np.mean([len(gts[q] & sets[q]) / a.k for q in range(NQ)])
            # of the judged passages that the encoder's top-k contains at all,
            # how many does the search return
            den = sum(1 for q in range(NQ) if ok[q] and (rel[q] & gts[q]))
            num = sum(1 for q in range(NQ) if ok[q] and (rel[q] & gts[q] & sets[q]))
            print(f"{name:12} {V:4} {r_knn:24.4f} {num/den:29.4f}")
            rows.append(dict(pool=name, P=P, V=V, recall_knn=float(r_knn),
                             recall_judged=float(num / den), judged_den=den))
    print(f"\nthe second column counts only queries whose judged passage the\n"
          f"encoder ranked in its own top-{a.k}; a search cannot return what the\n"
          f"ground truth it is graded against does not contain.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, nq=NQ, rows=rows), indent=2))


if __name__ == "__main__":
    main()
