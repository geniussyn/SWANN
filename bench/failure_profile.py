"""What is different about the one query in ten that does not return everything.

The loss is not a tail of broken queries: no query returns nothing, 89.2% return
all ten, and 83% of the missing answers come from queries that returned nine or
eight. So the question is not "which queries break" but "what does a query that
drops its last answer look like", and whether that answer was worth having.

Everything here compares the queries that lost at least one answer against the
ones that lost none, on the same run.
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
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Gnp = np.load(a.data / a.graph)
    G = torch.from_numpy(Gnp).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    gt = np.load(a.data / "gt_test_ids.npy")[:, : a.k]
    sim = np.load(a.data / "gt_test_sim.npy")[:, : a.k]
    frag = np.load(a.data / f"frag1_test_{Path(a.graph).stem}.npy")
    NQ = Q.shape[0]
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    near = select_a.select_nearest(Q, Vp, a.V, metric=metric)
    ids, _, st = engine.search(X, G, Q, pt[near].to(torch.int32), k=a.k, L=a.L,
                               B=a.B, metric=metric, team_size=a.team,
                               stop_eps=a.stop_eps, shared_cache=True,
                               shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(gt[q, i]) in {int(v) for v in got[q] if v >= 0}
                     for i in range(a.k)] for q in range(NQ)])
    bad = ~hit.all(1)
    exp = st.expansions.sum(1).cpu().numpy()
    deval = st.dist_evals.sum(1).cpu().numpy()

    recs = [json.loads(l) for l in open(
        "/workspace/SYN/corpora/msmarco-1m_queries.jsonl")]
    idx = np.load(f"{S}/test2syn.npy")
    qtext = [recs[i]["text"] for i in idx]
    rel = [set(int(d) for d, v in (recs[i].get("rel") or {}).items() if int(v) > 0)
           for i in idx]
    qlen = np.array([len(t.split()) for t in qtext])
    pieces = np.array([len(set(r)) for r in frag])
    flat = Gnp.reshape(-1)
    indeg = np.bincount(flat[flat >= 0], minlength=Gnp.shape[0])
    spread = sim[:, 0] - sim[:, -1] if ip else sim[:, -1] - sim[:, 0]

    print(f"{a.data.name}  recall {hit.mean():.4f}; "
          f"{int(bad.sum()):,} of {NQ:,} queries lost at least one answer "
          f"({100*bad.mean():.1f}%)\n")
    print(f"{'':34} {'lost one or more':>17} {'lost none':>11}")
    def row(lab, v, fmt="{:.3f}"):
        print(f"{lab:34} {fmt.format(float(np.mean(v[bad]))):>17} "
              f"{fmt.format(float(np.mean(v[~bad]))):>11}")
    row("pieces the answers sit in", pieces)
    row("query length, in words", qlen)
    row("top-1 minus top-10 similarity", spread, "{:.4f}")
    row("similarity to the 1st answer", sim[:, 0], "{:.4f}")
    row("expansions spent", exp, "{:.0f}")
    row("distance evaluations", deval, "{:.0f}")
    row("judged passage present in the top-k",
        np.array([float(len(rel[q] & set(map(int, gt[q]))) > 0) for q in range(NQ)]))

    print(f"\nwhich of the ten was lost")
    print(f"  {'slot':>5} {'lost':>7} {'share of loss':>14} {'similarity':>11} "
          f"{'in-degree':>10} {'judged':>8}")
    tot = int((~hit).sum())
    for i in range(a.k):
        m = ~hit[:, i]
        n = int(m.sum())
        d = gt[m, i]
        j = sum(1 for q in np.nonzero(m)[0] if int(gt[q, i]) in rel[q])
        print(f"  {i:5} {n:7,} {100*n/tot:13.1f}% {sim[m, i].mean():11.4f} "
              f"{np.median(indeg[d]):10.0f} {j:8}")
    print(f"  {'(corpus median in-degree ' + str(int(np.median(indeg))) + ')':>60}")

    print(f"\nhow many answers a failing query loses")
    nlost = (~hit).sum(1)
    for v in range(1, a.k + 1):
        n = int((nlost == v).sum())
        if n:
            print(f"  {v} of {a.k}: {n:5,} queries")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, recall=float(hit.mean()), failing=int(bad.sum()),
            pieces_bad=float(pieces[bad].mean()), pieces_ok=float(pieces[~bad].mean()),
            qlen_bad=float(qlen[bad].mean()), qlen_ok=float(qlen[~bad].mean()),
            per_slot=[int((~hit[:, i]).sum()) for i in range(a.k)]), indent=2))


if __name__ == "__main__":
    main()
