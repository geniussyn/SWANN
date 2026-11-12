"""Which queries fail, and do two systems fail the same ones?

Three questions a mean recall cannot answer. How many queries are returned
perfectly, rather than on average well. Whether the stopping rule spends its
budget on the queries that need it -- if a hard query is retired early, the rule
is cutting exactly where it should not. And whether the hard tail is a property
of the corpus or of the search: if SWANN and HNSW fail the same queries, the
difficulty is in the data and no traversal will remove it; if they fail
different ones, part of the tail is ours to fix.
"""
import argparse, json
from pathlib import Path

import numpy as np


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swann", type=Path, required=True, help="npz from entry_arms")
    ap.add_argument("--hnsw", type=Path, default=None, help="npz from hnsw_cpu")
    ap.add_argument("--decile", type=float, default=0.10)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    z = np.load(a.swann, allow_pickle=True)
    pq, lab = z["perq"], [str(x) for x in z["labels"]]
    exp = z["expansions"] if "expansions" in z else None

    print(f"SWANN, {pq.shape[1]:,} queries, {len(lab)} configurations\n")
    print(f"{'config':28} {'mean':>7} {'=1.0':>7} {'<0.9':>7} {'<0.5':>7} "
          f"{'p10':>6} {'확장 상관':>9}")
    rows = []
    for i, l in enumerate(lab):
        r = pq[i]
        corr = spearman(r, exp[i]) if exp is not None else float("nan")
        print(f"{l:28} {r.mean():7.4f} {(r >= 1).mean():7.1%} {(r < 0.9).mean():7.1%} "
              f"{(r < 0.5).mean():7.1%} {np.percentile(r, 10):6.2f} {corr:9.3f}")
        rows.append(dict(config=l, mean=float(r.mean()), perfect=float((r >= 1).mean()),
                         below09=float((r < 0.9).mean()), below05=float((r < 0.5).mean()),
                         p10=float(np.percentile(r, 10)),
                         spearman_recall_expansions=float(corr)))

    if a.hnsw is None:
        if a.out:
            a.out.write_text(json.dumps(dict(rows=rows), indent=2))
        return

    h = np.load(a.hnsw, allow_pickle=True)
    hq, hlab = h["perq"], [str(x) for x in h["labels"]]
    n = min(pq.shape[1], hq.shape[1])
    print(f"\n두 시스템의 어려운 쿼리 집합 (하위 {a.decile:.0%}), {n:,} 쿼리 공통\n")
    print(f"{'SWANN config':24} {'HNSW config':12} {'평균 recall':>11} {'겹침':>7} "
          f"{'Jaccard':>8} {'rank rho':>9}")
    pairs = []
    for i, l in enumerate(lab):
        si = pq[i][:n]
        # pair each SWANN point with the HNSW point closest in mean recall
        j = int(np.argmin([abs(hq[k][:n].mean() - si.mean()) for k in range(len(hlab))]))
        hj = hq[j][:n]
        m = max(1, int(round(a.decile * n)))
        sb = set(np.argsort(si, kind="stable")[:m].tolist())
        hb = set(np.argsort(hj, kind="stable")[:m].tolist())
        inter = len(sb & hb)
        jac = inter / len(sb | hb)
        rho = spearman(si, hj)
        print(f"{l:24} {hlab[j]:12} {si.mean():5.4f}/{hj.mean():5.4f} "
              f"{inter/m:7.1%} {jac:8.3f} {rho:9.3f}")
        pairs.append(dict(swann=l, hnsw=hlab[j], swann_mean=float(si.mean()),
                          hnsw_mean=float(hj.mean()), overlap=float(inter / m),
                          jaccard=float(jac), spearman=float(rho)))
    exp_rand = a.decile
    print(f"\n무작위였다면 겹침 {exp_rand:.1%}, Jaccard {exp_rand/(2-exp_rand):.3f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(rows=rows, pairs=pairs), indent=2))


if __name__ == "__main__":
    main()
