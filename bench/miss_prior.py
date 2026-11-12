"""Are the same places missed over and over, by different queries?

Nothing computable from a query says where its hard answers are, and nothing the
search itself produces says it either -- a first pass finds what a second pass
would have found anyway. That leaves the corpus. If the answers that get lost are
spread uniformly, there is nothing to learn offline; if they concentrate in
particular regions, a catalogue can be told about those regions once, before any
query arrives, and no query-time signal is needed.

The test has to generalise across queries or it is just memorisation: fit a
per-entry miss density on one query split, and ask whether it ranks the answers
another split loses above the ones it finds.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def misses(X, G, Q, gt, pt, Vp, V, k, L, Bx, metric, team, eps, ip):
    seeds = pt[select_a.select_nearest(Q, Vp, V, metric=metric)]
    ids, _, _ = engine.search(X, G, Q, seeds.to(torch.int32), k=k, L=L, B=Bx,
                              metric=metric, team_size=team, stop_eps=eps,
                              shared_cache=True, shared_bound=True,
                              exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--fit", default="val")
    ap.add_argument("--test", default="test")
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
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)

    out = {}
    for split in (a.fit, a.test):
        Q = torch.from_numpy(np.load(a.data / f"Q_{split}.npy")).cuda()
        gt = np.load(a.data / f"gt_{split}_ids.npy")[:, : a.k]
        hit = misses(X, G, Q, gt, pt, Vp, a.V, a.k, a.L, a.B, metric,
                     a.team, a.stop_eps, ip)
        # which catalogue entry each answer belongs to, as its region label
        tv = X[torch.from_numpy(gt.astype(np.int64)).cuda().reshape(-1)]
        s = (tv @ Vp.T) if ip else -torch.cdist(tv, Vp)
        reg = s.argmax(1).view(gt.shape).cpu().numpy(); del s, tv
        out[split] = dict(hit=hit, gt=gt, reg=reg)
        print(f"{split}: {gt.shape[0]} queries, recall {hit.mean():.4f}, "
              f"{int((~hit).sum())} answers missed")

    f, t = out[a.fit], out[a.test]
    # is a missed answer missed again? the same document, by a different query
    fm = set(map(int, f["gt"][~f["hit"]]))
    tm = f["gt"][~f["hit"]]
    tdoc_miss = set(map(int, t["gt"][~t["hit"]]))
    tdoc_hit = set(map(int, t["gt"][t["hit"]]))
    shared = len(fm & tdoc_miss)
    print(f"\ndocuments missed on {a.fit} that also appear as answers on {a.test}: "
          f"{len(fm & (tdoc_miss | tdoc_hit))}")
    print(f"  of those, missed again: {shared}")

    # the offline prior: how many fit-split misses land in each entry's region
    dens = np.bincount(f["reg"][~f["hit"]], minlength=a.P).astype(np.float64)
    tot = np.bincount(f["reg"].reshape(-1), minlength=a.P).astype(np.float64)
    rate = dens / np.maximum(tot, 1)
    print(f"\nmiss density over the {a.P} entry regions (fitted on {a.fit}):")
    print(f"  regions holding any answer: {int((tot>0).sum()):,}   "
          f"holding a missed one: {int((dens>0).sum()):,}")
    nz = rate[tot > 0]
    print(f"  per-region miss rate: p50 {np.percentile(nz,50):.4f}  "
          f"p90 {np.percentile(nz,90):.4f}  p99 {np.percentile(nz,99):.4f}")

    # does it rank test-split misses above test-split hits?
    sc_miss = rate[t["reg"][~t["hit"]]]
    sc_hit = rate[t["reg"][t["hit"]]]
    n = min(len(sc_miss), 200_000)
    rng = np.random.default_rng(0)
    A = rng.choice(sc_miss, n, replace=len(sc_miss) < n)
    Bb = rng.choice(sc_hit, n, replace=False)
    auc = float((A > Bb).mean() + 0.5 * (A == Bb).mean())
    print(f"\nranking {a.test} answers by that prior:")
    print(f"  mean score, missed {sc_miss.mean():.4f}   found {sc_hit.mean():.4f}")
    print(f"  AUC separating missed from found: {auc:.3f}   "
          f"(0.5 = the prior knows nothing)")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, P=a.P, auc=auc,
                                         miss_mean=float(sc_miss.mean()),
                                         hit_mean=float(sc_hit.mean()),
                                         repeat_miss=shared), indent=2))


if __name__ == "__main__":
    main()
