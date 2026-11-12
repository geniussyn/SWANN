"""The fragment decomposition, split by how large each fragment is.

A singleton answer is recovered in full the moment it is touched, so counting it
in "recall inside the fragments reached" guarantees a number near one whatever
the search does. The claim that mining is complete is only informative about
fragments with something left to mine, so this reports every quantity again over
fragments of size >= 2, and over size >= 3, beside the figure over all of them.

The same split is what distinguishes the two readings of a rising phi: an answer
cut into more pieces, or an answer cut into more singletons.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from bench.entry_arms import Arms                                   # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402


def piece_table(got, gt, frag, min_size):
    """Per query, over fragments of at least `min_size` slots:
    (fragments present, fragments reached, slots recovered in reached, slots in reached)."""
    nq, k = gt.shape
    out = np.zeros((nq, 4))
    for q in range(nq):
        ret = {int(v) for v in got[q] if v >= 0}
        labels = frag[q]
        nf = int(labels.max()) + 1
        present = reached = inside_hit = inside_tot = 0
        for f in range(nf):
            slots = np.flatnonzero(labels == f)
            if len(slots) < min_size:
                continue
            present += 1
            found = sum(1 for s in slots if int(gt[q, s]) in ret)
            if found:
                reached += 1
                inside_hit += found
                inside_tot += len(slots)
        out[q] = (present, reached, inside_hit, inside_tot)
    return out


def size_histogram(frag, k):
    sizes = []
    for q in range(frag.shape[0]):
        labels = frag[q]
        for f in range(int(labels.max()) + 1):
            sizes.append(int((labels == f).sum()))
    s = np.array(sizes)
    return dict(pieces=int(s.size), singletons=float((s == 1).mean()),
                mean_size=float(s.mean()), median_size=float(np.median(s)),
                share_of_slots_in_singletons=float(s[s == 1].sum() / s.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tag", default=None, help="phi/frag suffix; defaults to graph stem")
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--configs", nargs="+",
                    default=["distance:1:128:1024", "distance:32:12:64", "distance:128:12:64"])
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    tag = a.tag or Path(a.graph).stem
    man = json.loads((a.data / "manifest.json").read_text())
    # L2 corpora searched by inner product return nothing, quietly: the metric
    # has to come from the manifest, not from a default.
    metric_id = (engine.METRIC_IP if man["metric"].startswith("ip")
                 else engine.METRIC_L2)
    X = np.load(a.data / "X.npy")
    G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    phi = np.load(a.data / f"phi{a.h}_{a.split}_{tag}.npy")
    frag = np.load(a.data / f"frag{a.h}_{a.split}_{tag}.npy")
    if frag.shape[1] != a.k:
        raise SystemExit(f"frag has {frag.shape[1]} slots, --k is {a.k}: "
                         "run offline/phi_h.py against the matching ground truth")
    if a.nq:
        Q, gt, phi, frag = Q[:a.nq], gt[:a.nq], phi[:a.nq], frag[:a.nq]

    hist = size_histogram(frag, a.k)
    print(f"{a.data.name}  h={a.h}  k={a.k}  graph={a.graph}  nq={len(phi)}")
    print(f"  mean phi={phi.mean():.3f}  one piece {100*(phi==1).mean():.1f}%  "
          f"pieces {hist['pieces']:,}  singletons {100*hist['singletons']:.1f}% of pieces, "
          f"{100*hist['share_of_slots_in_singletons']:.1f}% of answer slots\n")

    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.1, 32, 24, metric=metric_id)

    rows = []
    for cfg in a.configs:
        arm, V, L, B = cfg.split(":")[0], *[int(x) for x in cfg.split(":")[1:4]]
        seeds, _, _ = arms.seeds(arm, V)
        ids, _, st = engine.search(Xg, Gg, Qg, seeds, k=a.k, L=L, B=B,
                                   policy=engine.POLICY_BEST, team_size=a.team,
                                   stop_eps=a.stop_eps, shared_cache=True,
                                   shared_bound=True, exclusive=True,
                                   metric=metric_id)
        got = ids.cpu().numpy()
        recall = float(np.mean([len({int(v) for v in got[q]} & {int(t) for t in gt[q]}) / a.k
                                for q in range(len(gt))]))
        print(f"--- {cfg}  recall@{a.k} {recall:.4f}")
        print(f"    {'fragments':>12}{'present':>10}{'reached':>10}{'recall in reached':>20}")
        for label, ms in (("all", 1), ("size >= 2", 2), ("size >= 3", 3)):
            s = piece_table(got, gt, frag, ms)
            sel = s[:, 0] > 0
            if not sel.any():
                print(f"    {label:>12}{'--':>10}")
                continue
            present = s[sel, 0].mean()
            reached = (s[sel, 1] / s[sel, 0]).mean()
            inr = (s[sel, 2] / np.maximum(s[sel, 3], 1))[s[sel, 1] > 0].mean()
            print(f"    {label:>12}{present:>10.2f}{reached:>10.4f}{inr:>20.4f}")
            rows.append(dict(config=cfg, recall=recall, size_class=label,
                             queries=int(sel.sum()), pieces_per_query=float(present),
                             pieces_reached=float(reached), recall_in_reached=float(inr)))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, h=a.h, k=a.k,
                                         graph=a.graph, histogram=hist,
                                         mean_phi=float(phi.mean()), rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
