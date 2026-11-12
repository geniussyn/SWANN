"""How far apart are a query's pieces? That is what decides L against V.

The fragmentation index counts a query's pieces but says nothing about how far
apart they are, and the two call for different searches. Pieces separated by a
few hundred neighbour ranks are crossed by a queue wide enough to carry a
candidate that is temporarily worse -- the regime a single wide queue is built
for. Pieces separated by thousands cannot be crossed at any practical width and
have to be arrived at separately, which is what entries are for.

So, per corpus: the number of pieces, and the mutual rank between them. The
mutual rank of a pair is min(rank_u(v), rank_v(u)), computed over the whole
corpus, which is the quantity a queue of width L can or cannot bridge.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["msmarco-1m", "gist-960", "glove-100"])
    ap.add_argument("--root", type=Path, default=Path("/dev/shm/gpuann"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--cap", type=int, default=20000,
                    help="ranks beyond this are reported as > cap")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    print(f"{'corpus':13} {'pieces':>7} {'mutual rank between pieces':>34}")
    print(f"{'':13} {'(phi_1)':>7} {'p10':>8} {'p50':>8} {'p90':>8} {'>cap':>7}")
    rows = []
    for name in a.data:
        d = a.root / name
        man = json.loads((d / "manifest.json").read_text())
        ip = man["metric"].startswith("ip")
        X = torch.from_numpy(np.load(d / "X.npy")).cuda()
        gt = np.load(d / f"gt_test_ids.npy")[: a.nq, : a.k]
        frag = np.load(d / f"frag1_test_{Path(a.graph).stem}.npy")[: a.nq]
        nq = gt.shape[0]
        nc = np.array([len(set(r)) for r in frag])
        mr = []
        for q in range(nq):
            ids = torch.from_numpy(gt[q].astype(np.int64)).cuda()
            V = X[ids]
            # rank of each answer among the other's neighbours, over the corpus
            # rank_u(v): how many corpus points are strictly closer to u than v is
            if ip:
                s = V @ X.T
                r = torch.empty((a.k, a.k), dtype=torch.long, device=X.device)
                for i in range(a.k):
                    thr = s[i, ids]                              # [k]
                    r[i] = (s[i].view(1, -1) > thr.view(-1, 1)).sum(1)
                del s
            else:
                s = torch.cdist(V, X)
                r = torch.empty((a.k, a.k), dtype=torch.long, device=X.device)
                for i in range(a.k):
                    thr = s[i, ids]
                    r[i] = (s[i].view(1, -1) < thr.view(-1, 1)).sum(1)
                del s
            mut = torch.minimum(r, r.T).cpu().numpy()
            lab = frag[q]
            for i in range(a.k):
                for j in range(i + 1, a.k):
                    if lab[i] != lab[j]:                        # different pieces
                        mr.append(int(mut[i, j]))
            del V, r
        mr = np.array(mr)
        over = float((mr > a.cap).mean())
        print(f"{name:13} {nc.mean():7.2f} {np.percentile(mr,10):8.0f} "
              f"{np.percentile(mr,50):8.0f} {np.percentile(mr,90):8.0f} "
              f"{100*over:6.1f}%")
        rows.append(dict(corpus=name, phi1=float(nc.mean()), n_pairs=len(mr),
                         p10=float(np.percentile(mr, 10)),
                         p50=float(np.percentile(mr, 50)),
                         p90=float(np.percentile(mr, 90)), over_cap=over))
        del X
        torch.cuda.empty_cache()
    print(f"\na queue of width L carries a candidate ranked up to roughly L x the\n"
          f"fan-out behind the frontier, so pieces at mutual rank in the hundreds\n"
          f"are bridgeable by widening L and pieces in the tens of thousands are not.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
