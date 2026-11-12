"""What is different about the corpora our search does worst on.

Fragmentation is a per-query property and the paper has no dataset-level
predictor for it. These are the descriptors that could be one, all computed the
same way on every corpus so the columns are comparable:

  contrast        d(q, random) / d(q, 1st) -- how much nearer the nearest
                  neighbour is than an arbitrary point. Low contrast means the
                  answers are not much closer than anything else.
  answer spread   d(q, 10th) / d(q, 1st) -- how wide the answer set itself is.
                  A wide set is one whose members are far from each other.
  hubness         skewness of the k-occurrence distribution, the number of times
                  a point appears in others' top-k. A few hubs absorbing all the
                  in-edges leaves everything else hard to arrive at.
  rank reach      the share of top-10 pairs within the builder's candidate width,
                  which bounds what any prune could keep.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import skew


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+",
                    default=["msmarco-1m", "gist-960", "glove-100"])
    ap.add_argument("--root", type=Path, default=Path("/dev/shm/gpuann"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cand", type=int, default=128)
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--nhub", type=int, default=200_000, help="corpus sample for hubness")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    print(f"{'corpus':13} {'d':>5} {'phi_1':>6} {'contrast':>9} {'spread':>7} "
          f"{'hubness':>8} {'pairs within C':>15}")
    rows = []
    for name in a.data:
        d = a.root / name
        man = json.loads((d / "manifest.json").read_text())
        ip = man["metric"].startswith("ip")
        X = torch.from_numpy(np.load(d / "X.npy")).cuda()
        Q = torch.from_numpy(np.load(d / "Q_test.npy")[: a.nq]).cuda()
        gt = np.load(d / f"gt_test_ids.npy")[: a.nq, : a.k]
        frag = np.load(d / f"frag1_test_G_d64.npy")[: a.nq]
        n = X.shape[0]
        phi = float(np.mean([len(set(r)) for r in frag]))

        # contrast and spread, in whatever the corpus's own distance is
        g = torch.Generator(device="cuda").manual_seed(0)
        samp = torch.randint(0, n, (4096,), generator=g, device="cuda")
        if ip:
            dq = 1.0 - Q @ X[samp].T
            dgt = 1.0 - torch.einsum("qd,qkd->qk", Q,
                                     X[torch.from_numpy(gt.astype(np.int64)).cuda()])
        else:
            dq = torch.cdist(Q, X[samp])
            dgt = torch.cdist(Q.unsqueeze(1),
                              X[torch.from_numpy(gt.astype(np.int64)).cuda()]).squeeze(1)
        contrast = float((dq.mean(1) / dgt[:, 0].clamp(min=1e-9)).mean())
        spread = float((dgt[:, -1] / dgt[:, 0].clamp(min=1e-9)).mean())

        # hubness: how skewed is the number of times a point is in others' top-k
        s = samp[: a.nhub // 16]
        occ = torch.zeros(n, dtype=torch.int32, device="cuda")
        for i in range(0, len(s), 256):
            V = X[s[i:i+256]]
            sc = (V @ X.T) if ip else -torch.cdist(V, X)
            nb = torch.topk(sc, a.k + 1, dim=1).indices[:, 1:]
            occ.scatter_add_(0, nb.reshape(-1),
                             torch.ones(nb.numel(), dtype=torch.int32, device="cuda"))
            del sc, nb, V
        hub = float(skew(occ.cpu().numpy().astype(np.float64)))

        # what share of top-10 pairs the candidate width can even offer
        within = tot = 0
        for q in range(min(a.nq, 150)):
            ids = torch.from_numpy(gt[q].astype(np.int64)).cuda()
            V = X[ids]
            sc = (V @ X.T) if ip else -torch.cdist(V, X)
            r = torch.empty((a.k, a.k), dtype=torch.long, device="cuda")
            for i in range(a.k):
                r[i] = (sc[i].view(1, -1) > sc[i, ids].view(-1, 1)).sum(1)
            mut = torch.minimum(r, r.T).cpu().numpy()
            for i in range(a.k):
                for j in range(i + 1, a.k):
                    tot += 1
                    within += int(mut[i, j] < a.cand)
            del sc, r, V
        print(f"{name:13} {X.shape[1]:5} {phi:6.2f} {contrast:9.2f} {spread:7.3f} "
              f"{hub:8.2f} {100*within/tot:14.1f}%")
        rows.append(dict(corpus=name, d=X.shape[1], phi1=phi, contrast=contrast,
                         spread=spread, hubness=hub, pairs_within_C=within / tot))
        del X, Q, occ
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
