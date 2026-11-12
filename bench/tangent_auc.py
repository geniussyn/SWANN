"""Is a fragment a direction seen from the query?

Fragmentation is the condition that a query's answers are near the query and far
from each other. Written as vectors that says the answers sit at different
*bearings* from q, not merely that they are mutually dissimilar: two documents
can be far apart because one lies further along the same bearing, which is not a
second fragment, and two can be close yet lie either side of q, which is.

The bearing from q to e, for normalised vectors, is the great-circle tangent

    t_e = (e - (q.e) q) / ||e - (q.e) q||          (SYN eq. 9 uses the same axis)

so this scores two separators of "same fragment or not": plain similarity e1.e2,
which the entry selectors already use, and the tangent cosine t_e1 . t_e2, which
nothing has used. AUC is over GT pairs of one query, labelled by the fragment
labels of the serving graph.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch


def auc(scores, labels):
    """P(score of a positive > score of a negative), ties at 0.5."""
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks over ties
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    npos = labels.sum()
    nneg = len(labels) - npos
    if npos == 0 or nneg == 0:
        return float("nan")
    return (ranks[labels == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    tag = Path(a.graph).stem
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")
    nq = min(a.nq, len(gt))
    Q, gt, frag = Q[:nq], gt[:nq], frag[:nq]

    sim_s, tan_s, lab = [], [], []
    for s in range(0, nq, 256):
        e = min(s + 256, nq)
        q = Q[s:e]                                            # [b, d]
        g = torch.from_numpy(gt[s:e].astype(np.int64)).cuda()
        V = X[g.reshape(-1)].view(e - s, a.k, -1)             # [b, k, d]
        if not ip:                                            # put L2 data on the sphere
            V = torch.nn.functional.normalize(V, dim=-1)
            q = torch.nn.functional.normalize(q, dim=-1)
        pair = torch.bmm(V, V.transpose(1, 2))                # plain similarity
        aq = (V * q.unsqueeze(1)).sum(-1, keepdim=True)       # [b, k, 1]
        T = V - aq * q.unsqueeze(1)                           # tangent at q
        T = torch.nn.functional.normalize(T, dim=-1)
        tcos = torch.bmm(T, T.transpose(1, 2))                # bearing agreement
        fl = torch.from_numpy(frag[s:e].astype(np.int64)).cuda()
        same = (fl.unsqueeze(2) == fl.unsqueeze(1))
        iu = torch.triu_indices(a.k, a.k, offset=1)
        sim_s.append(pair[:, iu[0], iu[1]].reshape(-1).cpu().numpy())
        tan_s.append(tcos[:, iu[0], iu[1]].reshape(-1).cpu().numpy())
        lab.append(same[:, iu[0], iu[1]].reshape(-1).cpu().numpy().astype(np.int8))
        del V, pair, T, tcos
    sim_s = np.concatenate(sim_s); tan_s = np.concatenate(tan_s); lab = np.concatenate(lab)
    print(f"{a.data.name} {a.graph}  q={nq}  pairs={len(lab):,}  "
          f"same-fragment {100*lab.mean():.1f}%")
    a_sim, a_tan = auc(sim_s, lab), auc(tan_s, lab)
    print(f"  AUC  plain similarity   {a_sim:.4f}")
    print(f"  AUC  tangent bearing    {a_tan:.4f}   ({a_tan - a_sim:+.4f})")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, graph=a.graph, nq=nq,
                                         pairs=int(len(lab)), auc_sim=float(a_sim),
                                         auc_tangent=float(a_tan)), indent=2))


if __name__ == "__main__":
    main()
