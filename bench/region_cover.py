"""Region coverage: the ceiling a region-partitioned index would impose.

If the index keeps only edges inside a region, a query can only recover the part
of its answer that sits in a region some entry was placed in:

    Recall@k(q) <= |T_k(q) ∩ ∪_{R : R has an entry}  R| / k

That bound needs no search -- a partition, the ground truth and the entries the
selector would pick are enough -- so it says whether dropping cross-region edges
is even allowed before any graph is built. Reported against E, because the whole
question is how many entries it takes to cover the regions the answer lives in.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch


def kmeans(Xg, R, iters=25, seed=0):
    """Lloyd on the GPU, seeded by a random sample. Regions are geometric: the
    partition a region-local index would be cut along."""
    n = Xg.shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    C = Xg[torch.randperm(n, generator=g)[:R].cuda()].clone()
    for _ in range(iters):
        lab = torch.empty(n, dtype=torch.int32, device=Xg.device)
        for s in range(0, n, 65536):
            e = min(s + 65536, n)
            lab[s:e] = torch.cdist(Xg[s:e], C).argmin(1).to(torch.int32)
        Cn = torch.zeros_like(C); cnt = torch.zeros(R, device=Xg.device)
        Cn.index_add_(0, lab.long(), Xg)
        cnt.index_add_(0, lab.long(), torch.ones(n, device=Xg.device))
        keep = cnt > 0
        C[keep] = Cn[keep] / cnt[keep].unsqueeze(1)
    return lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--R", type=int, nargs="+", default=[256, 1024, 4096],
                    help="regions in the partition")
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Qg = torch.from_numpy(Q).cuda()
    pool = np.load(a.data / "pools" / f"pool_{a.pool}.npy")[: a.P].astype(np.int64)

    # entries the shipped selector would pick: the V nearest pool nodes
    with torch.no_grad():
        Pg = Xg[torch.from_numpy(pool).cuda()]
        sim = Qg @ Pg.T if ip else -torch.cdist(Qg, Pg)
        order = sim.argsort(dim=1, descending=True).cpu().numpy()      # [nq, P]
    rows = []
    for R in a.R:
        lab = kmeans(Xg, R, seed=a.seed).cpu().numpy()
        sizes = np.bincount(lab, minlength=R)
        gtr = lab[gt]                                                  # [nq, k]
        n_gt_reg = np.array([len(set(r.tolist())) for r in gtr])
        print(f"\n== R={R} regions  size min/med/max "
              f"{sizes.min()}/{int(np.median(sizes))}/{sizes.max()}")
        print(f"   answer spans {n_gt_reg.mean():.2f} regions on average "
              f"(1 region: {100*(n_gt_reg==1).mean():.1f}% of queries)")
        print(f"{'V':>5} {'고른 구역 수':>12} {'커버리지 상한':>13} {'상한=1.0 비율':>13} "
              f"{'구역당 벡터':>11}")
        pool_reg = lab[pool]
        for V in a.V:
            if V > len(pool):
                continue
            sel = pool_reg[order[:, :V]]                               # [nq, V]
            nsel = np.array([len(set(r.tolist())) for r in sel])
            bound = np.array([np.isin(gtr[i], sel[i]).mean() for i in range(len(gtr))])
            vec = sizes[np.unique(sel)].mean()
            print(f"{V:5} {nsel.mean():12.2f} {bound.mean():13.4f} "
                  f"{100*(bound>=1.0).mean():12.1f}% {vec:11.0f}")
            rows.append(dict(R=R, V=V, regions_selected=float(nsel.mean()),
                             bound=float(bound.mean()),
                             frac_full=float((bound >= 1.0).mean()),
                             vectors_per_region=float(vec)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, k=a.k, nq=len(gt),
                                         pool=a.pool, P=a.P, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
