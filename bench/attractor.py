"""Can an entry ever land on an answer? The local-minimum ceiling.

If the objective is one entry per answer, the entry's descent has to stop *on*
the answer, and a greedy descent stops only at a local minimum of d(.,q) over
the graph. So an answer t is reachable this way only if no neighbour of t is
nearer to q than t is -- otherwise the descent walks past t to that neighbour.

    Recall@k of a pure descent <= |{t in T_k(q) : t is a local min for q}| / k

This needs no search, the same way the region bound needs no search: a graph,
the ground truth and the queries are enough. It is the ceiling L = 1 is working
against, and it also says what a wider queue is buying -- a queue of L keeps the
answers that are *not* attractors, which a descent would have thrown away.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")).cuda()
    gt = torch.from_numpy(np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]).cuda().long()
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    nq, k = gt.shape
    D = G.shape[1]

    def dist(q, ids):
        V = X[ids.reshape(-1)].view(*ids.shape, -1)
        if ip:
            return 1.0 - (V * q.unsqueeze(1)).sum(-1) if ids.dim() == 2 else \
                   1.0 - (V * q.unsqueeze(1).unsqueeze(1)).sum(-1)
        return ((V - (q.unsqueeze(1) if ids.dim() == 2 else q.unsqueeze(1).unsqueeze(1))) ** 2).sum(-1)

    is_min = torch.zeros((nq, k), dtype=torch.bool, device="cuda")
    better = torch.zeros((nq, k), dtype=torch.int32, device="cuda")
    for s in range(0, nq, 64):
        e = min(s + 64, nq)
        q = Q[s:e]                                   # [b, d]
        t = gt[s:e]                                  # [b, k]
        dt = dist(q, t)                              # [b, k]
        nb = G[t.reshape(-1)].view(e - s, k, D).long()      # [b, k, D]
        ok = nb >= 0
        Vn = X[nb.clamp(min=0).reshape(-1)].view(e - s, k, D, -1)
        if ip:
            dn = 1.0 - (Vn * q.view(e - s, 1, 1, -1)).sum(-1)
        else:
            dn = ((Vn - q.view(e - s, 1, 1, -1)) ** 2).sum(-1)
        dn = torch.where(ok, dn, torch.full_like(dn, float("inf")))
        nearer = (dn < dt.unsqueeze(-1))
        better[s:e] = nearer.sum(-1).to(torch.int32)
        is_min[s:e] = ~nearer.any(-1)
        del Vn, dn, nb

    bound = is_min.float().mean(1)                   # per query
    print(f"{a.data.name} {a.graph}  q={nq} k={k} D={D}\n")
    print(f"지역 최소점인 정답 비율 (= 순수 하강 recall 상한)  {bound.mean():.4f}")
    print(f"정답 전부가 지역 최소점인 쿼리                      {100*(bound>=1).float().mean():.1f}%")
    print(f"지역 최소점이 하나도 없는 쿼리                       {100*(bound<=0).float().mean():.1f}%")
    b = better.float()
    print(f"\n정답 하나당 '더 가까운 이웃' 수   평균 {b.mean():.2f}  "
          f"중위 {int(b.median())}  최대 {int(b.max())}")
    hist = torch.histc(b.clamp(max=16), bins=17, min=0, max=16).cpu().numpy()
    tot = hist.sum()
    print(f"\n{'더 가까운 이웃 수':>16} {'정답 비율':>10}")
    for i, c in enumerate(hist):
        if c == 0:
            continue
        print(f"{i if i < 16 else '16+':>16} {100*c/tot:9.1f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, nq=nq, k=k, D=D,
            descent_ceiling=float(bound.mean()),
            frac_all_min=float((bound >= 1).float().mean()),
            frac_none_min=float((bound <= 0).float().mean()),
            mean_nearer_neighbours=float(b.mean())), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
