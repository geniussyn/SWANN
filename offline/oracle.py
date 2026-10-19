"""H0: how much is there to win, inside this candidate pool and this budget.

Every entry-selection idea in the design is a bet that *which* entries you start
from decides what you recover. Before building any selector, this measures the
ceiling on that bet exactly: run every pool seed on its own under the same
budget, record which bits of the query's exact top-10 each one recovers, and ask
the best E seeds could have done -- by dynamic programming over the 2^10 subsets,
not by a greedy approximation.

    dp[mask] = fewest seeds whose union covers mask
    oracle   = max popcount over masks with dp[mask] <= E

The number is an upper bound no selector can pass, and it is read against the
same arms the engine already runs. A low oracle means the pool or the per-branch
budget is wrong and no cleverness in selection will rescue it; a high oracle with
low measured recall is exactly the gap A and B exist to close.

GT is used here and only here, as a diagnostic. No selector may see it.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                            # noqa: E402


def seed_masks(Xg, Gg, Qg, pool, gt, k, m, budget, team=32, qchunk=128):
    """masks[Q, P] uint16: bit t set iff seed p's own best-k contained gt[q, t]."""
    nq, P = Qg.shape[0], len(pool)
    masks = np.zeros((nq, P), dtype=np.uint16)
    pool_t = torch.from_numpy(pool).view(1, P)
    gt_t = torch.from_numpy(gt.astype(np.int64))
    for s in range(0, nq, qchunk):
        e = min(s + qchunk, nq)
        seeds = pool_t.expand(e - s, P).contiguous()
        st = engine.init_states(Xg, Qg[s:e].contiguous(), seeds, m=m,
                                max_expansions=budget, deg=Gg.shape[1], team_size=team)
        engine.advance(Xg, Gg, Qg[s:e].contiguous(), st, budget)
        st.check()
        ids, _ = engine.branch_topk(st, k)                       # [q, P, k]
        hit = (ids.long().unsqueeze(-1) == gt_t[s:e].to(ids.device).view(-1, 1, 1, k))
        bits = hit.any(dim=2)                                    # [q, P, k] -> which gt slots
        w = (1 << torch.arange(k, device=bits.device, dtype=torch.int64))
        masks[s:e] = (bits.long() * w).sum(-1).to(torch.int32).cpu().numpy().astype(np.uint16)
        del st, ids, hit, bits
        torch.cuda.empty_cache()
    return masks


def dp_oracle(masks, k, E_list, device="cuda"):
    """Exact minimum-seed cover DP, run for all queries at once."""
    nq, P = masks.shape
    M = 1 << k
    INF = np.int16(P + 1)
    dp = torch.full((nq, M), int(INF), dtype=torch.int16, device=device)
    dp[:, 0] = 0
    allm = torch.arange(M, device=device, dtype=torch.int64)
    mk = torch.from_numpy(masks.astype(np.int64)).to(device)
    for p in range(P):
        idx = allm.view(1, M) | mk[:, p].view(nq, 1)              # merged mask per query
        dp = dp.scatter_reduce(1, idx, (dp + 1).clamp(max=int(INF)), reduce="amin")
    pc = torch.tensor([bin(x).count("1") for x in range(M)], device=device, dtype=torch.int16)
    out = {}
    for E in E_list:
        reach = (dp <= E)
        out[E] = (torch.where(reach, pc.view(1, M), torch.full_like(pc.view(1, M), -1))
                  .max(dim=1).values.float().cpu().numpy() / k)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--budget", type=int, default=16, help="per-branch expansions")
    ap.add_argument("--E", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--qchunk", type=int, default=128)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy"); G = np.load(a.data / "G.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    if a.nq:
        Q, gt = Q[: a.nq], gt[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()

    from bench.entry_arms import pool_ids
    pool = pool_ids(X.shape[0], a.P)

    masks = seed_masks(Xg, Gg, Qg, pool, gt, a.k, a.m, a.budget, qchunk=a.qchunk)
    per_seed = np.array([bin(int(x)).count("1") for x in masks.ravel()]).reshape(masks.shape) / a.k
    union_all = np.zeros(masks.shape[0], dtype=np.int64)
    for p in range(masks.shape[1]):
        union_all |= masks[:, p].astype(np.int64)
    whole_pool = np.array([bin(int(x)).count("1") for x in union_all]) / a.k

    oracle = dp_oracle(masks, a.k, a.E)
    print(f"{man['corpus']} n={man['n']} graph={man['graph']}  {a.split} q={Q.shape[0]}  "
          f"P={a.P} per-branch budget={a.budget} m={a.m}\n")
    print(f"mean recall of one pool seed alone      {per_seed.mean():.4f}")
    print(f"best single seed in the pool (oracle 1) {oracle[1].mean() if 1 in oracle else float('nan'):.4f}")
    print(f"whole pool, all {a.P} seeds             {whole_pool.mean():.4f}   <- ceiling of any selection\n")
    print(f"{'E':>3s} {'oracle recall@10':>17s} {'queries at 1.0':>15s}")
    rows = []
    for E in a.E:
        v = oracle[E]
        print(f"{E:3d} {v.mean():17.4f} {float((v >= 1.0).mean()):15.3f}")
        rows.append(dict(E=int(E), oracle=float(v.mean()), frac_perfect=float((v >= 1.0).mean())))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k, m=a.m, P=a.P,
                                         branch_budget=a.budget, nq=int(Q.shape[0]),
                                         mean_single_seed=float(per_seed.mean()),
                                         whole_pool=float(whole_pool.mean()),
                                         rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
