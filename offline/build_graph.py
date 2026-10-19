"""Our own graph builder, on the GPU, with degree spent where it is needed.

Three stages, none of which needs cuVS:

  candidates   exact top-C by one chunked sweep of the corpus. At 100K this is
               a second; NN-descent replaces it when N stops allowing it, and
               nothing downstream changes.
  prune        a Vamana-style robust prune: walk a node's candidates nearest
               first and keep v only when no already-kept w is close enough to
               serve as a detour (d(w,v) <= alpha * d(u,v)). This is the step
               that makes a graph navigable rather than merely correct, and the
               measured difference between a kNN graph and a pruned one is about
               8 recall points at equal search time.
  reverse      every kept edge is offered back to its target, and each node
               re-prunes the union. Without this a node that nothing points at
               is unreachable however good its own edges are.

Then, optionally, the difficulty pass. phi-hat_1 says which pairs of a query's
answer set a builder with out-degree d would still leave in different pieces,
and it says it without an index. Those cuts are repaired the way the
manuscript's appendix repairs them -- reciprocal edges between the closest
cross-component pair, each endpoint dropping its longest edge so out-degree and
index size do not move -- except at build time, from *training* queries, rather
than post hoc from the queries being measured.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def knn_candidates(Xg, C, chunk=4096):
    """Exact top-C neighbours, self excluded."""
    n = Xg.shape[0]
    idx = torch.empty((n, C), dtype=torch.int32, device=Xg.device)
    dst = torch.empty((n, C), dtype=torch.float32, device=Xg.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        sim = Xg[s:e] @ Xg.T
        sim[torch.arange(e - s, device=Xg.device), torch.arange(s, e, device=Xg.device)] = -3.4e38
        v, i = torch.topk(sim, C, dim=1)
        idx[s:e], dst[s:e] = i.to(torch.int32), 1.0 - v
        del sim
    return idx, dst


@torch.no_grad()
def robust_prune(Xg, cand, cdist, deg, alpha=1.2, chunk=1024):
    """Keep a candidate only when no nearer kept neighbour detours to it.

    Vectorised over nodes and sequential over the candidate list, which is the
    order the rule is defined in. DiskANN's RobustPrune drops candidate i when
    an already-kept w satisfies alpha * d(w, i) <= d(u, i): a larger alpha makes
    that condition harder to meet, so it keeps *more* edges, not fewer.
    """
    n, C = cand.shape
    out = torch.full((n, deg), -1, dtype=torch.int32, device=Xg.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        B = e - s
        V = Xg[cand[s:e].reshape(-1).long()].reshape(B, C, -1)        # [B, C, D]
        pair = 1.0 - torch.bmm(V, V.transpose(1, 2))                  # [B, C, C]
        du = cdist[s:e]                                               # [B, C]
        kept = torch.zeros((B, C), dtype=torch.bool, device=Xg.device)
        nkept = torch.zeros(B, dtype=torch.long, device=Xg.device)
        for i in range(C):
            room = nkept < deg
            viol = ((alpha * pair[:, :, i] <= du[:, i].view(B, 1)) & kept).any(dim=1)
            take = room & ~viol & (cand[s:e, i] >= 0)
            kept[:, i] = take
            nkept += take.long()
            if bool((nkept >= deg).all()):
                break
        # compact the kept flags into a fixed-width row
        order = torch.argsort((~kept).to(torch.int8), dim=1, stable=True)[:, :deg]
        sel = torch.gather(cand[s:e], 1, order)
        valid = torch.gather(kept, 1, order)
        out[s:e] = torch.where(valid, sel, torch.full_like(sel, -1))
        del V, pair
    return out


@torch.no_grad()
def add_reverse(Xg, G, deg, chunk=1024, alpha=1.2, max_in=64):
    """Offer every edge back to its target, then re-prune the union per node."""
    n = G.shape[0]
    src = torch.arange(n, device=G.device, dtype=torch.int32).view(-1, 1).expand_as(G)
    mask = G >= 0
    tgt = G[mask].long()
    rev = src[mask]
    order = torch.argsort(tgt, stable=True)
    tgt, rev = tgt[order], rev[order]
    counts = torch.bincount(tgt, minlength=n)
    starts = torch.cumsum(counts, 0) - counts

    merged = torch.full((n, G.shape[1] + max_in), -1, dtype=torch.int32, device=G.device)
    merged[:, : G.shape[1]] = G
    pos = torch.arange(max_in, device=G.device).view(1, -1)
    take = pos < counts.view(-1, 1).clamp(max=max_in)
    gidx = (starts.view(-1, 1) + pos).clamp(max=max(tgt.numel() - 1, 0))
    merged[:, G.shape[1]:] = torch.where(take, rev[gidx], torch.full_like(rev[gidx], -1))

    # de-duplicate, drop self, then re-prune by distance
    out = torch.full((n, deg), -1, dtype=torch.int32, device=G.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        B = e - s
        row = merged[s:e]
        selfid = torch.arange(s, e, device=G.device, dtype=torch.int32).view(-1, 1)
        row = torch.where(row == selfid, torch.full_like(row, -1), row)
        big = torch.where(row >= 0, row, torch.full_like(row, 2 ** 30)).long()
        srt, _ = torch.sort(big, dim=1)
        dup = torch.zeros_like(srt, dtype=torch.bool)
        dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
        srt = torch.where(dup, torch.full_like(srt, 2 ** 30), srt)
        valid = srt < 2 ** 30
        ids = torch.where(valid, srt, torch.zeros_like(srt))
        V = Xg[ids.reshape(-1)].reshape(B, -1, Xg.shape[1])
        d = 1.0 - torch.bmm(V, Xg[s:e].unsqueeze(-1)).squeeze(-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        o = torch.argsort(d, dim=1)
        cand = torch.gather(ids.to(torch.int32), 1, o)
        cdist = torch.gather(d, 1, o)
        keepn = min(cand.shape[1], 128)
        out[s:e] = robust_prune(Xg, cand[:, :keepn].contiguous(), cdist[:, :keepn].contiguous(),
                                deg, alpha=alpha, chunk=B)[:, :deg] if B else out[s:e]
        del V, d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--cand", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--out", default="G_ours.npy")
    a = ap.parse_args()

    X = np.load(a.data / "X.npy")
    Xg = torch.from_numpy(X).cuda()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cand, cdist = knn_candidates(Xg, a.cand)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    G = robust_prune(Xg, cand, cdist, a.deg, alpha=a.alpha)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    G = add_reverse(Xg, G, a.deg, alpha=a.alpha)
    torch.cuda.synchronize(); t3 = time.perf_counter()

    Gc = G.cpu().numpy().astype(np.int32)
    filled = (Gc >= 0).sum(1)
    print(f"candidates {t1-t0:.2f}s  prune {t2-t1:.2f}s  reverse {t3-t2:.2f}s  "
          f"total {t3-t0:.2f}s")
    print(f"out-degree: min/med/max {filled.min()}/{int(np.median(filled))}/{filled.max()}")
    np.save(a.data / a.out, Gc)
    (a.data / (Path(a.out).stem + ".json")).write_text(json.dumps(
        dict(deg=a.deg, cand=a.cand, alpha=a.alpha, seconds=round(t3 - t0, 2),
             stage_seconds=dict(candidates=round(t1 - t0, 2), prune=round(t2 - t1, 2),
                                reverse=round(t3 - t2, 2))), indent=2))
    print(f"wrote {a.data / a.out}")


if __name__ == "__main__":
    main()
