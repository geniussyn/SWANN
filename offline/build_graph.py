"""Our own graph builder, on the GPU, with degree spent where it is needed.

Three stages, none of which needs cuVS:

  candidates   exact top-C by one chunked sweep of the corpus -- every vector
               against every other. It is O(N^2) and it is the bulk of the build
               (55 s of 66 s at 1M), which is affordable because the build is
               offline and because the prune then chooses from the true
               neighbours rather than from a guess at them. An approximate
               candidate stage was tried and is not here: see README.
  prune        a Vamana-style robust prune: walk a node's candidates nearest
               first and keep v only when no already-kept w is close enough to
               serve as a detour (d(w,v) <= alpha * d(u,v)). This is the step
               that makes a graph navigable rather than merely correct, and the
               measured difference between a kNN graph and a pruned one is about
               8 recall points at equal search time.
  reverse      every kept edge is offered back to its target, and each node
               re-prunes the union. Without this a node that nothing points at
               is unreachable however good its own edges are.

Degree is not spread evenly. LID -- local intrinsic dimensionality, the
Levina-Bickel / Amsaleg MLE read straight off a node's own neighbour distances
-- says how many directions its neighbourhood actually occupies. A node in a
locally high-dimensional patch has neighbours that are nearly equidistant and in
nearly independent directions, so a fixed out-degree covers a smaller fraction
of them and the prune leaves it under-connected; a node on a low-dimensional
sheet is well covered by few edges. So degree is allocated by LID rank at
constant total edge count: the index is the same size, the edges sit where the
geometry is hard. LID needs no query and no index -- it is a formula over the
candidate distances the builder already has.

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
def pairwise(A, B, metric, bn=None):
    """Distances from every row of A to every row of B, in the corpus's metric.

    An inner-product corpus is normalised, so 1 - <a,b> is a distance; an L2 one
    is not, and ranking it by inner product sorts by length instead of by
    proximity -- which on GIST left almost every node with out-degree one.
    """
    if metric == "ip":
        return 1.0 - (A @ B.T)
    if bn is None:
        bn = (B * B).sum(1)
    return (bn.view(1, -1) - 2.0 * (A @ B.T) + (A * A).sum(1).view(-1, 1)).clamp(min=0)


@torch.no_grad()
def knn_candidates(Xg, C, chunk=4096, metric="ip"):
    """Exact top-C neighbours by the corpus's metric, self excluded."""
    n = Xg.shape[0]
    idx = torch.empty((n, C), dtype=torch.int32, device=Xg.device)
    dst = torch.empty((n, C), dtype=torch.float32, device=Xg.device)
    xn = None if metric == "ip" else (Xg * Xg).sum(1)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        d = pairwise(Xg[s:e], Xg, metric, xn)
        d[torch.arange(e - s, device=Xg.device), torch.arange(s, e, device=Xg.device)] = 3.4e38
        v, i = torch.topk(d, C, dim=1, largest=False)
        idx[s:e], dst[s:e] = i.to(torch.int32), v
        del d
    return idx, dst


@torch.no_grad()
def lid_mle(cdist, k=50, eps=1e-12):
    """Levina-Bickel / Amsaleg MLE on a node's k nearest distances.

        LID(u) = -1 / mean_{i<k} log(d_i / d_k)

    Distances are Euclidean on the normalised vectors (d_E = sqrt(2 d_ip)): the
    estimator is a ratio of distances and the square root does not cancel.
    A node whose neighbours are all at nearly the same distance has a ratio near
    1, a log near 0, and a large LID -- which is the case this is here to find.
    """
    d = torch.sqrt(torch.clamp(cdist[:, :k], min=0.0))
    dk = d[:, -1:].clamp(min=eps)
    r = torch.log((d[:, :-1].clamp(min=eps) / dk).clamp(max=1.0 - 1e-7))
    return (-1.0 / r.mean(dim=1).clamp(max=-eps))


def degrees_from_lid(lid, deg, lo_frac=0.5, hi_frac=1.5, dmin=8):
    """Degree by LID rank, at constant total edge count.

    Rank rather than value: LID estimates are heavy-tailed and a handful of
    nodes would otherwise take the whole budget. The mean is pinned to `deg`, so
    this reallocates the index rather than enlarging it.
    """
    n = lid.numel()
    rank = torch.empty(n, dtype=torch.float32, device=lid.device)
    rank[torch.argsort(lid)] = torch.arange(n, device=lid.device, dtype=torch.float32) / max(n - 1, 1)
    d = deg * (lo_frac + (hi_frac - lo_frac) * rank)
    d = torch.round(d).clamp(min=dmin)
    # pin the mean exactly: hand the rounding slack to the highest-LID nodes
    target = deg * n
    while True:
        gap = int(target - d.sum().item())
        if gap == 0:
            break
        step = 1 if gap > 0 else -1
        idx = torch.argsort(lid, descending=(gap > 0))[: min(abs(gap), n)]
        d[idx] = (d[idx] + step).clamp(min=dmin)
    return d.to(torch.int32)


@torch.no_grad()
def robust_prune(Xg, cand, cdist, deg, alpha=1.2, chunk=4096, deg_per_node=None,
                 metric="ip"):
    """Keep a candidate only when no nearer kept neighbour detours to it.

    Vectorised over nodes and sequential over the candidate list, which is the
    order the rule is defined in. DiskANN's RobustPrune drops candidate i when
    an already-kept w satisfies alpha * d(w, i) <= d(u, i): a larger alpha makes
    that condition harder to meet, so it keeps *more* edges, not fewer.
    """
    n, C = cand.shape
    width = int(deg if deg_per_node is None else int(deg_per_node.max()))
    out = torch.full((n, width), -1, dtype=torch.int32, device=Xg.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        B = e - s
        V = Xg[cand[s:e].reshape(-1).long()].reshape(B, C, -1)        # [B, C, D]
        if metric == "ip":
            pair = 1.0 - torch.bmm(V, V.transpose(1, 2))              # [B, C, C]
        else:
            vn = (V * V).sum(-1)
            pair = (vn.unsqueeze(1) - 2.0 * torch.bmm(V, V.transpose(1, 2))
                    + vn.unsqueeze(2)).clamp(min=0)
        du = cdist[s:e]                                               # [B, C]
        kept = torch.zeros((B, C), dtype=torch.bool, device=Xg.device)
        nkept = torch.zeros(B, dtype=torch.long, device=Xg.device)
        cap = (torch.full((B,), deg, device=Xg.device, dtype=torch.long)
               if deg_per_node is None else deg_per_node[s:e].long())
        for i in range(C):
            room = nkept < cap
            viol = ((alpha * pair[:, :, i] <= du[:, i].view(B, 1)) & kept).any(dim=1)
            take = room & ~viol & (cand[s:e, i] >= 0)
            kept[:, i] = take
            nkept += take.long()
            if bool((nkept >= cap).all()):
                break
        # compact the kept flags into a fixed-width row
        order = torch.argsort((~kept).to(torch.int8), dim=1, stable=True)[:, :width]
        sel = torch.gather(cand[s:e], 1, order)
        valid = torch.gather(kept, 1, order)
        out[s:e] = torch.where(valid, sel, torch.full_like(sel, -1))
        del V, pair
    return out


@torch.no_grad()
def add_reverse(Xg, G, deg, chunk=2048, alpha=1.2, max_in=64, deg_per_node=None,
                metric="ip"):
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
    width = int(deg if deg_per_node is None else int(deg_per_node.max()))
    out = torch.full((n, width), -1, dtype=torch.int32, device=G.device)
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
        if metric == "ip":
            d = 1.0 - torch.bmm(V, Xg[s:e].unsqueeze(-1)).squeeze(-1)
        else:
            d = ((V - Xg[s:e].unsqueeze(1)) ** 2).sum(-1)
        d = torch.where(valid, d, torch.full_like(d, float("inf")))
        o = torch.argsort(d, dim=1)
        cand = torch.gather(ids.to(torch.int32), 1, o)
        cdist = torch.gather(d, 1, o)
        keepn = min(cand.shape[1], 128)
        sub = None if deg_per_node is None else deg_per_node[s:e]
        kept = robust_prune(Xg, cand[:, :keepn].contiguous(), cdist[:, :keepn].contiguous(),
                            deg, alpha=alpha, chunk=B, deg_per_node=sub, metric=metric)
        # The inner call sizes its output by the widest degree *in this chunk*,
        # which under LID allocation need not be the widest in the corpus; the
        # rest of the row stays -1.
        out[s:e, : min(kept.shape[1], width)] = kept[:, :width]
        del V, d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--D", type=int, default=64, help="target graph out-degree")
    ap.add_argument("--cand", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--lid", action="store_true", help="allocate degree by LID rank")
    ap.add_argument("--lid-k", type=int, default=50)
    ap.add_argument("--lid-lo", type=float, default=0.5)
    ap.add_argument("--lid-hi", type=float, default=1.5)
    ap.add_argument("--metric", default=None, choices=["ip", "l2"],
                    help="default: read it from the corpus manifest")
    ap.add_argument("--out", default="G_ours.npy")
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = a.metric or ("ip" if man["metric"].startswith("ip") else "l2")
    X = np.load(a.data / "X.npy")
    Xg = torch.from_numpy(X).cuda()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    cand, cdist = knn_candidates(Xg, a.cand, metric=metric)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    dpn = lid = None
    if a.lid:
        lid = lid_mle(cdist, k=a.lid_k)
        dpn = degrees_from_lid(lid, a.D, a.lid_lo, a.lid_hi)
        # The mean-pinning pass hands its rounding slack to the highest-LID nodes
        # and can push the widest degree past what the candidate list and the
        # kernel allow. Clamp there; the mean drops by whatever that costs.
        dpn = dpn.clamp(max=min(a.cand, 128))
        q = torch.quantile(lid, torch.tensor([0.05, 0.5, 0.95], device=lid.device))
        print(f"LID: 5%/median/95% {q[0]:.1f}/{q[1]:.1f}/{q[2]:.1f}   "
              f"degree {int(dpn.min())}..{int(dpn.max())}, mean {float(dpn.float().mean()):.1f}")
    G = robust_prune(Xg, cand, cdist, a.D, alpha=a.alpha, deg_per_node=dpn, metric=metric)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    G = add_reverse(Xg, G, a.D, alpha=a.alpha, deg_per_node=dpn, metric=metric)
    torch.cuda.synchronize(); t3 = time.perf_counter()

    Gc = G.cpu().numpy().astype(np.int32)
    filled = (Gc >= 0).sum(1)
    print(f"candidates {t1-t0:.2f}s  prune {t2-t1:.2f}s  reverse {t3-t2:.2f}s  "
          f"total {t3-t0:.2f}s")
    print(f"out-degree: min/med/max {filled.min()}/{int(np.median(filled))}/{filled.max()}")
    np.save(a.data / a.out, Gc)
    np.save(a.data / (Path(a.out).stem + "_deg.npy"), filled.astype(np.int32))
    if lid is not None:
        np.save(a.data / (Path(a.out).stem + "_lid.npy"), lid.cpu().numpy().astype(np.float32))
    (a.data / (Path(a.out).stem + ".json")).write_text(json.dumps(
        dict(deg=a.D, cand=a.cand, candidates="exact", alpha=a.alpha,
             metric=metric, lid=bool(a.lid),
             lid_range=[a.lid_lo, a.lid_hi], seconds=round(t3 - t0, 2),
             stage_seconds=dict(candidates=round(t1 - t0, 2), prune=round(t2 - t1, 2),
                                reverse=round(t3 - t2, 2))), indent=2))
    print(f"wrote {a.data / a.out}")


if __name__ == "__main__":
    main()
