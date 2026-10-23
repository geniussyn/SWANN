"""Graph construction: exact candidates, then a prune that keeps detours short.

This is the build the paper measures. It is deliberately the simple one --- an
exact candidate sweep rather than an approximate one --- because the search is
the contribution and a reader should be able to reproduce the graph without
trusting a second approximation. It costs minutes on a million rows; see the
paper's cost figure for what that buys against an IVF index built in seconds.
"""
from pathlib import Path
import numpy as np
import torch


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

def robust_prune(Xg, cand, cdist, deg, alpha=1.2, chunk=4096, deg_per_node=None,
                 metric="ip", protect=None):
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
            if protect is not None:
                viol = viol & ~protect[s:e, i]      # a mutual pair survives the detour test
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
        # which with a pool boost need not be the widest in the corpus; the
        # rest of the row stays -1.
        out[s:e, : min(kept.shape[1], width)] = kept[:, :width]
        del V, d
    return out


@torch.no_grad()


def build_graph(X, metric: str = "ip", degree: int = 96,
                candidates: int = 128, alpha: float = 1.2):
    """Return an (n, degree) int32 neighbour table on the GPU.

    `candidates` is how many nearest neighbours each node is pruned from, and
    `alpha` is the detour tolerance: an edge survives if no kept neighbour is
    `alpha` times closer to its target than the node is.
    """
    import torch
    if X.device.type != "cuda":
        X = X.cuda()
    Xg = X.float().contiguous()
    cand, cdist = knn_candidates(Xg, candidates, metric=metric)
    G = robust_prune(Xg, cand, cdist, degree, alpha=alpha, metric=metric)
    G = add_reverse(Xg, G, degree, alpha=alpha, metric=metric)
    return G.to(torch.int32).contiguous()
