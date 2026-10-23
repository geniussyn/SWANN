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

Degree is uniform. Allocating it by local intrinsic dimension (the
Levina-Bickel / Amsaleg MLE off a node's own candidate distances) was built and
measured against a fixed-degree graph at matched edge count -- 52.1M against
51.6M on MS MARCO 1M -- and moved recall by +0.0005, inside the measurement
noise. That path is gone. What degree does buy is monotone and dull: 48.0M
edges reach 0.9903 and 51.6M reach 0.9929 at V=128.

The entry pool can be widened instead (--pool-boost). Every branch starts at a
pool node, so a pool node's reach is the resource the search actually spends,
and widening only those few thousand nodes costs a fraction of a percent of the
edges.
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
def candidates_from_graph(Xg, Gc, C, metric="ip", chunk=1024):
    """Candidates from an existing approximate kNN graph, with exact distances.

    The exact sweep is O(N^2) and is the bulk of the build; a graph another GPU
    builder already produced costs one distance per listed edge. Measured on
    MS MARCO 1M: 1.52 s against 55.24 s, for 0.0126 less recall -- so the sweep
    stays the default and this is the cheap option, not the better one.
    """
    n = Xg.shape[0]
    C = min(C, Gc.shape[1])
    idx = Gc[:, :C].to(torch.int32).contiguous()
    dst = torch.full((n, C), 3.4e38, dtype=torch.float32, device=Xg.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        nb = idx[s:e].long()
        ok = nb >= 0
        V = Xg[nb.clamp(min=0).reshape(-1)].view(e - s, C, -1)
        A = Xg[s:e].unsqueeze(1)
        d = (1.0 - (A * V).sum(-1)) if metric == "ip" else ((A - V) ** 2).sum(-1)
        d = torch.where(ok, d, torch.full_like(d, 3.4e38))
        d = torch.where(nb == torch.arange(s, e, device=Xg.device).view(-1, 1),
                        torch.full_like(d, 3.4e38), d)
        v, order = torch.sort(d, dim=1)
        dst[s:e] = v
        idx[s:e] = torch.gather(idx[s:e], 1, order)
        del V, A, d
    return idx, dst


@torch.no_grad()
def mutual_rank(cand, M, chunk=2048):
    """For each listed pair, is it mutual within rank M?

    W(u,v) = min(rank_u(v), rank_v(u)), the quantity phi-hat_1 is defined on:
    two documents are adjacent when each is near the top of the other's list.
    Returned as a boolean over the candidate matrix, so the prune can read it
    per (node, candidate) without a second index.

    The reason to care is measured rather than assumed. Of the ground truth a
    search misses, 83.2% sits alone in its piece -- no edge to any other answer
    of the same query -- against 16.2% of the ground truth it finds, and those
    isolated documents have perfectly ordinary degree (median 48 out, 44 in).
    They are not badly connected to the corpus; they are disconnected from the
    rest of the answer. A prune that optimises each node's list on its own has
    no term that would prevent it: if u drops v and v drops u, the pair is cut
    for good and the reverse pass only restores some of it.
    """
    n, C = cand.shape
    M = min(M, C)
    out = torch.zeros((n, C), dtype=torch.bool, device=cand.device)
    head = cand[:, :M].contiguous()                       # v's own top-M
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        v = cand[s:e, :M].long().clamp(min=0)             # [b, M] -- u's top-M
        back = head[v.reshape(-1)].view(e - s, M, M)      # [b, M, M] their top-M
        me = torch.arange(s, e, device=cand.device, dtype=torch.int32).view(-1, 1, 1)
        out[s:e, :M] = (back == me).any(-1) & (cand[s:e, :M] >= 0)
    return out


@torch.no_grad()
def fill_degree(G, cand, rule="bridge", chunk=1024, deg_two=None):
    """Spend the degree slots the prune left empty, on the candidates it refused.

    The prune drops a candidate whenever an already-kept neighbour detours to it.
    That keeps navigation intact -- the detour is a usable path -- but it does not
    keep the two nodes mutually one hop apart, which is what a query's answers
    need in order to be mined together once one of them is reached. On GIST the
    cost of that is visible twice over: the degree is only 39% filled, and 82.7%
    of the cross-piece pairs that the candidate list did offer were discarded.

    So the slots go back to refused candidates. Which ones matters, and the three
    rules here differ in what they believe is missing:

      near    the nearest refused candidates. This is what an unpruned kNN graph
              is, and an unpruned kNN graph is the worst of all the GIST graphs
              measured (phi_1 7.58, recall 0.84): short edges duplicate reach
              the graph already has.
      far     the farthest refused candidates. Cross-piece pairs sit at mutual
              rank ~539 against a candidate width of 128, so they are at the far
              end of the list, not the near end.
      bridge  the refused candidates with no two-hop path back through the kept
              graph. This asks the prune's own question -- is there another way
              there -- but over paths instead of over one detour vertex, which is
              the cheap local stand-in for an effective resistance.
    """
    n, width = G.shape
    C = cand.shape[1]
    out = G.clone()
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        B = e - s
        kept = out[s:e].long()                                    # [B, width]
        cd = cand[s:e].long()                                     # [B, C]
        # a candidate already present, or pointing at the node itself, is not free
        here = torch.arange(s, e, device=G.device).view(-1, 1)
        taken = (cd.unsqueeze(2) == kept.unsqueeze(1)).any(2) | (cd == here) | (cd < 0)
        if rule == "bridge":
            # the kept neighbours' own neighbours: everything two hops out
            two = out[kept.clamp(min=0).reshape(-1)].long().reshape(B, -1)
            two = torch.where((kept < 0).repeat_interleave(width, dim=1), -1, two)
            two, _ = two.sort(dim=1)
            pos = torch.searchsorted(two, cd.contiguous())
            pos = pos.clamp(max=two.shape[1] - 1)
            reachable = torch.gather(two, 1, pos) == cd
            free = ~taken & ~reachable
            del two
        else:
            free = ~taken
        room = int((out[s:e] < 0).sum(1).max())
        if room == 0:
            continue
        # candidate columns are ordered by distance, so column order is the rule
        cols = torch.arange(C, device=G.device).view(1, -1).expand(B, -1)
        key = torch.where(free, cols if rule != "far" else (C - 1 - cols),
                          torch.full_like(cols, C + 1))
        pick = torch.argsort(key, dim=1)
        empty = torch.argsort((out[s:e] >= 0).to(torch.int8), dim=1, stable=True)
        nfree = free.sum(1)
        nempty = (out[s:e] < 0).sum(1)
        take = torch.minimum(nfree, nempty)
        for t in range(int(take.max())):
            m = take > t
            if not bool(m.any()):
                break
            rows = m.nonzero().view(-1)
            src = torch.gather(cd[rows], 1, pick[rows, t].view(-1, 1)).view(-1)
            dst = empty[rows, t]
            out[s + rows, dst] = src.to(torch.int32)
    return out


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
def enforce_min_in(Xg, G, cand, min_in, metric="ip", rounds=4):
    """Give every node a floor on its in-degree, at a fixed edge count.

    Out-degree is what the prune spends; in-degree is what decides whether a
    node can be arrived at, and the two are not the same. Measured on MS MARCO
    1M our graph leaves its first percentile at 9 in-edges where CAGRA's own
    builder holds 35, and on GIST 0.26% of nodes -- 0.14% of the ground truth --
    end with none at all, which no search can recover. The prune has no reason
    to notice: it optimises each node's own list.

    The repair offers a weak node to its nearest candidates, and each of those
    makes room by dropping one edge. *Which* edge decides whether this
    converges: dropping the farthest one regardless of its head sends the
    shortfall somewhere else, and the pass stalls (measured: 13,339 edges to
    place, then 7,225, then 7,216, with the floor still unmet). Dropping the
    farthest edge whose head is itself above the floor moves in-degree from rich
    to poor and cannot create a new deficit.
    """
    n, W = G.shape
    for r in range(rounds):
        indeg = torch.zeros(n, dtype=torch.int32, device=G.device)
        m = G >= 0
        indeg.scatter_add_(0, G[m].long(),
                           torch.ones(int(m.sum()), dtype=torch.int32, device=G.device))
        weak = torch.nonzero(indeg < min_in).squeeze(1)
        if weak.numel() == 0:
            print(f"  min-in {min_in}: met after {r} rounds")
            break
        need = (min_in - indeg[weak]).clamp(max=cand.shape[1])
        print(f"  min-in {min_in} round {r}: {weak.numel():,} below, "
              f"{int(need.sum()):,} edges to place")
        placed = skipped = 0
        remaining = torch.zeros(n, dtype=torch.int32, device=G.device)
        remaining[weak] = need.to(torch.int32)
        # Scan the whole candidate list, not just the shortfall. A node short on
        # in-edges is usually already linked from its very nearest neighbours --
        # that is why its in-degree is low in the first place, not high -- so the
        # first `need` candidates are the ones least likely to have room.
        # Measured: offering only that prefix placed 5,749 edges and then stalled
        # with 6,129 pairs already linked.
        for i in range(cand.shape[1]):
            sel = torch.nonzero(remaining > 0).squeeze(1)
            if sel.numel() == 0:
                break
            tgt = cand[sel, i].long()
            ok = (tgt >= 0) & (tgt != sel)
            sel, tgt = sel[ok], tgt[ok]
            if sel.numel() == 0:
                continue
            already = (G[tgt] == sel.to(torch.int32).view(-1, 1)).any(1)
            skipped += int(already.sum())
            sel, tgt = sel[~already], tgt[~already]
            if sel.numel() == 0:
                continue
            order = torch.argsort(tgt)                   # one writer per target
            tgt, sel = tgt[order], sel[order]
            keep = torch.ones_like(tgt, dtype=torch.bool)
            keep[1:] = tgt[1:] != tgt[:-1]
            tgt, sel = tgt[keep], sel[keep]
            rows = G[tgt]                                # distance-sorted
            head = rows.long().clamp(min=0)
            rich = (rows >= 0) & (indeg[head] > min_in)  # safe to take from
            idx = torch.arange(W, device=G.device).view(1, -1).expand_as(rich)
            slot = torch.where(rich, idx, torch.full_like(idx, -1)).max(dim=1).values
            have = slot >= 1                             # never evict a node's own best
            tgt, sel, slot = tgt[have], sel[have], slot[have]
            if tgt.numel() == 0:
                continue
            victim = G[tgt, slot].long()
            indeg[victim] -= 1
            indeg[sel] += 1
            G[tgt, slot] = sel.to(torch.int32)
            remaining[sel] -= 1
            placed += int(tgt.numel())
        print(f"    placed {placed:,}  already-linked skips {skipped:,}  "
              f"still short {int((remaining > 0).sum()):,}")
    return G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m-100k"))
    ap.add_argument("--D", type=int, default=64, help="target graph out-degree")
    ap.add_argument("--cand", type=int, default=128)
    ap.add_argument("--fill", default=None, choices=["near", "far", "bridge"],
                    help="spend the degree slots the prune left empty on the "
                         "candidates it refused, chosen by this rule")
    ap.add_argument("--cand-graph", default=None,
                    help="take candidates from this graph inside --data instead of "
                         "the exact O(N^2) sweep (e.g. G_cagra_d96.npy)")
    ap.add_argument("--pool-boost", type=int, default=0,
                    help="extra out-degree for the entry-pool nodes: every branch "
                         "starts at one, so their reach is the scarce resource")
    ap.add_argument("--pool-name", default="random4k")
    ap.add_argument("--mutual", type=int, default=0,
                    help="protect pairs that are mutual within this rank from the "
                         "prune's detour test; 0 disables it")
    ap.add_argument("--min-in", type=int, default=0,
                    help="floor on a node's in-degree; 0 disables the repair")
    ap.add_argument("--min-in-rounds", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.2)
    ap.add_argument("--metric", default=None, choices=["ip", "l2"],
                    help="default: read it from the corpus manifest")
    ap.add_argument("--out", default="G_ours.npy")
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = a.metric or ("ip" if man["metric"].startswith("ip") else "l2")
    X = np.load(a.data / "X.npy")
    Xg = torch.from_numpy(X).cuda()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    if a.cand_graph:
        Gc = torch.from_numpy(np.load(a.data / a.cand_graph)).cuda()
        cand, cdist = candidates_from_graph(Xg, Gc, a.cand, metric=metric)
        del Gc
        print(f"candidates from {a.cand_graph}: {cand.shape[1]} per node")
    else:
        cand, cdist = knn_candidates(Xg, a.cand, metric=metric)
    torch.cuda.synchronize(); t1 = time.perf_counter()
    dpn = None
    if a.pool_boost:
        pool = np.load(a.data / "pools" / f"pool_{a.pool_name}.npy")
        dpn = torch.full((Xg.shape[0],), a.D, dtype=torch.int32, device=Xg.device)
        pi = torch.from_numpy(pool.astype(np.int64)).cuda()
        dpn[pi] = min(a.D + a.pool_boost, min(a.cand, 128))
        extra = 100.0 * len(pool) * a.pool_boost / (Xg.shape[0] * a.D)
        print(f"pool boost: {len(pool)} entry nodes at degree {int(dpn[pi][0])}, "
              f"the rest at {a.D}  (+{extra:.2f}% edges)")

    prot = None
    if a.mutual:
        prot = mutual_rank(cand, a.mutual)
        print(f"mutual within rank {a.mutual}: {int(prot.sum()):,} protected pairs "
              f"({100.0 * prot.float().mean():.2f}% of the candidate matrix)")
    G = robust_prune(Xg, cand, cdist, a.D, alpha=a.alpha, deg_per_node=dpn,
                     metric=metric, protect=prot)
    torch.cuda.synchronize(); t2 = time.perf_counter()
    G = add_reverse(Xg, G, a.D, alpha=a.alpha, deg_per_node=dpn, metric=metric)
    if a.min_in:
        G = enforce_min_in(Xg, G, cand, a.min_in, metric=metric, rounds=a.min_in_rounds)
    if a.fill:
        before = int((G >= 0).sum())
        G = fill_degree(G, cand, rule=a.fill)
        after = int((G >= 0).sum())
        print(f"fill '{a.fill}': {before:,} -> {after:,} edges "
              f"(+{100.0 * (after - before) / max(before, 1):.1f}%), "
              f"degree now {100.0 * after / (G.shape[0] * G.shape[1]):.0f}% full")
    torch.cuda.synchronize(); t3 = time.perf_counter()

    Gc = G.cpu().numpy().astype(np.int32)
    filled = (Gc >= 0).sum(1)
    print(f"candidates {t1-t0:.2f}s  prune {t2-t1:.2f}s  reverse {t3-t2:.2f}s  "
          f"total {t3-t0:.2f}s")
    print(f"out-degree: min/med/max {filled.min()}/{int(np.median(filled))}/{filled.max()}")
    np.save(a.data / a.out, Gc)
    np.save(a.data / (Path(a.out).stem + "_deg.npy"), filled.astype(np.int32))
    (a.data / (Path(a.out).stem + ".json")).write_text(json.dumps(
        dict(deg=a.D, cand=a.cand, candidates=("graph" if a.cand_graph else "exact"), alpha=a.alpha,
             metric=metric, candidate_graph=a.cand_graph,
             pool_boost=a.pool_boost, min_in=a.min_in, mutual=a.mutual,
             fill=a.fill,
             seconds=round(t3 - t0, 2),
             stage_seconds=dict(candidates=round(t1 - t0, 2), prune=round(t2 - t1, 2),
                                reverse=round(t3 - t2, 2))), indent=2))
    print(f"wrote {a.data / a.out}")


if __name__ == "__main__":
    main()
