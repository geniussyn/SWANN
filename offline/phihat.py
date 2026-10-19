"""phi-hat_1, index-free, and the degree-allocation signal that falls out of it.

    rank_u(v) = |{w : sim(u,w) > sim(u,v)}|      1-based, excluding self
    W[i,j]    = min(rank_i(j), rank_j(i))        mutual rank
    adj       = W <= d                           "a builder with out-degree d
                                                  would have joined this pair"
    phi1_hat  = components(adj)

No index is built: one corpus scan answers every threshold at once. That is what
makes it the right quantity for a *builder* rather than a diagnostic -- the
threshold in the definition is literally the out-degree budget, so "which pairs
stay cut" and "where the degree went" are the same axis. Raising node u's degree
from d to d' joins exactly the pairs with rank_u(.) in (d, d'].

Target sets are the corpus's own neighbourhoods by default -- each document's
top-k standing in for an answer set -- so nothing here sees a query and the
signal exists before any query does. Passing --queries uses a held-out-free
query file instead, as the manuscript's screen does.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


@torch.no_grad()
def mutual_ranks(Xg, targets, anchor_chunk=1024, doc_chunk=50_000):
    """W[q, i, j] = min(rank_i(j), rank_j(i)) for every target set.

    One pass over the corpus per anchor chunk, and the anchors are the distinct
    documents appearing in any target set, so the whole thing costs one N x N
    sweep -- the same work as a brute-force kNN, not one scan per query.
    """
    nq, k = targets.shape
    dev = Xg.device
    flat = targets.reshape(-1)
    anchors, inv = torch.unique(flat, return_inverse=True)
    inv = inv.reshape(nq, k)
    na = anchors.numel()

    # Partner list per anchor: every co-member of every set it belongs to.
    part = [set() for _ in range(na)]
    inv_c = inv.cpu().numpy()
    for qi in range(nq):
        row = inv_c[qi]
        for a in row:
            part[int(a)].update(int(b) for b in row if b != a)
    P = max(1, max(len(s) for s in part))
    pad = np.zeros((na, P), dtype=np.int64)
    slot = [dict() for _ in range(na)]
    for i, ss in enumerate(part):
        srt = sorted(ss)
        for j, v in enumerate(srt):
            pad[i, j] = v; slot[i][v] = j
        if len(srt) < P:
            pad[i, len(srt):] = srt[0] if srt else 0
    anchors_cpu = anchors.cpu().numpy()
    padded_ids = torch.from_numpy(anchors_cpu[pad]).to(dev)           # corpus ids
    ranks = torch.zeros((na, P), dtype=torch.int32, device=dev)
    n = Xg.shape[0]

    for s in range(0, na, anchor_chunk):
        e = min(s + anchor_chunk, na)
        A = Xg[anchors[s:e]]                                          # [B, D]
        Pv = Xg[padded_ids[s:e].reshape(-1)].reshape(e - s, P, -1)    # [B, P, D]
        thr = torch.bmm(A.unsqueeze(1), Pv.transpose(1, 2)).squeeze(1)  # [B, P]
        thr_sorted, order = torch.sort(thr, dim=1)
        cnt = torch.zeros((e - s, P + 1), dtype=torch.int32, device=dev)
        for cs in range(0, n, doc_chunk):
            sim = A @ Xg[cs:min(cs + doc_chunk, n)].T
            pos = torch.searchsorted(thr_sorted.contiguous(), sim.contiguous(), right=False)
            cnt.scatter_add_(1, pos, torch.ones_like(pos, dtype=torch.int32))
            del sim, pos
        # how many sims are strictly greater than each threshold
        above = torch.flip(torch.cumsum(torch.flip(cnt, [1]).to(torch.int64), 1), [1])[:, 1:]
        r = torch.empty_like(above)
        r.scatter_(1, order, above)
        ranks[s:e] = r.to(torch.int32)

    # Gather W from the per-anchor rank tables.
    slot_idx = torch.full((na, P), -1, dtype=torch.int64)
    for i, sl in enumerate(slot):
        for v, j in sl.items():
            slot_idx[i, j] = v
    lookup = {}
    for i, sl in enumerate(slot):
        lookup[i] = sl
    W = torch.zeros((nq, k, k), dtype=torch.int32, device=dev)
    rk = ranks.cpu().numpy()
    Wc = np.zeros((nq, k, k), dtype=np.int32)
    for qi in range(nq):
        row = inv_c[qi]
        for a in range(k):
            ia = int(row[a])
            for b in range(k):
                if a == b:
                    continue
                Wc[qi, a, b] = rk[ia, lookup[ia][int(row[b])]]
    Wc = np.minimum(Wc, Wc.transpose(0, 2, 1))
    return torch.from_numpy(Wc).to(dev)


def components_from_adj(adj):
    """Component labels of a batch of k x k boolean adjacencies, by propagation."""
    nq, k, _ = adj.shape
    a = adj | torch.eye(k, dtype=torch.bool, device=adj.device).view(1, k, k)
    lab = torch.arange(k, device=adj.device).view(1, k).expand(nq, k).contiguous()
    for _ in range(k):
        cand = torch.where(a, lab.view(nq, 1, k), torch.full_like(lab.view(nq, 1, k), k))
        new = cand.min(dim=2).values
        if torch.equal(new, lab):
            break
        lab = new
    phi = torch.stack([(lab == c).any(1) for c in range(k)], 1).sum(1)
    return phi.to(torch.int32), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--degrees", type=int, nargs="+", default=[16, 32, 48, 64, 96, 128, 192, 256])
    ap.add_argument("--nq", type=int, default=20000, help="pseudo-queries (corpus rows) to use")
    ap.add_argument("--queries", default=None, help="use a query split instead of the corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    X = np.load(a.data / "X.npy")
    Xg = torch.from_numpy(X).cuda()
    if a.queries:
        Q = torch.from_numpy(np.load(a.data / f"Q_{a.queries}.npy")).cuda()
        gt = np.load(a.data / f"gt_{a.queries}_ids.npy")
        targets = torch.from_numpy(gt[:, : a.k].astype(np.int64)).cuda()
        rows = None
        label = f"queries:{a.queries}"
    else:
        rng = np.random.default_rng(a.seed)
        rows = np.sort(rng.choice(X.shape[0], min(a.nq, X.shape[0]), replace=False))
        rt = torch.from_numpy(rows).cuda()
        sims = torch.empty((len(rows), a.k + 1), device="cuda")
        idxs = torch.empty((len(rows), a.k + 1), dtype=torch.int64, device="cuda")
        for s in range(0, len(rows), 4096):
            e = min(s + 4096, len(rows))
            sim = Xg[rt[s:e]] @ Xg.T
            v, i = torch.topk(sim, a.k + 1, dim=1)
            sims[s:e], idxs[s:e] = v, i
            del sim
        targets = idxs[:, : a.k].contiguous()          # the document itself is rank 0
        label = f"corpus pseudo-queries (n={len(rows)})"

    t = time.perf_counter()
    W = mutual_ranks(Xg, targets)
    secs = time.perf_counter() - t
    print(f"{label}, k={a.k}: mutual ranks in {secs:.1f}s "
          f"(one corpus sweep, no index)\n")

    off = ~torch.eye(a.k, dtype=torch.bool, device=W.device).view(1, a.k, a.k)
    med = W[off.expand_as(W)].float().median().item()
    print(f"{'out-degree d':>12s} {'phi1_hat':>9s} {'Pr[phi=1]':>10s} {'pairs joined':>13s}")
    rows_out = []
    for d in a.degrees:
        adj = (W <= d) & off
        phi, _ = components_from_adj(adj)
        pj = float((W[off.expand_as(W)] <= d).float().mean())
        print(f"{d:12d} {phi.float().mean().item():9.3f} "
              f"{float((phi == 1).float().mean()):10.3f} {pj:13.3f}")
        rows_out.append(dict(degree=int(d), phi1_hat=float(phi.float().mean()),
                             pr_phi1=float((phi == 1).float().mean()), pairs_joined=pj))
    print(f"\nmedian mutual rank over target pairs: {med:.0f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.out.with_suffix(".W.npy"), W.cpu().numpy().astype(np.int32))
        if rows is not None:
            np.save(a.out.with_suffix(".rows.npy"), rows)
        np.save(a.out.with_suffix(".targets.npy"), targets.cpu().numpy().astype(np.int32))
        a.out.write_text(json.dumps(dict(k=a.k, source=label, seconds=round(secs, 1),
                                         median_mutual_rank=med, rows=rows_out), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
