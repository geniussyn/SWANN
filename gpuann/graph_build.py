"""Difficulty-aware graph construction: spend degree where the graph fragments.

The manuscript's varphi_h counts how many pieces a *query's* exact top-k is cut
into, and its appendix shows that reconnecting those pieces -- closest
cross-component pair, reciprocal edges, out-degree preserved by dropping each
endpoint's longest edge -- raises recall at a fixed budget. That is a
post-hoc surgery and it needs queries.

Here the same quantity is asked of the corpus instead: a document's own top-k
neighbourhood stands in for an answer set, and a node whose neighbourhood the
graph has already cut into pieces is a node sitting where the graph is thin. No
query is involved, so the signal is available at build time and cannot overfit a
query distribution.
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

_EXT = None


def ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        src = Path(__file__).parent / "csrc" / "graph_build.cu"
        _EXT = load(name="gpuann_graph_build", sources=[str(src)],
                    extra_cuda_cflags=["-O3", "-lineinfo"], extra_cflags=["-O3"],
                    verbose=False)
    return _EXT


def neighborhood_components(G, members, h=2):
    """(phi[N], labels[N, K]) for every node's own neighbourhood.

    Two members are joined when each reaches the other within h hops -- mutual
    reachability, paths through non-members allowed, exactly the manuscript's
    definition. Components are found by label propagation, which converges in at
    most K rounds for K members and costs nothing at this size.
    """
    reach = ext().neighborhood_reach(G.contiguous(), members.contiguous(), int(h))
    N, K = members.shape
    bit = (1 << torch.arange(K, device=reach.device, dtype=torch.int32)).view(1, 1, K)
    # reached[u, i, j] = member i reaches member j
    reached = ((reach.view(N, 1, K) & bit.transpose(1, 2)) != 0)
    conn = reached & reached.transpose(1, 2)                     # mutual
    conn = conn | torch.eye(K, dtype=torch.bool, device=conn.device).view(1, K, K)
    conn &= (members >= 0).view(N, 1, K) & (members >= 0).view(N, K, 1)

    labels = torch.arange(K, device=conn.device).view(1, K).expand(N, K).contiguous()
    for _ in range(K):
        cand = torch.where(conn, labels.view(N, 1, K), torch.full_like(labels.view(N, 1, K), K))
        new = cand.min(dim=2).values
        if torch.equal(new, labels):
            break
        labels = new
    valid = members >= 0
    labels = torch.where(valid, labels, torch.full_like(labels, -1))
    phi = torch.stack([(labels == c).any(dim=1) for c in range(K)], dim=1).sum(dim=1)
    return phi.to(torch.int32), labels.to(torch.int8)
