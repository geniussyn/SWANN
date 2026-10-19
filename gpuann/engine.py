"""The common search engine: explicit entries, a resumable state, exact distances.

Three designs (reach sketch, probe-then-select, GPU batched selection) are meant
to differ in *which* entries they search from and in nothing else. That only
holds if the search itself is one piece of code with one set of semantics, so
the semantics are fixed here and the design variants are not allowed to restate
them:

  queue      best-m over everything evaluated so far, sorted by (distance, id),
             with the `expanded` flag carried along by every merge.
  stop       budget spent, or no retained candidate left unexpanded. Never "the
             queue did not change this round".
  budget     node expansions, counted per branch. Distance evaluations are a
             separate counter and are never conflated with it.
  resume     advance(a) then advance(b) == advance(a+b). Nothing is recomputed
             and nothing is thrown away between calls.

The harness is torch rather than CuPy (the design document's wording): torch is
what this box has, it owns the device arrays and the C++/CUDA extension
toolchain, and the kernel reads raw contiguous pointers either way.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import torch

METRIC_IP = 0   # distance = 1 - dot. Vectors must be L2-normalised.
METRIC_L2 = 1   # distance = ||q - x||^2
NCTR = 4        # expansions, distance evaluations, hash failures, exhausted

_EXT = None


def ext():
    """Build (once, cached) and return the CUDA extension."""
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        # Blackwell RTX PRO 6000 is sm_120; naming it keeps the build off the
        # "compile for every arch torch knows" path.
        os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
        src = Path(__file__).parent / "csrc" / "local_search.cu"
        _EXT = load(
            name="gpuann_local_search",
            sources=[str(src)],
            # -fmad=false: without it nvcc contracts q*x+acc into an FMA, whose
            # unrounded product the CPU reference cannot reproduce, and the
            # design's own acceptance test is "the CPU local search agrees".
            # The distance loop reads D floats per distance and does D flops, so
            # it is bound by memory, not by the multiply-add: the fix is free.
            extra_cuda_cflags=["-O3", "-lineinfo", "-fmad=false", "--expt-relaxed-constexpr"],
            extra_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


def next_pow2(x: int) -> int:
    n = 1
    while n < x:
        n <<= 1
    return n


def visited_size(max_expansions: int, deg: int) -> int:
    """Load factor <= 0.5 at the worst case of every expansion inserting deg new
    nodes, plus the seed. Overflow is a reported failure, not a silent miss, so
    this bound is a correctness argument and not just a performance one."""
    return next_pow2(2 * (1 + max_expansions * deg))


@dataclass
class SearchState:
    """Everything a branch needs to be stopped and resumed, and nothing else."""
    ids: torch.Tensor        # int32  [Q, J, m]  -1 = empty slot
    dist: torch.Tensor       # float32[Q, J, m]
    expanded: torch.Tensor   # uint8  [Q, J, m]
    visited: torch.Tensor    # int32  [Q, J, H]  open addressing, slot = id+1
    counters: torch.Tensor   # int32  [Q, J, 4]
    max_expansions: int
    metric: int
    team_size: int

    @property
    def shape(self):
        return tuple(self.ids.shape)          # (Q, J, m)

    @property
    def expansions(self) -> torch.Tensor:     # [Q, J]
        return self.counters[..., 0]

    @property
    def dist_evals(self) -> torch.Tensor:     # [Q, J]
        return self.counters[..., 1]

    @property
    def hash_failures(self) -> torch.Tensor:  # [Q, J]
        return self.counters[..., 2]

    @property
    def exhausted(self) -> torch.Tensor:      # [Q, J] stopped for lack of candidates
        return self.counters[..., 3]

    def check(self):
        """A full visited table means the search dropped nodes. Never continue past it."""
        nf = int(self.hash_failures.sum())
        if nf:
            raise RuntimeError(
                f"visited table overflowed on {int((self.hash_failures > 0).sum())} branches "
                f"({nf} failed inserts): raise max_expansions used for sizing, or the table size"
            )

    def clone(self) -> "SearchState":
        return SearchState(self.ids.clone(), self.dist.clone(), self.expanded.clone(),
                           self.visited.clone(), self.counters.clone(),
                           self.max_expansions, self.metric, self.team_size)


def init_states(X, Qv, seeds, m=16, max_expansions=16, deg=None,
                metric=METRIC_IP, team_size=32) -> SearchState:
    """One state per (query, branch). `seeds` is int32 [Q, J]; -1 leaves a branch empty."""
    assert X.is_cuda and X.dtype == torch.float32 and X.is_contiguous()
    assert Qv.is_cuda and Qv.dtype == torch.float32 and Qv.is_contiguous()
    seeds = seeds.to(device=X.device, dtype=torch.int32).contiguous()
    Q, J = seeds.shape
    if deg is None:
        raise ValueError("deg is required: it sizes the visited table")
    H = visited_size(max_expansions, deg)
    dev = X.device
    st = SearchState(
        ids=torch.full((Q, J, m), -1, dtype=torch.int32, device=dev),
        dist=torch.full((Q, J, m), float("inf"), dtype=torch.float32, device=dev),
        expanded=torch.ones((Q, J, m), dtype=torch.uint8, device=dev),
        visited=torch.zeros((Q, J, H), dtype=torch.int32, device=dev),
        counters=torch.zeros((Q, J, NCTR), dtype=torch.int32, device=dev),
        max_expansions=max_expansions, metric=metric, team_size=team_size,
    )
    ext().init_states(X, Qv, seeds, st.ids, st.dist, st.expanded, st.visited,
                      st.counters, metric, team_size)
    return st


def advance(X, G, Qv, state: SearchState, delta_expansions: int) -> SearchState:
    """Advance every branch by up to delta more expansions, in place."""
    assert G.is_cuda and G.dtype == torch.int32 and G.is_contiguous()
    ext().advance(X, G, Qv, state.ids, state.dist, state.expanded, state.visited,
                  state.counters, state.metric, state.team_size,
                  int(delta_expansions), int(state.max_expansions))
    return state


def branch_topk(state: SearchState, k: int):
    """Per-branch best-k, straight off the queue (it is already sorted)."""
    assert k <= state.ids.shape[-1], "k must not exceed the queue width m"
    return state.ids[..., :k].clone(), state.dist[..., :k].clone()


def merge_branches(ids: torch.Tensor, dist: torch.Tensor, k: int):
    """Branch results -> one top-k per query, duplicates removed by id.

    Every branch contributes its best-k and every contributed id keeps its own
    exact distance, so the merge is a pure selection: no branch result is
    discarded before the dedup, including a probe branch that lost its selection.
    """
    Q = ids.shape[0]
    flat_i = ids.reshape(Q, -1).to(torch.int64)
    flat_d = dist.reshape(Q, -1).clone()
    flat_d[flat_i < 0] = float("inf")

    # (distance, id) order, then drop the second and later copies of an id.
    order = torch.argsort(flat_i, dim=1, stable=True)
    sd = torch.gather(flat_d, 1, order)
    si = torch.gather(flat_i, 1, order)
    dup = torch.zeros_like(si, dtype=torch.bool)
    dup[:, 1:] = (si[:, 1:] == si[:, :-1]) & (si[:, 1:] >= 0)
    sd = sd.masked_fill(dup, float("inf"))
    si = si.masked_fill(dup, -1)

    kk = min(k, sd.shape[1])
    best_d, pos = torch.topk(sd, kk, dim=1, largest=False, sorted=True)
    best_i = torch.gather(si, 1, pos)
    best_i = best_i.masked_fill(torch.isinf(best_d), -1)
    return best_i.to(torch.int32), best_d


def search(X, G, Qv, seeds, k=10, m=16, budget=16, metric=METRIC_IP, team_size=32):
    """init -> advance(budget) -> per-branch best-k -> merged top-k. One shot."""
    st = init_states(X, Qv, seeds, m=m, max_expansions=budget, deg=G.shape[1],
                     metric=metric, team_size=team_size)
    advance(X, G, Qv, st, budget)
    st.check()
    bi, bd = branch_topk(st, k)
    ids, dist = merge_branches(bi, bd, k)
    return ids, dist, st
