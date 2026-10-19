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

POLICY_BEST = 0          # expand the nearest unexpanded candidate, always
POLICY_ROUND_ROBIN = 1   # give each seed lineage its turn

METRIC_IP = 0   # distance = 1 - dot. Vectors must be L2-normalised.
METRIC_L2 = 1   # distance = ||q - x||^2
NCTR = 6        # expansions, distance evals, hash failures, exhausted,
                # cache hits, cache slots claimed

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
    lineage: torch.Tensor    # uint8  [Q, J, m]  which seed this entry descends from
    visited: torch.Tensor    # int32  [Q, J, H]  per branch: keeps its queue unique
    counters: torch.Tensor   # int32  [Q, J, 6]
    max_expansions: int
    metric: int
    team_size: int
    n_seeds: int = 1         # seeds per branch; lineage ids run 0..n_seeds-1
    policy: int = POLICY_BEST
    stop_eps: float = -1.0   # < 0 disables early stopping (run the queue out)
    kstop: int = 10          # the k whose current best distance sets the radius
    # One distance table per *query*, shared by all its branches. Empty tensors
    # disable it. It changes what is computed, never what is returned.
    cache_id: torch.Tensor = None
    cache_d: torch.Tensor = None
    # The query's best k-th distance so far, across all of its branches. Sharing
    # it couples the CTAs, so results stop being reproducible bit for bit: which
    # branch retires depends on which CTA got there first. Off by default.
    bound: torch.Tensor = None
    # Exclusive: a node claimed by one branch is off limits to the others, which
    # forces their trajectories apart. Needs the shared cache to be on.
    exclusive: bool = False

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

    @property
    def cache_hits(self) -> torch.Tensor:     # [Q, J] distances another branch had done
        return self.counters[..., 4]

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
                           self.lineage.clone(), self.visited.clone(), self.counters.clone(),
                           self.max_expansions, self.metric, self.team_size,
                           self.n_seeds, self.policy, self.stop_eps, self.kstop,
                           None if self.cache_id is None else self.cache_id.clone(),
                           None if self.cache_d is None else self.cache_d.clone(),
                           None if self.bound is None else self.bound.clone(),
                           self.exclusive)


def cache_size(J, max_expansions, deg, share=4.0, cap=1 << 20):
    """Slots for the shared table.

    Sized for what the branches actually touch together, not for the worst case
    they could: measured on this corpus, 100 branches of 16 expansions reach
    about an eighth of the nodes their budgets allow, because that overlap is the
    entire point of sharing. `share` is the assumed overlap factor. Getting it
    wrong costs nothing but recomputation -- a bounded probe that finds no room
    just computes the distance -- while sizing for the worst case means clearing
    a table eight times larger than anything will be written into, every search.
    """
    want = 2 * (1 + J * (1 + max_expansions * deg) / share)
    return min(next_pow2(int(want)), cap)


def init_states(X, Qv, seeds, m=16, max_expansions=16, deg=None,
                metric=METRIC_IP, team_size=32, policy=POLICY_BEST,
                stop_eps=-1.0, kstop=10, shared_cache=False,
                shared_bound=False, exclusive=False) -> SearchState:
    """One state per (query, branch).

    `seeds` is int32 [Q, J] for one seed per branch, or [Q, J, S] for S seeds
    sharing one branch's queue, visited table and budget. -1 is an empty slot.
    The S form is the one that matters: several entries in one state cost barely
    more than one, where several branches cost several times as much.
    """
    assert X.is_cuda and X.dtype in (torch.float32, torch.float16) and X.is_contiguous()
    assert Qv.is_cuda and Qv.dtype == torch.float32 and Qv.is_contiguous()
    seeds = seeds.to(device=X.device, dtype=torch.int32).contiguous()
    if seeds.dim() == 2:
        seeds = seeds.unsqueeze(-1)
    Q, J, S = seeds.shape
    if deg is None:
        raise ValueError("deg is required: it sizes the visited table")
    H = visited_size(max_expansions, deg)
    dev = X.device
    st = SearchState(
        ids=torch.full((Q, J, m), -1, dtype=torch.int32, device=dev),
        dist=torch.full((Q, J, m), float("inf"), dtype=torch.float32, device=dev),
        expanded=torch.ones((Q, J, m), dtype=torch.uint8, device=dev),
        lineage=torch.zeros((Q, J, m), dtype=torch.uint8, device=dev),
        visited=torch.zeros((Q, J, H), dtype=torch.int32, device=dev),
        counters=torch.zeros((Q, J, NCTR), dtype=torch.int32, device=dev),
        max_expansions=max_expansions, metric=metric, team_size=team_size,
        n_seeds=S, policy=policy, stop_eps=float(stop_eps), kstop=int(kstop),
    )
    if shared_cache and J > 1:
        Hc = cache_size(J, max_expansions, deg)
        st.cache_id = torch.zeros((Q, Hc), dtype=torch.int32, device=dev)
        # FLT_MAX, not inf: the kernel's "nothing cached here" sentinel is
        # FLT_MAX, and inf would read back as a perfectly good distance.
        st.cache_d = torch.full((Q, Hc), torch.finfo(torch.float32).max,
                                dtype=torch.float32, device=dev)
    else:
        st.cache_id = torch.empty((0, 0), dtype=torch.int32, device=dev)
        st.cache_d = torch.empty((0, 0), dtype=torch.float32, device=dev)
    st.exclusive = bool(exclusive and shared_cache and J > 1)
    st.bound = (torch.full((Q,), torch.finfo(torch.float32).max,
                           dtype=torch.float32, device=dev)
                if (shared_bound and J > 1)
                else torch.empty((0,), dtype=torch.float32, device=dev))
    ext().init_states(X, Qv, seeds, st.ids, st.dist, st.expanded, st.lineage,
                      st.visited, st.counters, metric, team_size)
    return st


def advance(X, G, Qv, state: SearchState, delta_expansions: int) -> SearchState:
    """Advance every branch by up to delta more expansions, in place."""
    assert G.is_cuda and G.dtype == torch.int32 and G.is_contiguous()
    ext().advance(X, G, Qv, state.ids, state.dist, state.expanded, state.lineage,
                  state.visited, state.counters, state.metric, state.team_size,
                  int(delta_expansions), int(state.max_expansions),
                  int(state.policy), int(state.n_seeds),
                  float(state.stop_eps), int(state.kstop),
                  state.cache_id, state.cache_d, state.bound, int(state.exclusive))
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


def search(X, G, Qv, seeds, k=10, m=16, budget=16, metric=METRIC_IP, team_size=32,
           policy=POLICY_BEST, stop_eps=-1.0, shared_cache=False,
           shared_bound=False, exclusive=False):
    """init -> advance(budget) -> per-branch best-k -> merged top-k. One shot."""
    st = init_states(X, Qv, seeds, m=m, max_expansions=budget, deg=G.shape[1],
                     metric=metric, team_size=team_size, policy=policy,
                     stop_eps=stop_eps, kstop=k, shared_cache=shared_cache,
                     shared_bound=shared_bound, exclusive=exclusive)
    advance(X, G, Qv, st, budget)
    st.check()
    bi, bd = branch_topk(st, k)
    ids, dist = merge_branches(bi, bd, k)
    return ids, dist, st
