"""The common search engine: explicit entries, a resumable state, exact distances.

Three designs (reach sketch, probe-then-select, GPU batched selection) are meant
to differ in *which* entries they search from and in nothing else. That only
holds if the search itself is one piece of code with one set of semantics, so
the semantics are fixed here and the design variants are not allowed to restate
them:

  queue      best-L over everything evaluated so far, sorted by (distance, id),
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

import numpy as np
import torch

POLICY_BEST = 0          # expand the nearest unexpanded candidate, always
POLICY_ROUND_ROBIN = 1   # give each seed lineage its turn

METRIC_IP = 0   # distance = 1 - dot. Vectors must be L2-normalised.
METRIC_L2 = 1   # distance = ||q - x||^2
NCTR = 10        # expansions, distance evals, hash failures, exhausted,
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
            name="swann_local_search",
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


def visited_size(B: int, D: int) -> int:
    """Load factor <= 0.5 at the worst case of every expansion inserting deg new
    nodes, plus the seed. Overflow is a reported failure, not a silent miss, so
    this bound is a correctness argument and not just a performance one."""
    return next_pow2(2 * (1 + B * D))


@dataclass
class SearchState:
    """Everything a branch needs to be stopped and resumed, and nothing else."""
    ids: torch.Tensor        # int32  [Q, J, L]  -1 = empty slot
    dist: torch.Tensor       # float32[Q, J, L]
    expanded: torch.Tensor   # uint8  [Q, J, L]
    lineage: torch.Tensor    # uint8  [Q, J, L]  which seed this entry descends from
    visited: torch.Tensor    # int32  [Q, J, H]  per branch: keeps its queue unique
    counters: torch.Tensor   # int32  [Q, J, 6]
    B: int                   # expansion budget per branch
    metric: int
    team_size: int
    V: int = 1               # entries (seeds) per branch; lineage ids run 0..V-1
    policy: int = POLICY_BEST
    stop_eps: float = -1.0   # < 0 disables early stopping (run the queue out)
    # Steps a branch may go without improving its k-th best before it gives up.
    # 0 disables it. Independent of stop_eps: that one tests distance, this one
    # tests progress, and a hard query can fail the second while passing the first.
    stall_patience: int = 0
    # Retire a lineage whose expansion lands on ground already covered -- its own
    # visited set, or nodes another branch has claimed. 0 disables it. This reads
    # the cache rather than the distances, so it fires on branches that are
    # retreading even while their frontier still looks close.
    crowd_frac: float = 0.0
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
        return tuple(self.ids.shape)          # (Q, J, L)

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
    def stalled(self) -> torch.Tensor:        # [Q, J] stopped because k-th stopped moving
        return self.counters[..., 6]

    @property
    def crowded(self) -> torch.Tensor:        # [Q, J] lineages retired for retreading
        return self.counters[..., 7]

    @property
    def fanout(self):
        """Mean candidates per expansion that beat the queue's worst -- the part
        of the out-degree that was ever in play. Returns (sum, expansions)."""
        return self.counters[..., 8], self.counters[..., 9]

    @property
    def cache_hits(self) -> torch.Tensor:     # [Q, J] distances another branch had done
        return self.counters[..., 4]

    def check(self):
        """Raise if the visited table overflowed, which means nodes were dropped.

        This reads a counter back to the host, so it synchronises: at a batch of
        one that is a tenth of the whole search, and inside a CUDA graph capture
        it is illegal outright. `search` therefore defers it -- see
        `overflow_guard` -- and this method remains for the caller who wants the
        answer now.
        """
        nf = int(self.hash_failures.sum())
        if nf:
            raise RuntimeError(
                f"visited table overflowed on {int((self.hash_failures > 0).sum())} branches "
                f"({nf} failed inserts): raise max_expansions used for sizing, or the table size"
            )

    def overflow(self) -> torch.Tensor:
        """The overflow count, left on the device. Free, and asynchronous."""
        return self.hash_failures.sum()

    def clone(self) -> "SearchState":
        return SearchState(self.ids.clone(), self.dist.clone(), self.expanded.clone(),
                           self.lineage.clone(), self.visited.clone(), self.counters.clone(),
                           self.B, self.metric, self.team_size,
                           self.V, self.policy, self.stop_eps, self.kstop,
                           None if self.cache_id is None else self.cache_id.clone(),
                           None if self.cache_d is None else self.cache_d.clone(),
                           None if self.bound is None else self.bound.clone(),
                           self.exclusive)


def cache_size(J, B, D, share=4.0, cap=1 << 20):
    """Slots for the shared table.

    Sized for what the branches actually touch together, not for the worst case
    they could: measured on this corpus, 100 branches of 16 expansions reach
    about an eighth of the nodes their budgets allow, because that overlap is the
    entire point of sharing. `share` is the assumed overlap factor. Getting it
    wrong costs nothing but recomputation -- a bounded probe that finds no room
    just computes the distance -- while sizing for the worst case means clearing
    a table eight times larger than anything will be written into, every search.
    """
    want = 2 * (1 + J * (1 + B * D) / share)
    return min(next_pow2(int(want)), cap)




# Whether `search` verifies, on the host and therefore synchronously, that no
# branch overflowed its visited table. The check is cheap in work and expensive
# in latency: it reads one integer back, which drains the pipeline and costs a
# tenth of a single-query search. On by default because a silent overflow
# silently loses results; turn it off for a serving loop that has already
# validated its configuration, and read `state.overflow()` when convenient.
_OVERFLOW_GUARD = True


def overflow_guard(on: bool) -> bool:
    """Turn the per-search overflow check on or off. Returns the old setting."""
    global _OVERFLOW_GUARD
    was, _OVERFLOW_GUARD = _OVERFLOW_GUARD, bool(on)
    return was


def init_states(X, Qv, seeds, L=16, B=16, D=None,
                metric=METRIC_IP, team_size=32, policy=POLICY_BEST,
                stop_eps=-1.0, kstop=10, shared_cache=False,
                shared_bound=False, exclusive=False, stall_patience=0,
                crowd_frac=0.0, elide_visited=False) -> SearchState:
    """One state per (query, branch).

    `seeds` is int32 [Q, J] for one entry per branch, or [Q, J, V] for V entries
    sharing one branch's queue, visited table and budget. -1 is an empty slot.
    The S form is the one that matters: several entries in one state cost barely
    more than one, where several branches cost several times as much.
    """
    assert X.is_cuda and X.dtype in (torch.float32, torch.float16) and X.is_contiguous()
    assert Qv.is_cuda and Qv.dtype == torch.float32 and Qv.is_contiguous()
    seeds = seeds.to(device=X.device, dtype=torch.int32).contiguous()
    if seeds.dim() == 2:
        seeds = seeds.unsqueeze(-1)
    Q, J, V = seeds.shape
    if D is None:
        raise ValueError("D (the graph out-degree) is required: it sizes the visited table")
    # In exclusive mode the per-query cache already refuses a node anyone has
    # claimed, so the per-branch visited table is redundant: H = 1 switches it
    # off, dropping a global probe per candidate and the clear before every
    # search. Only with one seed per branch -- otherwise init needs it to dedup
    # repeated seeds.
    elide = bool(elide_visited and exclusive and shared_cache and J > 1 and V == 1)
    H = 1 if elide else visited_size(B, D)
    dev = X.device
    # Everything below is written in full by init_kernel before it is read --
    # the queue's L slots, the visited table's H words, the counters -- so
    # filling it here first was six whole-tensor writes thrown away. The cache
    # and the bound are cleared by the same kernel for the same reason.
    st = SearchState(
        ids=torch.empty((Q, J, L), dtype=torch.int32, device=dev),
        dist=torch.empty((Q, J, L), dtype=torch.float32, device=dev),
        expanded=torch.empty((Q, J, L), dtype=torch.uint8, device=dev),
        lineage=torch.empty((Q, J, L), dtype=torch.uint8, device=dev),
        visited=torch.empty((Q, J, H), dtype=torch.int32, device=dev),
        counters=torch.empty((Q, J, NCTR), dtype=torch.int32, device=dev),
        B=B, metric=metric, team_size=team_size,
        V=V, policy=policy, stop_eps=float(stop_eps), kstop=int(kstop),
        stall_patience=int(stall_patience), crowd_frac=float(crowd_frac),
    )
    if shared_cache and J > 1:
        Hc = cache_size(J, B, D)
        # FLT_MAX, not inf, is the kernel's "nothing cached here" sentinel, and
        # the kernel writes it; inf would read back as a perfectly good distance.
        st.cache_id = torch.empty((Q, Hc), dtype=torch.int32, device=dev)
        st.cache_d = torch.empty((Q, Hc), dtype=torch.float32, device=dev)
    else:
        st.cache_id = torch.empty((0, 0), dtype=torch.int32, device=dev)
        st.cache_d = torch.empty((0, 0), dtype=torch.float32, device=dev)
    st.exclusive = bool(exclusive and shared_cache and J > 1)
    st.bound = (torch.empty((Q,), dtype=torch.float32, device=dev)
                if (shared_bound and J > 1)
                else torch.empty((0,), dtype=torch.float32, device=dev))
    ext().init_states(X, Qv, seeds, st.ids, st.dist, st.expanded, st.lineage,
                      st.visited, st.counters, metric, team_size,
                      st.cache_id, st.cache_d, st.bound)
    return st


def advance(X, G, Qv, state: SearchState, delta_expansions: int,
            active=None, codes=None) -> SearchState:
    """Advance every branch by up to delta more expansions, in place.

    `active` is an int32 list of flattened (query, branch) indices; when given,
    the grid covers only those and the rest are left untouched. That is what
    makes chunked advancing worthwhile: branches finish at very different times,
    so after a few expansions most of the rectangle is blocks that will exit on
    their first test.

    `codes` is an optional (codes, alpha, mu, margin) from
    `offline/build_codes.py`. Given it, an expansion estimates each neighbour's
    distance from its sign code -- 1/32 of the bytes -- and reads the full vector
    only when the estimate leaves it able to enter the queue. Measured on MS
    MARCO, 87% of the bytes an expansion moves are spent on candidates it then
    rejects, and the code keeps 100% of the ones the exact test admits. It
    changes which distances are computed, so it is off unless asked for.
    """
    assert G.is_cuda and G.dtype == torch.int32 and G.is_contiguous()
    dev = state.ids.device
    empty_i = torch.empty(0, dtype=torch.int32, device=dev)
    empty_f = torch.empty(0, dtype=torch.float32, device=dev)
    if codes is None:
        cd, al, cm, mg = empty_i, empty_f, empty_f, 0.0
    else:
        cd, al, cm, mg = codes
        assert cd.dtype == torch.int32 and al.dtype == torch.float32
    ext().advance(X, G, Qv, state.ids, state.dist, state.expanded, state.lineage,
                  state.visited, state.counters, state.metric, state.team_size,
                  int(delta_expansions), int(state.B),
                  int(state.policy), int(state.V),
                  float(state.stop_eps), int(state.kstop),
                  state.cache_id, state.cache_d, state.bound, int(state.exclusive),
                  int(state.stall_patience), float(state.crowd_frac),
                  active if active is not None else empty_i,
                  cd, al, cm, float(mg))
    return state


def load_codes(data, name=None, margin=0.20, device="cuda"):
    """The sign code, its per-node scale and the corpus mean, as `advance` wants.

    `margin` is in the corpus's distance units and has to be set per corpus: on
    MS MARCO 0.20 keeps every admitted candidate while letting 27.4% of the
    neighbour slots through, and 0.30 lets 85.6% through and buys nothing.
    """
    import json
    from pathlib import Path
    import numpy as np
    d = Path(data)
    if name is None:
        man = json.loads((d / "manifest.json").read_text())
        name = "1bit" if man["metric"].startswith("ip") else "int8"
    cd = torch.from_numpy(np.load(d / f"codes_{name}.npy")).to(device)
    al = torch.from_numpy(np.load(d / f"alpha_{name}.npy")).to(device)
    cm = torch.from_numpy(np.load(d / f"mu_{name}.npy")).to(device)
    return cd.contiguous(), al.contiguous(), cm.contiguous(), margin


def branch_topk(state: SearchState, k: int):
    """Per-branch best-k, straight off the queue (it is already sorted).

    L < k is allowed and is not a degenerate case: the answer is assembled by
    merging V branches, so a branch holding fewer than k results is only
    contributing its share. At L = 1 each branch is a greedy descent that stops
    at its own local minimum -- the queue runs out of unexpanded entries by
    itself -- and V of them merged is V candidates for the k best.

    Note what L < kstop costs inside the kernel: the stop_eps radius is read from
    the queue's kstop-th entry, which no longer exists, so eps goes inert and the
    shared bound is never written. Termination is then by local minimum or by
    budget, which is the natural rule for a narrow queue anyway.
    """
    kk = min(k, state.ids.shape[-1])
    return state.ids[..., :kk].clone(), state.dist[..., :kk].clone()


def merge_branches(ids: torch.Tensor, dist: torch.Tensor, k: int,
                   fused: bool = True):
    """Branch results -> one top-k per query, duplicates removed by id.

    The fused path is one kernel, one block per query. The tensor path below it
    is kept because it is the definition the kernel is tested against, and
    because the kernel asks for shared memory proportional to V*kk and will
    refuse a configuration the tensor path still serves.
    """
    if fused and ids.is_cuda and ids.dim() == 3:
        try:
            oi, od = ext().merge_branches(ids.contiguous().to(torch.int32),
                                          dist.contiguous().float(), k)
            return oi, od
        except Exception:
            pass          # shared memory refusal, or an unsupported shape
    return _merge_branches_tensor(ids, dist, k)


def _merge_branches_tensor(ids: torch.Tensor, dist: torch.Tensor, k: int):
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


def advance_chunked(X, G, Qv, state: SearchState, total: int, chunk: int,
                    codes=None) -> int:
    """Advance in chunks, re-forming the grid over the branches still running.

    Returns the number of kernel launches. A branch is done when it reports no
    candidate left or has spent its budget; dropping it from the grid frees its
    block slot for another query's branch instead of leaving it to exit on its
    first test while the slowest block in flight drains.
    """
    launches, spent = 0, 0
    while spent < total:
        alive = ((state.exhausted == 0) &
                 (state.expansions < state.B)).reshape(-1).nonzero().squeeze(1)
        if alive.numel() == 0:
            break
        step = min(chunk, total - spent)
        advance(X, G, Qv, state, step, active=alive.to(torch.int32).contiguous(),
                codes=codes)
        spent += step
        launches += 1
    return launches


def search(X, G, Qv, seeds, k=10, L=16, B=16, metric=METRIC_IP, team_size=32,
           policy=POLICY_BEST, stop_eps=-1.0, shared_cache=False,
           shared_bound=False, exclusive=False, stall_patience=0, crowd_frac=0.0,
           elide_visited=False, chunk=0, codes=None):
    """init -> advance(budget) -> per-branch best-k -> merged top-k. One shot."""
    st = init_states(X, Qv, seeds, L=L, B=B, D=G.shape[1],
                     metric=metric, team_size=team_size, policy=policy,
                     stop_eps=stop_eps, kstop=k, shared_cache=shared_cache,
                     shared_bound=shared_bound, exclusive=exclusive,
                     stall_patience=stall_patience, crowd_frac=crowd_frac,
                     elide_visited=elide_visited)
    if chunk:
        advance_chunked(X, G, Qv, st, B, chunk, codes=codes)
    else:
        advance(X, G, Qv, st, B, codes=codes)
    if _OVERFLOW_GUARD:
        st.check()
    bi, bd = branch_topk(st, k)
    ids, dist = merge_branches(bi, bd, k)
    return ids, dist, st
