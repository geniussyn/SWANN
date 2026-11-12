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


@torch.no_grad()
def _best_clip(X, rot, lvl=7.0, n=20000,
               grid=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0)):
    """The clip point in standard deviations that minimises reconstruction MSE.

    max|x| is the wrong scale this far down. It puts the outermost level at the
    single largest coordinate, so the step is too coarse for the bulk: measured
    on MS MARCO, max-abs gave raw recall 0.9164 at 4 bits where a fitted clip
    gives 0.9235, and at 2 bits max-abs collapses to 0.2375 because the levels
    become {-max, 0, +max} and almost everything rounds to zero. A rotated
    vector's coordinates are near-Gaussian, so the scale is a multiple of sigma
    with the tails clipped, and the multiple is fitted here against
    reconstruction error -- no queries and no ground truth.
    """
    g = torch.Generator(device="cpu").manual_seed(3)
    sel = torch.randperm(X.shape[0], generator=g)[:n].to(X.device)
    v = X[sel] @ rot
    sd = v.std(1, keepdim=True).clamp(min=1e-12)
    best, bc = float("inf"), grid[0]
    for c in grid:
        sc = (c * sd) / lvl
        q = torch.round(v / sc).clamp(-lvl - 1, lvl) * sc
        err = ((v - q) ** 2).mean().item()
        if err < best:
            best, bc = err, c
    return bc


def pack_narrow(X: torch.Tensor, bits: int = 4, rot: torch.Tensor = None,
                chunk: int = 200_000, clip: float = None):
    """(uint8 [N, d/2], scale [N], rot) -- the corpus as two 4-bit values a byte.

    Asymmetric: only the stored side is quantised and the query stays fp32,
    which is the only form the lab's quantisation work found usable. A fixed
    random rotation is applied first and is what makes a per-vector scale fit:
    without it the scale is set by one outlier coordinate and the error roughly
    doubles at every width (bench/stored_precision.py).

    The rotation has to reach the query too, which `rotate_queries` does once a
    batch -- one d x d multiply, 0.03 ms for 2000 queries at d = 1024.

    Values land in [-8, 7] and the scale is a fitted multiple of the per-vector
    standard deviation over 7, not max|x| / 7: see `_best_clip`, which is worth
    0.007 of raw recall here and is the difference between working and not at 2
    bits.

    The nibble is stored **offset-binary**, q + 8 in 0..15, and not in two's
    complement. That is not cosmetic: the kernel unpacks with the exponent trick
    -- OR the nibble into the fp32 pattern for 1.5 * 2^23 and subtract -- which
    yields the raw nibble minus a constant and therefore needs the offset form.
    In two's complement the same trick maps 7 to -1 and -8 to 0, which reads as
    a working kernel returning recall 0.
    """
    assert X.is_cuda and X.dtype == torch.float32 and X.is_contiguous()
    if bits not in (2, 4):
        raise ValueError(f"bits must be 2 or 4; got {bits}")
    N, d = X.shape
    if d % 32:
        raise ValueError(f"a packed corpus needs d % 32 == 0; got {d}")
    per = 8 // bits                      # values a byte
    lvl = float(2 ** (bits - 1) - 1)     # 7 at 4 bits, 1 at 2
    off = 2 ** (bits - 1)                # offset-binary shift
    if rot is None:
        g = torch.Generator(device="cpu").manual_seed(0)
        rot = torch.linalg.qr(torch.randn(d, d, generator=g).to(X.device))[0]
    if clip is None:
        clip = _best_clip(X, rot, lvl=lvl)
    out = torch.empty((N, d // per), dtype=torch.uint8, device=X.device)
    scale = torch.empty(N, dtype=torch.float32, device=X.device)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        v = X[s0:e] @ rot
        sc = (clip * v.std(1, keepdim=True).clamp(min=1e-12)) / lvl
        q = (torch.round(v / sc).clamp(-lvl - 1, lvl) + off).to(torch.uint8)
        packed = q[:, 0::per]
        for i in range(1, per):
            packed = packed | (q[:, i::per] << (bits * i))
        out[s0:e] = packed
        scale[s0:e] = sc.squeeze(1)
        del v, q, packed
    return out.contiguous(), scale.contiguous(), rot


@torch.no_grad()
def pack_sign(X: torch.Tensor, per_vector: bool = False, chunk: int = 200_000):
    """(uint8 [N, d/8], scale [N], identity) -- BBQ's 1-bit corpus, for xbits=1.

    The code is sign(x) per dimension: no rotation and no centring, which is
    the estimator BBQ ships (q . sign(x), bbq-rev/code/bbq_hnsw.py); rotation
    was measured to cost query-tail budget on 18/18 cells. Bit 1 means x > 0.
    Byte k holds dimensions 8k..8k+7, low bit first, so a 32-bit word holds 32
    consecutive dimensions the way the kernel's nib1 loader reads them.

    The scale only puts the score in distance units, so the queue's stop rule
    (radius * (1 + eps)) means roughly what it means in fp32. With
    per_vector=False it is one constant, mean ||x||_1 / d, and the ranking is
    exactly BBQ's. per_vector=True uses each vector's own ||x||_1 / d -- the
    least-squares scale for alpha * sign(x) -- which is a different estimator.
    """
    assert X.is_cuda and X.dtype == torch.float32 and X.is_contiguous()
    N, d = X.shape
    if d % 32:
        raise ValueError(f"a packed corpus needs d % 32 == 0; got {d}")
    out = torch.empty((N, d // 8), dtype=torch.uint8, device=X.device)
    scale = torch.empty(N, dtype=torch.float32, device=X.device)
    for s0 in range(0, N, chunk):
        e = min(s0 + chunk, N)
        v = X[s0:e]
        b = (v > 0).to(torch.uint8)
        packed = b[:, 0::8]
        for i in range(1, 8):
            packed = packed | (b[:, i::8] << i)
        out[s0:e] = packed
        scale[s0:e] = v.abs().mean(1)
        del b, packed
    if not per_vector:
        scale.fill_(float(scale.mean()))
    return out.contiguous(), scale.contiguous(), None


def pack4(X, rot=None, chunk=200_000, clip=None):
    """`pack_narrow` at 4 bits, kept because callers name it."""
    return pack_narrow(X, 4, rot, chunk, clip)


def rotate_queries(Qv: torch.Tensor, rot: torch.Tensor):
    """Queries into the basis `pack4` quantised in. Charged to the search."""
    return (Qv @ rot).contiguous()


def graph_degrees(G: torch.Tensor):
    """(deg[N] int32, max degree) for a padded adjacency matrix, or None if full.

    A pruned graph does not fill its rows: the prune keeps what is navigable and
    leaves the rest at -1, and a degree-allocated build widens the matrix to its
    *widest* node, so on GIST a 96-wide row carries 37 edges. Handed this, the
    kernel bounds every per-expansion loop by the node's own degree instead of
    by the row width, which is what makes a variable-degree graph cost what its
    edges cost rather than what its widest node costs.

    Returns None when every row is full, because then the array would be a
    pointless indirection and the kernel's `deg == nullptr` path is the same
    code with one fewer load.

    The kernel reads slots `0 .. deg[p)` and no others, so the edges have to sit
    at the front of the row. Every builder here compacts them there; a graph
    that does not is refused rather than silently searched with holes in it.
    """
    assert G.is_cuda and G.dtype == torch.int32
    valid = G >= 0
    deg = valid.sum(1).to(torch.int32)
    dmax = int(deg.max())
    if dmax == G.shape[1] and bool(valid.all()):
        return None
    # front-compacted: a row's valid slots are exactly its first deg[i]
    pos = torch.arange(G.shape[1], device=G.device).view(1, -1)
    if not bool((valid == (pos < deg.view(-1, 1))).all()):
        raise ValueError(
            "graph rows are not front-compacted: some node has a -1 before its "
            "last edge, and the kernel reads only the first deg[p] slots")
    return deg.contiguous(), dmax


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
    # Which generation this search tagged into the table's slots. A slot from an
    # earlier generation reads as empty, which is what lets the table be kept
    # between searches instead of cleared.
    cache_epoch: int = 1
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
        """A deep copy. Every field is passed by name: the positional form
        silently shifted when fields were added in the middle -- `cache_d`
        landed in `kstop` and the next `advance` raised on it -- and a dataclass
        gives no warning for that."""
        def cp(t):
            return None if t is None else t.clone()
        return SearchState(
            ids=self.ids.clone(), dist=self.dist.clone(),
            expanded=self.expanded.clone(), lineage=self.lineage.clone(),
            visited=self.visited.clone(), counters=self.counters.clone(),
            B=self.B, metric=self.metric, team_size=self.team_size,
            V=self.V, policy=self.policy, stop_eps=self.stop_eps,
            stall_patience=self.stall_patience, crowd_frac=self.crowd_frac,
            kstop=self.kstop,
            cache_id=cp(self.cache_id), cache_d=cp(self.cache_d),
            cache_epoch=self.cache_epoch, bound=cp(self.bound),
            exclusive=self.exclusive)


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


# The shared distance table is the one piece of state worth keeping between
# searches. Clearing it costs 8 bytes a slot -- 8.4 GB for a 2000-query batch at
# V = 128, which measured as the whole of the init stage -- and a generation tag
# in each slot makes the clear unnecessary: a slot written by an earlier search
# reads as empty. The tag is one byte, so every 255th search clears for real.
_CACHE_WS = {}
_CACHE_EPOCH_MAX = 127   # the kernel keeps 7 bits for it (CEPOCH_MASK)


def _cache_workspace(Q, Hc, dev):
    """(cache_id, cache_d, epoch, clear) for this shape, reused across calls."""
    key = (int(Q), int(Hc), str(dev))
    ws = _CACHE_WS.get(key)
    if ws is None:
        ws = {"id": torch.empty((Q, Hc), dtype=torch.int32, device=dev),
              "d": torch.empty((Q, Hc), dtype=torch.float32, device=dev),
              "epoch": 0}
        _CACHE_WS[key] = ws
    ws["epoch"] += 1
    clear = ws["epoch"] > _CACHE_EPOCH_MAX or ws["epoch"] == 1
    if clear:
        ws["epoch"] = 1
    return ws["id"], ws["d"], ws["epoch"], clear


def reset_cache_workspace():
    """Drop the kept tables. Only needed when the device is being freed."""
    _CACHE_WS.clear()


def auto_team(dim, dtype):
    """Lanes per vector, when the caller has not fixed it.

    A lane reads 16 bytes, so one pass over a vector covers team * 4 dimensions
    in fp32 and team * 8 in fp16, and a team that covers a vector in one pass
    leaves only BLOCK/team loads in flight from the block. Where a distance is
    cheap the limit is how many of those loads are outstanding rather than how
    wide they are, and halving the team doubles them.

    The boundaries are measured, not derived: over six corpora and three branch
    counts, team 8 beats team 16 by 1.07--1.15x in fp16 at d = 104, 128 and 256,
    by 1.01x at d = 960, and by nothing at d >= 1024. In fp32 a lane covers half
    as much, so team 16 is already several passes at those widths and wins
    everywhere below d = 2048, above which the wider team wins back addressing.
    Between d = 256 and d = 960 in fp16 nothing was measured; the rule takes the
    conservative side of that gap.
    """
    import torch as _t
    half = dtype in (_t.float16, _t.half)
    if half and dim <= 256:
        return 8
    return 32 if dim >= 2048 else 16


def init_states(X, Qv, seeds, L=16, B=16, D=None,
                metric=METRIC_IP, team_size=32, policy=POLICY_BEST,
                stop_eps=-1.0, kstop=10, shared_cache=False,
                shared_bound=False, exclusive=False, stall_patience=0,
                crowd_frac=0.0, elide_visited=False, xscale=None,
                xbits=4) -> SearchState:
    """One state per (query, branch).

    `seeds` is int32 [Q, J] for one entry per branch, or [Q, J, V] for V entries
    sharing one branch's queue, visited table and budget. -1 is an empty slot.
    The S form is the one that matters: several entries in one state cost barely
    more than one, where several branches cost several times as much.
    """
    assert X.is_cuda and X.is_contiguous() and X.dtype in (
        torch.float32, torch.float16, torch.uint8)
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
        st.cache_id, st.cache_d, st.cache_epoch, cache_clear = \
            _cache_workspace(Q, Hc, dev)
    else:
        st.cache_id = torch.empty((0, 0), dtype=torch.int32, device=dev)
        st.cache_d = torch.empty((0, 0), dtype=torch.float32, device=dev)
        st.cache_epoch, cache_clear = 1, False
    st.exclusive = bool(exclusive and shared_cache and J > 1)
    st.bound = (torch.empty((Q,), dtype=torch.float32, device=dev)
                if (shared_bound and J > 1)
                else torch.empty((0,), dtype=torch.float32, device=dev))
    ext().init_states(X, Qv, seeds, st.ids, st.dist, st.expanded, st.lineage,
                      st.visited, st.counters, metric, team_size,
                      st.cache_id, st.cache_d, st.bound, int(cache_clear),
                      xscale if xscale is not None
                      else torch.empty(0, dtype=torch.float32, device=dev),
                      int(xbits))
    return st


def advance(X, G, Qv, state: SearchState, delta_expansions: int,
            active=None, codes=None, deg=None, xscale=None,
            xbits=4) -> SearchState:
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

    `deg` is an optional (deg[N], max_deg) from `graph_degrees(G)`. Given it,
    an expansion costs the node's own degree rather than the matrix width: the
    graph read, the cache probes, the expansion rounds and the candidate sort
    all stop at deg[p]. It changes no answer -- the slots it skips are the -1
    ones the kernel was already rejecting -- so the result is bitwise identical
    either way, and that is a test.
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
    if deg is None:
        dg, dmax = empty_i, 0
    else:
        dg, dmax = deg
    ext().advance(X, G, Qv, state.ids, state.dist, state.expanded, state.lineage,
                  state.visited, state.counters, state.metric, state.team_size,
                  int(delta_expansions), int(state.B),
                  int(state.policy), int(state.V),
                  float(state.stop_eps), int(state.kstop),
                  state.cache_id, state.cache_d, state.bound,
                  int(state.cache_epoch), int(state.exclusive),
                  int(state.stall_patience), float(state.crowd_frac),
                  active if active is not None else empty_i,
                  cd, al, cm, float(mg), dg, int(dmax),
                  xscale if xscale is not None else empty_f, int(xbits))
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


@torch.no_grad()
def rerank(state: SearchState, Xf, Qv, k=10, depth=32, metric=METRIC_IP,
           chunk=256):
    """Best k of the queue, rescored against the full-precision corpus.

    A narrow stored vector loses recall by mis-ordering what it reaches, not by
    failing to reach it: on MS MARCO the 4-bit search returns 0.9185 where fp16
    returns 0.9949, and rescoring the queue takes it back to 0.9949
    (bench/stored_precision.py). So the narrow corpus is a traversal format and
    the full-precision one is the ranking format, and only `depth` of the
    queue's entries ever need the second.

    `depth` is the number of queue entries kept by the *narrow* score before any
    full vector is read, which is what an implementation has available. The knee
    is at 24 and it is saturated by 32: measured at 4 bits over three eps on MS
    MARCO 1M, FEVER 1M and NQ 500K, depth 24/32/64/128 give 0.9947/0.9948/0.9948/
    0.9948, 0.9865/0.9866/0.9866/0.9866 and 0.9985 four times. Depth 16 is
    0.0019--0.0023 short and depth 8 collapses to 0.785, because the narrow
    score's top eight does not contain the answer.

    The default was 64 and is now 32, which halves the bytes this stage reads --
    65 KB a query at d = 1024 rather than 131 -- for at most 0.0001 of recall.
    The time it saves is smaller than that suggests (2.2 ms to 1.5) because the
    stage is dominated by the dedup and sort over the whole queue rather than by
    the gather; only past depth 64 does the gather take over, and depth 128
    costs 4.7 ms.
    """
    ids = state.ids.reshape(state.ids.shape[0], -1)
    dist = state.dist.reshape(state.dist.shape[0], -1)
    out = torch.full((ids.shape[0], k), -1, dtype=torch.int32, device=ids.device)
    for s in range(0, ids.shape[0], chunk):
        e = min(s + chunk, ids.shape[0])
        I, D = ids[s:e].long(), dist[s:e]
        ok = I >= 0
        # one node can sit in several branches; keep its first slot only
        srt, order = torch.sort(I, dim=1)
        dup = torch.zeros_like(srt, dtype=torch.bool)
        dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
        bad = torch.zeros_like(dup)
        bad.scatter_(1, order, dup)
        live = ok & ~bad
        D = torch.where(live, D, torch.full_like(D, float("inf")))
        # Narrow the candidate set *before* touching a full vector. Gathering the
        # whole queue and masking afterwards reads eight times the bytes and cost
        # 17 ms against the 4-bit search's own 17; the point of the narrow format
        # is not to spend it back here.
        nd = min(depth, D.shape[1]) if depth is not None else D.shape[1]
        keep = torch.topk(D, nd, dim=1, largest=False).indices
        Ik = torch.gather(I, 1, keep)
        alive = torch.gather(live, 1, keep)
        V = Xf[Ik.clamp(min=0)]
        q = Qv[s:e].unsqueeze(1)
        dd = (1.0 - (V.float() * q).sum(-1)) if metric == METRIC_IP else \
             ((V.float() - q) ** 2).sum(-1)
        dd = torch.where(alive, dd, torch.full_like(dd, float("inf")))
        sel = torch.topk(dd, min(k, nd), dim=1, largest=False).indices
        got = torch.gather(Ik, 1, sel)
        got = torch.where(torch.gather(dd, 1, sel).isinf(),
                          torch.full_like(got, -1), got)
        out[s:e, : got.shape[1]] = got.to(torch.int32)
        del V, dd
    return out


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
                    codes=None, deg=None, xscale=None, xbits=4) -> int:
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
                codes=codes, deg=deg, xscale=xscale, xbits=xbits)
        spent += step
        launches += 1
    return launches


def search(X, G, Qv, seeds, k=10, L=16, B=16, metric=METRIC_IP, team_size=None,
           policy=POLICY_BEST, stop_eps=-1.0, shared_cache=False,
           shared_bound=False, exclusive=False, stall_patience=0, crowd_frac=0.0,
           elide_visited=False, chunk=0, codes=None, deg=None, xscale=None,
           xbits=4):
    """init -> advance(budget) -> per-branch best-k -> merged top-k. One shot.

    `deg` is (deg[N], max_deg) from `graph_degrees(G)`; pass `deg="auto"` to
    derive it here. It is a cost, not an answer: the top-k is identical with and
    without it.
    """
    if deg == "auto":
        deg = graph_degrees(G)
    if team_size is None:
        team_size = auto_team(
            X.shape[1] * ((8 // xbits) if X.dtype == torch.uint8 else 1), X.dtype)

    st = init_states(X, Qv, seeds, L=L, B=B, D=G.shape[1],
                     metric=metric, team_size=team_size, policy=policy,
                     stop_eps=stop_eps, kstop=k, shared_cache=shared_cache,
                     shared_bound=shared_bound, exclusive=exclusive,
                     stall_patience=stall_patience, crowd_frac=crowd_frac,
                     elide_visited=elide_visited, xscale=xscale, xbits=xbits)
    if chunk:
        advance_chunked(X, G, Qv, st, B, chunk, codes=codes, deg=deg,
                        xscale=xscale, xbits=xbits)
    else:
        advance(X, G, Qv, st, B, codes=codes, deg=deg, xscale=xscale, xbits=xbits)
    if _OVERFLOW_GUARD:
        st.check()
    bi, bd = branch_topk(st, k)
    ids, dist = merge_branches(bi, bd, k)
    return ids, dist, st
