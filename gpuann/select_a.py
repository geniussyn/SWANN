"""A: pick entries by how much new region coverage their reach sketch promises.

Per query, with no probing: score the pool, keep a short candidate list, weight
every region by how close its representative is to the query, and greedily take
the E candidates whose static reach masks add the most uncovered weight.

    F(S) = sum_r weight[r] * 1[some chosen entry can reach r]

F is monotone submodular in S for fixed masks and non-negative weights, so the
greedy carries the 1-1/e guarantee *for F*. That guarantee says nothing about
recall: it bounds how well the greedy maximises a surrogate, and whether the
surrogate tracks recall is the hypothesis being tested, not an assumption.

Everything here is batched torch rather than the document's fused kernel. The
fused version is C, and it is only worth writing once A is known to be worth
running: it changes launch count and sync, not which entries come out.
"""
from __future__ import annotations

import numpy as np
import torch

from .engine import METRIC_IP, METRIC_L2   # noqa: F401

INF = float("inf")


def unpack_mask(mask_u32: np.ndarray, R: int) -> np.ndarray:
    """mask[P, ceil(R/32)] uint32 -> dense bool [P, R]."""
    bits = np.unpackbits(np.ascontiguousarray(mask_u32).view(np.uint8), bitorder="little")
    return bits.reshape(mask_u32.shape[0], -1)[:, :R].astype(bool)


class reduced_precision:
    """Scoped, because allow_tf32 is a global and the rest of the system is fp32.

    Flipping it process-wide would quietly change every other measurement in the
    same run, which is how a precision ablation stops being an ablation.
    """

    def __init__(self, on=True):
        self.on = bool(on)

    def __enter__(self):
        self.prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self.on
        return self

    def __exit__(self, *exc):
        torch.backends.cuda.matmul.allow_tf32 = self.prev
        return False


def pool_vectors(Xg, pool_t):
    """The catalogue as its own matrix, to be held for the life of the index.

    Gathering P rows out of the corpus is 537 MB at P=131072 and the pool never
    changes, so doing it inside the scan charged every query batch 0.75 ms for a
    copy that could have been made once.
    """
    return Xg[pool_t.long()].contiguous()


def catalog_scores(Qg, Xg, pool_t, metric=METRIC_IP, Vp=None, precision="tf32"):
    """Distance from every query to every pool entry, in the corpus's own metric.

    Scoring an unnormalised L2 corpus by inner product picks entries by length
    rather than by proximity, so the metric has to come along here too. The cost
    of the catalog is real and is timed with the selection, never outside it.

    This is the one matmul in the system whose precision cannot affect an answer:
    all that leaves here is which entries a search starts from, and the search
    then runs in fp32 against the real vectors. Measured at Q=2000, P=131072,
    d=1024: fp32 10.1 ms at 53 TFLOP/s, tf32 4.1 ms at 130, fp16 2.2 ms at 242,
    and tf32 picks the identical 32 entries for every query (fp16 agrees on
    99.7%). So tf32 is the default and fp32 stays available for the comparison.

    Pass `Vp` from `pool_vectors` to skip the gather.
    """
    V = Vp if Vp is not None else Xg[pool_t.long()]
    if precision == "fp16":
        qh, vh = Qg.half(), V.half()
        s = (qh @ vh.T).float()
    else:
        with reduced_precision(precision == "tf32"):
            s = Qg @ V.T
    if metric == METRIC_IP:
        return 1.0 - s                                            # [Q, P]
    return ((V * V).sum(1).view(1, -1) - 2.0 * s
            + (Qg * Qg).sum(1).view(-1, 1))


def select_nearest(Qg, Vp, E, metric=METRIC_IP, precision="tf32",
                   slice_P=8192):
    """The E nearest entries, without ever holding the whole score matrix.

    A dense catalogue is what makes multi-entry search reach the pieces it
    otherwise misses -- at P=131072 against P=4096 the nearest entry moves from
    the answer's 170th neighbour to its 7th -- and the only thing standing in the
    way is that selection used to cost a flat scan. Two things paid for it: the
    scan ran in fp32 when its whole output is a choice of starting points, and
    the [Q, P] scores went out to HBM and came straight back into the top-V. So
    the catalogue is scored a slice at a time and each slice folded into a
    running best, which is exact -- the best E overall are among the best E of
    every slice -- and reads the scores while they are still in cache.

    Q=2000, P=131072, d=1024: 15.2 ms as a gather plus fp32 scan plus topk,
    6.7 ms here, with identical entries for every query.
    """
    n, P = Qg.shape[0], Vp.shape[0]
    bi = torch.empty(n, 0, dtype=torch.long, device=Qg.device)
    bv = torch.empty(n, 0, dtype=Qg.dtype, device=Qg.device)
    qn = (Qg * Qg).sum(1).view(-1, 1) if metric != METRIC_IP else None
    for s in range(0, P, slice_P):
        V = Vp[s:s + slice_P]
        if precision == "fp16":
            sc = (Qg.half() @ V.half().T).float()
        else:
            with reduced_precision(precision == "tf32"):
                sc = Qg @ V.T
        c = (1.0 - sc) if metric == METRIC_IP else \
            ((V * V).sum(1).view(1, -1) - 2.0 * sc + qn)
        v, i = torch.topk(c, min(E, c.shape[1]), dim=1, largest=False)
        bv = torch.cat([bv, v], 1)
        bi = torch.cat([bi, i + s], 1)
        if bv.shape[1] > E:
            v, j = torch.topk(bv, E, dim=1, largest=False)
            bv, bi = v, torch.gather(bi, 1, j)
    return bi


def candidates(cat_dist, L=32, n_near=24, rng_seed=0):
    """L candidates per query: the nearest `n_near` of the pool, plus a fixed
    random sample of the pool -- fixed meaning the same entries for every query,
    drawn once from a seed. They are the document's "나머지 풀의 고정 난수": a
    standing chance for an entry no query would ever rank near, at no per-query
    cost. A collision with a query's near set costs that query a slot and is
    absorbed by the greedy, which never selects the same candidate twice.

    Every arm that selects from candidates uses this generator, so the arms
    differ in selection and not in what they were allowed to select from.
    """
    Q, P = cat_dist.shape
    n_rand = L - n_near
    near = torch.topk(cat_dist, n_near, dim=1, largest=False).indices
    if n_rand <= 0:
        return near
    g = torch.Generator().manual_seed(rng_seed)
    fixed = torch.randperm(P, generator=g)[:n_rand].to(cat_dist.device)
    return torch.cat([near, fixed.view(1, -1).expand(Q, n_rand)], dim=1)      # [Q, L]


def region_weights(Qg, Xg, reps_t, tau):
    """weight[r] = exp(-(d(q, rep_r) - min_r d) / tau). A soft "how much does
    this query care about region r", read off the representatives alone."""
    d = 1.0 - (Qg @ Xg[reps_t.long()].T)                          # [Q, R]
    d = torch.where(reps_t.view(1, -1) >= 0, d, torch.full_like(d, INF))
    return torch.exp(-(d - d.min(dim=1, keepdim=True).values) / tau)


def greedy_cover(maskd, weights, cand, cat_dist, pool_t, E):
    """E rounds of weighted maximum coverage over the candidates' reach masks.

    Ties are broken by (distance, pool id), exactly as the kernel would, so the
    arm is reproducible and a GPU backend can be checked against it.
    Candidates that add nothing fall back to distance order, and how often that
    happens is returned -- a selector that is mostly falling back is a distance
    arm wearing a sketch.
    """
    Q, L = cand.shape
    dev = cand.device
    cmask = maskd[cand.reshape(-1)].reshape(Q, L, -1).float()     # [Q, L, R]
    cdist = torch.gather(cat_dist, 1, cand)                       # [Q, L]
    cids = pool_t.long().to(dev)[cand]                            # [Q, L]
    covered = torch.zeros_like(weights)
    chosen = torch.zeros((Q, E), dtype=torch.long, device=dev)
    used = torch.zeros((Q, L), dtype=torch.bool, device=dev)
    fallbacks = 0
    for i in range(E):
        resid = weights * (1.0 - covered)                         # [Q, R]
        gain = torch.einsum("qlr,qr->ql", cmask, resid)
        gain = gain.masked_fill(used, -INF)
        best = gain.max(dim=1, keepdim=True).values
        none_left = (best <= 0)
        fallbacks += int(none_left.sum())
        # positive gain: tie-break by distance then id; no gain: pure distance order
        tie = (gain >= best - 1e-9) & ~used
        tie = torch.where(none_left, ~used, tie)
        key = torch.where(tie, cdist, torch.full_like(cdist, INF))
        bd = key.min(dim=1, keepdim=True).values
        tie2 = tie & (key <= bd)
        key2 = torch.where(tie2, cids.float(), torch.full_like(cdist, INF))
        pick = key2.argmin(dim=1)                                 # [Q]
        chosen[:, i] = pick
        used.scatter_(1, pick.view(-1, 1), True)
        covered = torch.maximum(covered, cmask[torch.arange(Q, device=dev), pick])
    seeds = torch.gather(cids, 1, chosen)                         # [Q, E] node ids
    return seeds.to(torch.int32), fallbacks / max(1, Q * E)


def select(Qg, Xg, pool_t, reps_t, maskd, E, tau, L=32, n_near=24, rng_seed=0,
           Vp=None, precision="tf32"):
    """The whole of A's online path, as one call."""
    cat = catalog_scores(Qg, Xg, pool_t, Vp=Vp, precision=precision)
    cand = candidates(cat, L=L, n_near=n_near, rng_seed=rng_seed)
    w = region_weights(Qg, Xg, reps_t, tau)
    return greedy_cover(maskd, w, cand, cat, pool_t, E)


def select_spread(Qg, Xg, pool_t, E, n_cand=64, Vp=None, precision="tf32"):
    """Near the query, but spread apart from each other.

    The documents a search never reaches are not far from the query -- they are
    in its exact top-k -- they are far from the piece of the graph the search
    settled in. So the useful entry is one that is close to the query and *not*
    close to the entries already chosen. Take the n_cand nearest pool entries and
    greedily add the one whose nearest already-chosen entry is furthest away:
    max-min in vector space, over a candidate set that is near by construction.

    No sketch, no regions, no probe. The only extra cost over distance ordering
    is one n_cand x n_cand Gram matrix per query.
    """
    cat = catalog_scores(Qg, Xg, pool_t, Vp=Vp, precision=precision)  # [Q, P]
    cand = torch.topk(cat, n_cand, dim=1, largest=False).indices  # [Q, C]
    V = Xg[pool_t.long()][cand]                                   # [Q, C, D]
    pair = 1.0 - torch.bmm(V, V.transpose(1, 2))                  # [Q, C, C]
    Q, C = cand.shape
    chosen = torch.zeros((Q, E), dtype=torch.long, device=cand.device)
    far = torch.full((Q, C), INF, device=cand.device)             # distance to chosen set
    pick = torch.zeros(Q, dtype=torch.long, device=cand.device)   # start at the nearest
    for i in range(E):
        chosen[:, i] = pick
        far = torch.minimum(far, pair[torch.arange(Q, device=cand.device), pick])
        far.scatter_(1, pick.view(-1, 1), -INF)                   # never pick twice
        if i + 1 < E:
            pick = far.argmax(dim=1)
    ids = pool_t.long()[cand]
    return torch.gather(ids, 1, chosen).to(torch.int32)


def select_mmr(Qg, Xg, pool_t, E, lam=0.5, n_cand=64, Vp=None, precision="tf32"):
    """Near the query, and unlike each other, with the trade-off exposed.

    Two documents' plain similarity predicts whether they sit in the same
    fragment of a query's answer at AUC 0.78-0.80, which is the only usable
    signal measured so far for "these two are different pieces". Pure max-min
    spread (lam = 0) was measured and lost, because entries chosen only for being
    unlike each other are also far from the query. MMR keeps both terms:

        score(e) = lam * sim(q, e) - (1 - lam) * max_{c in chosen} sim(e, c)

    lam = 1 is plain nearest; lam = 0 is the spread arm that failed.
    """
    cat = catalog_scores(Qg, Xg, pool_t, Vp=Vp, precision=precision)  # [Q, P] distances
    cand = torch.topk(cat, n_cand, dim=1, largest=False).indices  # [Q, C]
    V = Xg[pool_t.long()][cand]                                   # [Q, C, D]
    pair = torch.bmm(V, V.transpose(1, 2))                        # similarity
    qsim = 1.0 - torch.gather(cat, 1, cand)                       # [Q, C]
    Q, C = cand.shape
    dev = cand.device
    chosen = torch.zeros((Q, E), dtype=torch.long, device=dev)
    worst = torch.full((Q, C), -INF, device=dev)                  # max sim to chosen
    pick = qsim.argmax(dim=1)
    for i in range(E):
        chosen[:, i] = pick
        worst = torch.maximum(worst, pair[torch.arange(Q, device=dev), pick])
        worst.scatter_(1, pick.view(-1, 1), INF)                  # never pick twice
        if i + 1 < E:
            pick = (lam * qsim - (1.0 - lam) * worst).argmax(dim=1)
    ids = pool_t.long()[cand]
    return torch.gather(ids, 1, chosen).to(torch.int32)
