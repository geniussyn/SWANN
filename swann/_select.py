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

from ._engine import METRIC_IP, METRIC_L2   # noqa: F401

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


def select_nearest(Qg, Vp, E, metric=METRIC_IP, precision="tf32", slice_P=8192):
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

