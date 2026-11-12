"""Solved (2026-10-27): round-robin was not reproducible because `init_kernel`
deduped a branch's seed list with the visited table.

**The bug.** A node named twice in one branch's seeds was kept once, by whichever
team won the `atomicCAS` for it. The table dedups the search itself correctly
that way, but in the seed loop the slot index *is* the lineage (`c_l[si] = si`),
so the label that survived was the one warp scheduling happened to pick. On a
query holding a duplicate seed, `lineage` came out different run to run while
`ids`, `dist`, `expanded`, `visited` and every counter matched, and
`POLICY_ROUND_ROBIN` -- the one policy that reads the label to choose a parent --
then ran a different search from the same state.

The repro's query 16 is the whole thing: its seeds are
`[827, 86, 148, 1926, 872, 1142, 827, 1109]`, with 827 at index 0 and index 6,
and it is the only one of the 24 that ever diverged.

**The fix.** Dedup the seed list by index: a node named twice is kept as the
lowest of its indices, which is a property of the seed list and of nothing else.
It measures free -- init is 0.434 -> 0.435 ms at V=16 and 3.218 -> 3.212 at
V=128 on MS MARCO 1M, because the scan hides behind the seed distance at
d=1024 -- and seeds from `select_nearest` are distinct by construction, so no
measured result moved. `test_duplicate_seeds_keep_the_lowest_lineage` asserts
the rule; on the old kernel it reports `team 8 q2: node 5 labelled [6], want 0`.

**Why it took so long**, kept because the next bug of this shape will hide the
same way. Every signal pointed away from the cause:

  - it looked like a bug in `advance`, because that is where the output moved,
    and `advance` from one cloned state is reproducible 40/40. The clone was
    copying `lineage` and so copying the answer.
  - `init_states` was checked for reproducibility and passed. The check compared
    `ids`, `dist`, `expanded`, `visited` and `counters` -- the five fields that
    do not move -- and not `lineage`. Diffing *every* field was what found it,
    in one run, after a day of narrowing.
  - it needed a sub-warp team, which reads as a shuffle or lane-mask bug. It is
    not. With BLOCK = 128 and the duplicate at seed indices 0 and 6: team 4 puts
    both in warp 0 (lanes 0 and 24 -- one instruction, and the hardware
    serialises same-address atomics by lane), and team 32 puts them in different
    `base` iterations with a `__syncthreads()` between. Both are deterministic.
    Teams 8 and 16 straddle warps and race. That is exactly the measured pattern
    -- 0/13, 3/13, 10/13, 0/13 for teams 4, 8, 16, 32 -- and it is the reason the
    team size looked like the mechanism instead of a symptom.
  - it looked like it needed a padded graph, round-robin and (at first) L2. None
    of the three is real, and widening this script to compare every field showed
    why in one run: on the old kernel all four graph x policy cells read the same
    0/7, 2/7, 6/7, 0/7 across teams 4, 8, 16, 32. The team size is the only axis
    that matters. `init_kernel` never reads `G`, and `POLICY_BEST` relabels
    exactly as often; the graph and the policy decided only whether the relabel
    went on to surface in `ids`, which was the one field being compared. Three of
    the four conditions in the original narrowing were artifacts of the test.
  - instrumenting thread 0 made it vanish, which reads as a timing race.
  - compute-sanitizer initcheck, racecheck, synccheck and memcheck are all
    clean, and *correctly* so. Nothing is read out of bounds or without a
    barrier. The atomic does exactly what it was asked to do, on a question that
    has no order-independent answer. No sanitizer can flag that; only comparing
    runs can.

The lesson, cheaply stated: when two runs disagree, diff **every** field of the
state before narrowing anything. The field that moves is the field nobody
thought to compare -- and narrowing against a partial comparison manufactures
conditions that are not conditions.

This script stays as the regression -- all four cells must read 0/7.

    python bench/repro_rr_nondeterminism.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402


def toy(n=2000, d=64, deg=64, seed=23):
    """A small L2 corpus and its exact kNN graph."""
    rng = np.random.default_rng(seed)
    X = np.ascontiguousarray(rng.standard_normal((n, d), dtype=np.float32))
    sim = -((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(sim, -np.inf)
    return X, np.ascontiguousarray(np.argsort(-sim, axis=1)[:, :deg].astype(np.int32))


def ragged(G, seed=29, lo=0.25):
    """Each row truncated to a random degree and padded with -1 at the tail."""
    rng = np.random.default_rng(seed)
    n, W = G.shape
    deg = rng.integers(max(1, int(lo * W)), W + 1, n)
    out = G.copy()
    out[np.arange(W).reshape(1, -1) >= deg.reshape(-1, 1)] = -1
    return np.ascontiguousarray(out)


def main():
    X, G0 = toy()
    rng = np.random.default_rng(31)
    Qv = np.ascontiguousarray(rng.standard_normal((24, X.shape[1]), dtype=np.float32))
    seeds = rng.integers(0, X.shape[0], (24, 1, 8)).astype(np.int32)
    dups = [q for q in range(24) if len(set(seeds[q, 0].tolist())) < 8]
    sd = torch.from_numpy(np.ascontiguousarray(seeds))
    Xg, Qg = torch.from_numpy(X).cuda(), torch.from_numpy(Qv).cuda()
    print(f"queries holding a duplicate seed: {dups}"
          + "".join(f"\n  q={q} {seeds[q, 0].tolist()}" for q in dups))

    # Every field, not only `ids`: `lineage` is where this one moved, and it was
    # the field the original check left out.
    FIELDS = ("ids", "dist", "expanded", "lineage", "visited", "counters")
    print(f"\n{'graph':8s} {'policy':7s} " +
          " ".join(f"{'team ' + str(t):>9s}" for t in (4, 8, 16, 32)))
    bad = 0
    for gtag, G in (("full", G0), ("padded", ragged(G0))):
        Gg = torch.from_numpy(G).cuda()
        for ptag, policy in (("best", engine.POLICY_BEST),
                             ("rr", engine.POLICY_ROUND_ROBIN)):
            cells = []
            for team in (4, 8, 16, 32):
                out = []
                for _ in range(8):
                    st = engine.init_states(Xg, Qg, sd, L=48, B=32, D=G.shape[1],
                                            metric=engine.METRIC_L2, team_size=team,
                                            policy=policy, kstop=10)
                    engine.advance(Xg, Gg, Qg, st, 32)
                    out.append(tuple(getattr(st, f).clone() for f in FIELDS))
                    bi, bd = engine.branch_topk(st, 10)
                    engine.merge_branches(bi, bd, 10)
                n = sum(0 if all(torch.equal(a, b) for a, b in zip(out[0], o)) else 1
                        for o in out[1:])
                bad += n > 0
                cells.append(f"{n:7d}/7")
            print(f"{gtag:8s} {ptag:7s} " + " ".join(f"{c:>9s}" for c in cells))
        del Gg
    print(f"\n{bad} of 16 cells non-reproducible" +
          ("   <-- REGRESSION" if bad else "   (fixed)"))


if __name__ == "__main__":
    main()
