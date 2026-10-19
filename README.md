# gpuann — multi-entry ANN on one GPU

Implementation of *GPU 다중 엔트리 ANN 구현 설계안 3종* (2026-10-19). Fixed
vectors, fixed search graph; the only thing the designs are allowed to differ in
is **which entries a query starts from**.

Stage 1 of the task order (common engine) and the H0 oracle diagnostic are
done and tested. A, B and C are not started.

```
gpuann/csrc/local_search.cu   the engine: init_states / advance, one CTA per (query, branch)
gpuann/engine.py              state, budgets, branch merge; the semantics live here
gpuann/refcpu.py              the same search in Python, to the bit
offline/prep_data.py          100K corpus, held-out query split, exact GT, graph
offline/oracle.py             H0: the exact ceiling on entry selection (2^10 DP)
bench/entry_arms.py           baseline arms: single seed / random / nearest-by-distance
tests/test_engine.py          stage 1 acceptance, 8 checks
```

## Fixed semantics

Restated nowhere else, so that A/B/C cannot quietly redefine them:

- the queue is best-*m* over everything evaluated so far, ordered by
  `(distance, id)`, and the `expanded` flag moves with its entry through every
  merge;
- a branch stops on **budget spent** or **no retained candidate unexpanded** —
  never because a round left the queue unchanged;
- the budget counts **node expansions**, per branch; distance evaluations are a
  separate counter and are never substituted for it;
- `advance(a)` then `advance(b)` equals `advance(a+b)`: queue, distances, flags,
  visited table and counters all survive the call boundary (tested);
- a full visited table raises. It never drops a node quietly.

## Three deviations from the document, and why

**torch, not CuPy/pybind11.** CuPy is not installed on this box and torch is,
with the C++/CUDA extension toolchain it needs. The kernel reads raw contiguous
pointers, so the harness choice changes nothing it computes.

**Built with `-fmad=false`.** Left on, nvcc contracts `q*x + acc` into an FMA
whose unrounded product no CPU reference can reproduce, and the document's own
acceptance test is agreement with a CPU local search. The distance loop reads D
floats to do D flops, so it is memory-bound and the fix costs nothing
measurable. In exchange `tests/test_engine.py` asserts *bitwise* equality
against `refcpu.py` instead of equality within a tolerance that a real
reordering bug could hide under.

**The graph is an exact kNN graph, as a stand-in.** It makes the engine
measurable on real vectors today. It is *not* a navigable built graph, so no
number from it is a CAGRA comparison, and recall against it is lower than a
built graph would give. `prep_data.py` is the only file that changes when a real
graph replaces it.

## Measured so far

`msmarco-1m` first 100K (d=1024, L2-normalised, IP), **CAGRA-built graph**
(degree 64, cuVS 26.10), 3490 validation queries, k=10. Nothing is tuned and the
test half is untouched. Every arm below searches the same graph, the same
vectors and the same queries, and every latency is a batch of all 3490 queries,
best of three after a warm-up.

The exact-kNN stand-in graph was replaced once cuVS was available: it cost about
8 recall points at equal time, which would have made every comparison a
comparison against a handicapped baseline. It is kept as `G_knn.npy`.

### Against native CAGRA, at matched recall

`bench/compare.py` reads CAGRA's own recall-latency frontier and reports, for
each of our configurations, the time CAGRA needs for *that* recall.
`>1x` means ours is faster.

| ours | recall@10 | ours ms | CAGRA ms | |
|---|---|---|---|---|
| m=16, E=1 | 0.8702 | 7.61 | 6.89 | 0.90x |
| m=24, E=1 | 0.9080 | 10.03 | 9.52 | 0.95x |
| m=32, E=1 | 0.9294 | 11.86 | 11.78 | 0.99x |
| **m=48, E=1** | **0.9503** | **15.72** | **17.12** | **1.09x** |
| **m=64, E=1** | **0.9635** | **19.61** | **21.32** | **1.09x** |
| m=16, E=8 distance | 0.9372 | 26.42 | 13.60 | 0.51x |
| m=16, E=16 distance | 0.9493 | 26.29 | 16.86 | 0.64x |
| m=64, E=2 distance | 0.9646 | 29.60 | 21.86 | 0.74x |

So the engine itself is competitive: at recall 0.95 and above it is about 9%
faster than native CAGRA, and at lower recall about 1-10% slower. Nothing in it
has been tuned for speed yet -- the merge is serial in one thread and branches
do not share a visited set.

### The finding that matters: multi-entry is currently dominated

At a fixed 128 node expansions per query, **every multi-entry arm loses to
simply making one branch's queue wider**:

| | recall@10 | ms | distances/query |
|---|---|---|---|
| E=1, m=64 | 0.9635 | 19.61 | 2495 |
| E=16, m=16, nearest-by-distance | 0.9493 | 26.29 | 6910 |

Worse recall, 34% more time, 2.8x the distance work. The branches re-evaluate
each other's nodes -- they hold independent visited tables by design -- and each
gets a budget too small to go deep. Multiple starting points do not pay for
themselves here, and no entry selector can be credited for a gain until it beats
this row.

(At a fixed budget, `m` only matters once `budget > m`: with 16 expansions per
branch, m=16, 32 and 64 produce bitwise identical results, which is why the
E>=8 rows repeat.)

### H0 oracle: the headroom is real, and it is at high recall

P=512 random pool seeds, 16 expansions per branch, E=8 -- the *same* 128
expansions the arms above spend:

| | recall@10 |
|---|---|
| one pool seed, alone | 0.7914 |
| best single seed (exact DP, E=1) | 0.9310 |
| best 2 / 4 / 8 seeds (exact DP) | 0.9826 / 0.9985 / **0.9996** |
| measured, nearest-by-distance E=8 | 0.9372 |

Selection is worth 0.9372 -> 0.9996 at identical work. And a selector that
reached it would sit at a recall native CAGRA needs well over 41.6ms to
approach (its measured frontier ends at 0.9830 @ 41.6ms) while spending 26ms.

That sharpens what A and B have to show: **the case for multi-entry is at recall
>= 0.98, not at 0.93**, and it needs near-oracle selection plus a cheaper
multi-branch cost structure (shared visited, or fewer and deeper branches) to
survive its own overhead.

## A (reach sketch): built, measured, and falsified on this corpus

A is implemented end to end -- METIS k-way regions, representatives, the forward
reach sketch, query-weighted greedy coverage -- and four more placements were
built so the pool could be compared rather than assumed.

**Placement is not a lever.** Five placements of P=512 entries sharing 1-4 ids
between them give the same oracle ceiling to four decimals at every E (random,
METIS, k-means, farthest-point, degree hubs; 1-seed 0.763-0.769, E=8
0.9992-0.9996). Graph regions, geometric regions, maximal spread and pure hubs
are indistinguishable.

**The sketch maximises its own objective and does not move recall.** The greedy
lifts F by 38% at E=2 and covers 99.5% of all region weight at E=8; recall moves
by 0.0008 to 0.0033 and costs 1-2.4ms to choose. The document's own failure test
applies: the surrogate is mis-specified, not under-optimised. Static,
query-independent information does not predict which entries recover a query's
missing top-10.

## Fragments: where the recall actually goes

phi_h (`offline/phi_h.py`, the manuscript's Definition 2.1 unchanged) counts the
pieces the serving graph cuts a query's exact top-10 into. It is the instrument
the whole multi-entry argument needed, and it reframes the problem:

| graph | mean phi_1 | mean phi_2 | one piece (h=2) |
|---|---|---|---|
| exact kNN (cheap, unpruned) | 3.887 | 2.287 | 50.6% |
| CAGRA-built | 3.273 | 1.328 | 80.8% |

CAGRA's expensive optimisation step is, measured this way, largely a
fragmentation repair.

Splitting the loss (`bench/fragments.py`, cheap graph, single seed m=64):

| queries | recall@10 | fragments reached | recall *inside* reached fragments |
|---|---|---|---|
| phi=1 (51%) | 0.9904 | 0.9989 | 0.9904 |
| phi=2 (20%) | 0.9206 | 0.7553 | 0.9742 |
| phi>=3 (29%) | 0.7452 | 0.6010 | 0.9649 |

**Every point of loss is a fragment never reached.** Once a piece is touched it
is 96-99% recovered on a small budget -- so "recover each fragment and stop" is
not aspirational, the stopping half is already free. The whole problem is
arrival.

And the split-branch multi-entry of the original design makes arrival *worse*
(0.5490 fragments reached at phi>=3, against 0.6010 for one deep wide search):
eight shallow branches from eight nearby entries see less than one branch that
is allowed to go deep.

## What works: one state, several seeds, taken in turn

The fix follows from the diagnosis. Keep the seeds, drop the branches: S entries
share one queue, one visited table and one budget, and every queue entry carries
the lineage of the seed it descends from so expansion can be taken **in turn
across lineages**. Sharing removes the duplicated work (17.9 distances per
expansion against 28.1 for split branches, tested); taking turns stops the
nearest seed's basin from eating the budget, which is the only reason the extra
seeds were there.

Both halves are necessary (cheap graph, P=512, m=64, 128 expansions):

| structure | recall@10 | total ms | fragments reached |
|---|---|---|---|
| one seed | 0.9046 | 15.69 | 0.8334 |
| 8 seeds, split branches | 0.8692 | 18.20 | 0.8102 |
| 8 seeds, shared, nearest-first | 0.9054 | 15.03 | 0.8343 |
| **8 seeds, shared, round-robin** | **0.9150** | 16.18 | **0.8476** |

Nearest-first sharing changes nothing -- the nearest basin still takes every
expansion. Round-robin is the whole effect.

The remaining lever is catalog density, not selection rule. Selecting the 8
entries by coverage (A) or by near-but-spread max-min gives 0.9118 and 0.9114
against 0.9150 for plain nearest. Enlarging the catalog does what no rule did:

| entry catalog | recall@10 | distances/query in search |
|---|---|---|
| P=512 | 0.9150 | 2064 |
| P=4096 | 0.9226 | 1984 |
| P=16384 | 0.9337 | 1866 |

A denser catalog makes the *search* cheaper as well as better, and pays for it
in catalog scoring (3.5ms at P=16384, which a two-level catalog should mostly
remove -- not done).

### Against native CAGRA, on CAGRA's own graph

Selection cost included, matched recall, 3490 val queries. `>1x` is ours faster:

| ours | recall@10 | ours ms | CAGRA ms | |
|---|---|---|---|---|
| S=1, m=64 (the old best) | 0.9635 | 19.61 | 21.28 | 1.09x |
| S=8, m=64, B=64 | 0.9662 | 18.45 | 23.34 | 1.26x |
| S=8, m=64 | 0.9718 | 20.93 | 28.41 | **1.36x** |
| S=8, m=96 | 0.9807 | 30.97 | 37.87 | 1.22x |
| S=8, m=128 | 0.9854 | 38.09 | 48.55 | 1.27x |
| S=32, m=128 | 0.9871 | 40.37 | 53.51 | 1.33x |

1.22-1.36x across the whole 0.966-0.987 band, against 1.09x before. The search
also **stops by itself**: at S=8, m=64 the queue runs out of unexpanded
candidates at ~96 expansions, so budgets of 128, 192, 256 and 384 all return the
same answer in the same time. The binding constraint is queue width, never the
budget.

### Dropping CAGRA: not yet earned

On the cheap exact-kNN graph the same method reaches 0.9337 / 0.9483 / 0.9572 at
m=64/96/128 (19.8 / 26.6 / 34.1 ms) and cannot reach 0.97 within the queue
ceiling at all. Build is 0.92s against CAGRA's 4.85s at 100K, so for this corpus
the cheap graph is ahead on *total* cost until roughly a million queries have
been served -- and behind forever after that. Exact kNN is also O(N^2) and stops
being cheap at 1M. A real answer needs a fast approximate builder of our own
(NN-descent) measured on build *and* query together; that is not written yet.

## A builder of our own, and what phi-hat can and cannot drive

`offline/build_graph.py` builds the search graph on the GPU with no cuVS:
exact top-C candidates, a DiskANN-style robust prune (keep v unless an
already-kept w satisfies alpha*d(w,v) <= d(u,v)), then reverse edges with a
re-prune of the union. 40s at 100K against CAGRA's 4.85s -- the prune loop is
unoptimised and that is the one number here that is worse.

| graph | build | mean varphi_2 | one piece |
|---|---|---|---|
| exact kNN (no prune) | 0.92s | 2.287 | 50.6% |
| **ours (pruned + reverse)** | **40.0s** | **1.539** | **73.4%** |
| CAGRA | 4.85s | 1.328 | 80.8% |

Searched with the shared round-robin engine, against native CAGRA on *CAGRA's*
graph, at matched recall:

| ours, on our graph | recall@10 | ours ms | CAGRA ms | |
|---|---|---|---|---|
| S=8, m=64 | 0.9679 | 20.27 | 24.80 | 1.22x |
| S=8, m=96 | 0.9778 | 26.77 | 33.16 | 1.24x |
| S=8, m=128 | 0.9835 | 33.93 | 43.25 | 1.27x |

Which is the same 1.22-1.27x the CAGRA graph gave us. **The cuVS dependency is
gone from the query side at no cost**; what it still buys is a build 8x faster,
and that is an engineering gap, not a quality one.

### phi-hat_1, and why the difficulty pass has to be query-aware

`offline/phihat.py` computes the manuscript's index-free screen --
`W = min(rank_i(j), rank_j(i))`, `adj = W <= d`, components -- in one corpus
sweep (1.0s at 100K, no index). It is the natural signal for a builder because
the threshold in its definition *is* the out-degree budget: raising node u's
degree from d to d' joins exactly the pairs with rank_u(.) in (d, d'].

On real queries the signal is strong:

| out-degree d | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| phi-hat_1 | 3.994 | 3.146 | 2.490 | 1.970 | 1.569 |
| Pr[phi=1] | 0.208 | 0.338 | 0.468 | 0.596 | 0.718 |

On the corpus's own neighbourhoods it is flat at **phi-hat_1 = 1.000 for every
degree from 16 to 256**, with median mutual rank 15 against 99 for queries. A
document's top-k is centred on an actual document and is always cohesive.

So difficulty is not a property of the corpus that a query-free builder could
read off: it is a property of where queries land relative to it. A
difficulty-aware pass must see *training* queries, held out from everything it
is later measured on. That is a design constraint the measurement imposed, and
it is the next thing to build.

## LID-allocated degree, a faster build, and an honest fp16 comparison

**Build.** The prune was 128 sequential passes over chunks of 1024 nodes -- 12k
tiny kernel launches. Chunking at 4096 took the build from 40.0s to 7.8s cold
(2.8s warm), against CAGRA's 4.85s.

**Degree by LID.** `lid_mle` is the Levina-Bickel/Amsaleg MLE over the candidate
distances the builder already has: no query, no index, no extra pass. Degree is
allocated by LID *rank* between 0.5x and 1.5x of the target, with the mean pinned
so the index does not grow. At 100K: LID 5%/median/95% = 11.4/27.8/55.8, degrees
32..96.

| graph (deg 64, equal total edges) | mean varphi_2 | one piece |
|---|---|---|
| ours, flat degree | 1.539 | 73.4% |
| **ours, LID-allocated** | **1.479** | **74.2%** |
| CAGRA | 1.328 | 80.8% |

alpha for the prune was swept: 1.0 gives 2.704, **1.2 gives 1.479**, 1.5 gives
2.011, 2.0 gives 2.206 (at 2.0 almost nothing is pruned and it is a kNN graph
again). 1.2 was already the default and is the right one.

**The search is at the memory roofline.** At fp32, m=128: 3938 distances/query
is 52.4 GiB of vector reads for the batch in 31.9ms = **1.76 TB/s**, on a part
whose HBM is about 1.8. Kernel-level work cannot help from there, and measurement
agreed: team size 8/16/32 differ by 2%, and replacing the serial merge with a
parallel merge-path merge (still bitwise identical to the CPU reference) changed
little. The only levers left are fewer vectors or smaller ones.

**fp16, and the comparison it reverses.** Storing the corpus in half with
16-byte vectorised loads costs 0.0003 recall and 1.6x less time. But CAGRA gains
*more* from the same change -- 41.6ms to 11.1ms at recall 0.983, a 3.7x -- and
with both sides on fp16 the result flips:

| ours (fp16, our graph) | recall@10 | ours ms | CAGRA fp16 ms | |
|---|---|---|---|---|
| m=128, S=8 | 0.9852 | 20.98 | 12.70 | 0.61x |
| m=96, S=8 | 0.9800 | 16.87 | 10.00 | 0.59x |
| m=64, S=8 | 0.9707 | 12.97 | 7.47 | 0.58x |

**So the earlier 1.22-1.36x was an artefact of CAGRA's fp32 path, not a win.**
At the precision anyone would actually serve, native CAGRA is about 1.6x ahead of
us, consistently across the band. Reported here rather than left in the earlier
table.

The gap decomposes: we run the search at ~1.65 TB/s and CAGRA would need 2.45
TB/s to do our distance count in its time, so it is doing roughly a third fewer
distance evaluations as well as moving bytes better. Ours stops only when the
queue holds no unexpanded candidate, which is thorough and late -- the budget
sweep at m=128 shows the tail is expensive (0.9770 at 2975 distances, 0.9854 at
3843) and that budget, not queue width, is the real knob.

**On a global distance cache** (superseded -- see the latency section below;
this holds only for the single-CTA shared state). It is already there, and
measurement says there is nothing left for it to collect: with one shared state per query the visited
table makes every node's distance computed exactly once, so `dist_evals` equals
the number of distinct nodes evaluated and no lineage ever recomputes another's
work (17.9 distances per expansion against 28.1 for split branches). Caching
*values* rather than ids would only matter if branches were independent again,
and the measurements say they should not be.

## Stopping: retire a lineage when nothing it can reach will beat the answer

The rule was chosen by replaying searches one expansion at a time
(`bench/stopping.py`) and recording, per step, whether a ground-truth document
arrived and whether it was the *first of its fragment*. That distinction decides
everything, because it says what the tail is doing:

| expansions | new gt documents | first of a fragment |
|---|---|---|
| 1-96 | 9976 | 14% |
| 97-256 | 104 | **47%** |

The tail is arrival, not mining. A rule that stops on "no improvement" cuts
exactly the documents that were hardest to get. But at expansion 97 only 3 of
1024 queries still moved, while the current rule runs *every* query to ~131
expansions (median 130, p95 138) -- no per-query adaptivity at all. That is the
waste worth taking.

Rules, simulated on the recorded traces:

| rule | recall@10 | mean expansions |
|---|---|---|
| exhaust the queue (old) | 0.9844 | 131.4 |
| top-k idle for 64 expansions | 0.9796 | 97.5 |
| **all lineages beyond radius x 1.1** | **0.9824** | **99.3** |
| all lineages beyond radius x 1.25 | 0.9844 | 130.4 |
| oracle: stop at the last useful expansion | 0.9844 | 36.1 |

The radius rule dominates the idle rule on both axes, so that is the one, applied
**per lineage rather than globally**: a lineage retires when its own best
unexpanded candidate lies beyond `(1 + eps)` times the current k-th best
distance, and the search stops when all have retired. Retiring individually
matters because round-robin otherwise keeps handing turns to lineages that are
finished. It is free to evaluate -- the queue is sorted, so a lineage's best
unexpanded candidate is the first one the turn-scan already finds -- and the
reference implements it too, so the tests still assert bitwise agreement (11/11).

`eps` is the travel allowance, and it is measured rather than guessed
(m=128, S=4, fp16):

| eps | recall@10 | distances/query | total ms |
|---|---|---|---|
| off (exhaust) | 0.9854 | 3855 | 20.64 |
| 0.25 | 0.9854 | 3791 | 20.65 |
| **0.15** | **0.9851** | **3422** | **19.97** |
| 0.10 | 0.9803 | 2538 | 16.43 |
| 0.05 | 0.9384 | 1185 | 11.09 |

**What it does not do.** eps=0.10 removes 34% of the distance work, but at the
recall it lands on, simply running a narrower queue to exhaustion costs the same
(m=96 reaches 0.9800 in 16.87ms against 0.9803 in 16.43ms). The rule is worth
having -- eps=0.15 is 11% fewer distances for 0.0003 recall, and it is free --
but it does not move the Pareto, and the measurement says why: the tail is
travel, travel needs slack, and slack is what stops the rule from firing.

The oracle spends 36 expansions where we spend 131, and no radius or idle rule
comes near it. That gap is not a stopping problem. It is an arrival problem: a
search that started *inside* the far fragment would not need to travel to it, and
the entry oracle (0.9996 at E=8) says such entries are in the pool. Denser
catalogs move in that direction and cost scanning time to do it -- P=16384 raises
recall to 0.9883 from 0.9857 and spends 3.5ms of flat fp32 scan to get there,
which a two-level catalog should mostly remove.

## The regime the batch numbers hid: one query at a time

Every number above this section was measured at Q=3490, where the GPU is already
saturated and an extra branch costs its own work in wall time. A single query
saturates nothing: one CTA per branch means E=100 branches occupy 100 of 188 SMs
and run at the same time as one does. `bench/latency.py` measures that axis.

| config | Q=1 | Q=8 | Q=64 | Q=256 | Q=1024 |
|---|---|---|---|---|---|
| shared state, S=8, m=128 | 1.71ms | 5.92 | 7.95 | 9.24 | **9.99** |
| **split branches, E=100, m=16, B=16** | **0.63** | **0.89** | **2.91** | 9.17 | 34.0 |

The crossover is near Q=256. Below it wide fan-out wins outright -- at Q=64 it is
2.7x faster *and* more accurate (0.9953 against 0.9938). The design the batch
benchmarks rejected is the right one for interactive latency, and the earlier
conclusion that "multi-entry is dominated" holds only in the saturated regime
those benchmarks measured.

E is a clean knob there (Q=64, scattered entries): E=32/100/200/400 gives
0.9828/0.9922/0.9953/0.9969 at 1.29/3.18/5.34/9.81 ms. Entry placement still
barely matters -- at E=100 the nearest-100 reach 0.9953, maximally spread and
random both 0.9922.

### But CAGRA already does this, and does it better

| Q=64, matched recall | ours | native CAGRA multi_cta |
|---|---|---|
| 0.9953 | 2.96ms | ~0.74ms |
| 0.9969 | 9.81ms (E=400) | 0.79ms (itopk 256) |

`multi_cta` *is* the many-CTAs-per-query mode, so the idea is right and the
implementation is 4-12x behind. The cause is measurable and specific:

| config (Q=64) | distance evals/query | distinct nodes touched | redundancy |
|---|---|---|---|
| E=100 split branches | 71,963 | 8,826 | **8.2x** |
| shared state, S=8 | 4,234 | 4,234 | 1.0x |

A hundred independent branches re-evaluate each other's nodes eight times over.
Which makes the global distance cache -- a per-query visited/distance table
shared by all of a query's CTAs, written with atomics -- not an optimisation but
the thing that makes fan-out affordable. It was dismissed earlier in this README
on the grounds that the shared single-CTA design had no redundancy left to
collect; that was true of *that* design and is wrong for this one.

## The global distance cache

A distance belongs to a (query, node) pair, not to a branch. The per-branch
visited table stays -- it is what keeps a node out of the same branch's queue
twice -- but the arithmetic now goes through one table per *query*, shared by all
of its branch CTAs, in L2 with volatile access and `atomicCAS` to claim a slot.

The design has a property worth more than its speed: **a recomputed distance is
bitwise equal to a cached one**, because the inputs and the reduction order are
the same. So a miss, a lost race or a full table costs work and never
correctness -- which is why the probe is bounded at 32 steps and a full table
just recomputes -- and `test_shared_cache_is_free_of_consequence` asserts that
the whole search comes out identical with the cache on, with only the distance
count moving (12/12).

| Q=64, E=100 branches, m=16, B=16 | recall@10 | distances/query | p50 |
|---|---|---|---|
| no cache | 0.9953 | 71,963 | 3.06ms |
| **shared cache** | **0.9953** | **10,212** | **2.51ms** |
| (shared state, S=8, m=128, for scale) | 0.9938 | 4,234 | 7.95ms |

86% of the distance work is gone and 18% of the time with it; at Q=256 it is 24%
(9.70 to 7.33ms). The gap between those two numbers is the finding: **after the
cache, distance computation is no longer what the fan-out kernel spends its time
on.** A budget sweep separates it -- 0.28ms before any expansion happens, then
about 0.14ms per expansion across 6400 CTAs -- and the remainder is spread across
graph reads, two hash probes per neighbour, the candidate sort and the merge,
with no single dominant term. Narrowing the bitonic sort from a fixed 128 slots
to `next_pow2(deg)` (21 stages instead of 28) and sizing the cache for the
overlap that actually occurs rather than the worst case both landed inside the
noise, which is the same message.

### What is still missing, and it is structural

Native CAGRA `multi_cta` reaches 0.9953 at Q=64 in about 0.74ms against our
2.51ms. The table above says why: at the same recall our fan-out needs 10,212
distances where the shared state needs 4,234. A hundred independent queues
re-discover each other's neighbourhoods even when the distances are free.

CAGRA's multi_cta does not run independent searches -- its CTAs cooperate on one
candidate list. That is the difference between sharing *distances* and sharing
*the search*, and it is the next change: a top-m list per query, in global
memory, that every CTA of that query takes its next expansion from.

## The shared bound: one number the whole query agrees on

The distance cache stopped branches recomputing each other's arithmetic; it did
not stop them *searching* places the answer had already passed. A branch tests
its frontier against a radius, and with m=16 its own radius comes from sixteen
slots and is far too loose to cut anything. So the radius becomes a property of
the query rather than of the branch: one `bound[q]` holding the best k-th
distance any of its CTAs has reached, updated by `atomicMin` on the bit pattern
(distances are non-negative, so integer order is float order) and read by every
branch when it decides whether to keep expanding.

Measured at Q=64, E=100, m=16, fp16, recall 0.9953 throughout:

| | p50 |
|---|---|
| plain | 3.05ms |
| + shared distance cache | 2.54ms |
| + retirement on the branch's *own* radius | 2.53ms |
| **+ retirement on the shared bound** | **1.95ms** |

The local radius is worth nothing and the shared one is worth 1.3x, which is the
whole point: a branch cannot tell that it is wasting its time, and the query can.
eps is the same travel slack as before -- 0.0 gives 0.9875 at 1.35ms, 0.1 gives
0.9953 at 1.95ms.

**It should not be reproducible and in practice it is.** Which branch retires
depends on which CTA lowered the bound first, so the bitwise determinism the rest
of the engine guarantees does not extend here; the flag is off by default for
that reason. Across 8 runs the returned ids were identical and the recall
standard deviation was 0.00000, because the branches the bound cuts are cut by a
wide margin. That is an observation, not a guarantee.

### Entries become nearly free, which was the point of fan-out

At Q=1 the GPU is nowhere near full, and with useless branches dying immediately
the cost of more entries almost vanishes (recall does not depend on Q -- queries
are independent -- so it is the Q=64 figure):

| E | recall@10 | p50 at Q=1 | p50 at Q=64 |
|---|---|---|---|
| 100 | 0.9953 | 0.48ms | 1.93ms |
| 200 | 0.9969 | 0.50ms | 2.91ms |
| 400 | 0.9984 | 0.52ms | 4.18ms |

Four hundred entries cost 8% more than a hundred at Q=1 and buy 0.3 recall
points. At Q=64 the SMs are full again and E costs what it weighs.

Against native CAGRA `multi_cta` at Q=1: 0.9969 costs it 0.33ms and costs us
0.50ms. The gap is now about 1.5x, from 4x before the cache and the bound.

## Head to head with CAGRA: fp32, test split, each on its own graph

Everything in this section is fp32 -- the design document's "first comparison is
the fp32 exact distance" -- on the **test** half, which nothing in this project
has been tuned on. Each system searches the graph its own builder made: ours from
`offline/build_graph.py --lid`, CAGRA's from `cagra.build`. QPS is always the
full 3490-query batch; what changes between the two tables is only which queries
the recall is read on.

### Build

| | build | mean varphi_2 |
|---|---|---|
| **ours** (candidates, robust prune, reverse, LID degree) | **1.81s** | 1.479 |
| CAGRA (graph_degree 64, intermediate 128) | 4.65s | 1.328 |

2.6x faster to build, on a graph that is still measurably more fragmented. Both
timed in-process, warm, best of three.

### recall-QPS, all queries

| ours | recall@10 | QPS | CAGRA at the same recall | |
|---|---|---|---|---|
| m=32 | 0.9412 | 241,683 | 15.17ms | 1.06x |
| m=48 | 0.9623 | 191,927 | 21.33ms | 1.17x |
| m=64 | 0.9711 | 160,446 | 28.27ms | 1.30x |
| m=96 | 0.9815 | 125,008 | 39.99ms | 1.43x |
| m=128 | 0.9861 | 103,540 | 50.92ms | 1.51x |
| m=128, P=16k | 0.9877 | 98,655 | 55.06ms | **1.56x** |

The advantage grows with recall: a tie at 0.94, half again as fast at 0.986.

### recall-QPS, hardest 10% of queries

The decile is chosen by phi-hat_1 at out-degree 64 -- index-free, computed from
the embedding alone -- so **both systems are scored on the same 349 queries**,
and the choice cannot favour either graph. Those queries average phi-hat_1 7.02
against 2.46 overall. The batch stays whole at 3490 queries; only the recall is
restricted, because running 349 queries as their own batch leaves 349 CTAs on
188 SMs and measures occupancy rather than difficulty (it reverses the result if
you let it).

| ours | hard@10 | all@10 | QPS | CAGRA at the same hard@10 | |
|---|---|---|---|---|---|
| m=32 | 0.7524 | 0.9412 | 241,683 | 15.24ms | 1.06x |
| m=48 | 0.8175 | 0.9623 | 191,927 | 21.17ms | 1.16x |
| m=64 | 0.8550 | 0.9711 | 160,446 | 28.17ms | 1.30x |
| m=96 | 0.9026 | 0.9815 | 125,008 | 38.87ms | 1.39x |
| m=128 | 0.9249 | 0.9861 | 103,540 | 48.26ms | 1.43x |
| m=128, S=16 | 0.9327 | 0.9873 | 97,990 | 52.84ms | 1.48x |

**The hard decile is not where the advantage lives.** The speedup profile is the
same shape as on all queries and slightly smaller at the top -- the same m=128
run is 1.51x by all-query recall and 1.43x by hard-decile recall. Hard queries
are proportionally harder for both systems.

What does differ, a little, is how much recall each loses on them: at matched
work ours falls from 0.9861 to 0.9249 (93.8% of its own all-query recall) and
CAGRA from 0.9826 to 0.9135 (93.0%). A real difference and a small one; the
multi-entry machinery is not preferentially rescuing fragmented queries at this
operating point, which is worth knowing before any claim is built on it.

## Running it

```bash
python offline/prep_data.py                      # -> /dev/shm/gpuann/msmarco-1m-100k
python tests/test_engine.py                      # 8/8
python bench/entry_arms.py  --out results/...
python offline/oracle.py    --out results/...
```

`GPUANN_DATA` relocates the corpus (default `/dev/shm`, because `/workspace` is
quota-limited). The CUDA extension builds on first import and caches.

## cuVS

Installed isolated at `/opt/cuvsenv` (3.4GB) because it brings CUDA 12.9
libraries and torch here is built against 12.8; letting them shadow each other
in one process is how a comparison becomes a bug report. The CAGRA scripts run
in their own process:

```bash
PYTHONPATH=/opt/cuvsenv python offline/build_cagra_graph.py
PYTHONPATH=/opt/cuvsenv python bench/cagra_native.py --out results/...
python bench/compare.py
```

`rm -rf /opt/cuvsenv` undoes it. Note that `cagra.from_graph` and `cagra.build`
do **not** copy: the caller owns the lifetime of the device arrays, and letting
one go is an illegal access inside the next search rather than an error at the
call.

## Next

1. **A candidate list shared across a query's CTAs.** The cache shares distances
   and the bound shares the stopping criterion; what is still unshared is the
   list itself, and the remaining ~1.5x against multi_cta is most likely there.
   A top-m list per query in global memory, with CTAs taking their next expansion
   from it, is the shape. Note what it costs: reproducibility is already gone
   with the bound, and a shared list makes that structural rather than incidental.
2. **Close the 1.6x.** Two named causes with numbers on them: a third too many
   distance evaluations (an earlier stopping rule -- top-k unchanged for X
   expansions -- instead of exhausting the queue), and ~1.65 TB/s where the part
   gives 1.8 (occupancy: the fp32 query vector costs 4 KiB of shared memory per
   CTA and caps us near 6 CTAs/SM).
3. **The difficulty pass.** Train queries (the lab's MS MARCO train pool, GT
   recomputed on this subset), phi-hat_1 to find the pairs a degree-d builder
   leaves cut, and the manuscript's own repair -- reciprocal bridge, each
   endpoint drops its longest edge -- applied at build time instead of post hoc.
4. **varphi_2 1.479 against CAGRA's 1.328.** LID closed part of it at no cost in
   edges; the query-aware pass is the rest.
5. **A two-level entry catalog.** Density is the lever that works and P=16384
   costs 3.5ms of flat scoring; a coarse-to-fine catalog should keep the density
   and drop most of the cost.
6. **Arrival, directly.** 36% of fragments are still never reached at phi>=3 and
   the oracle says entries that reach them are in the pool. No query-independent
   rule finds them and neither does proximity or spread. B (probe then select) is
   the untried source of signal; so is raising the queue ceiling past m=128,
   which is currently where the recall curve ends.
7. C (fused greedy) stays on hold: it makes selection cheaper, and selection is
   not where anything is lost.
