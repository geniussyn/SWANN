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

1. **The difficulty pass.** Train queries (the lab's MS MARCO train pool, GT
   recomputed on this subset), phi-hat_1 to find the pairs a degree-d builder
   leaves cut, and the manuscript's own repair -- reciprocal bridge, each
   endpoint drops its longest edge -- applied at build time instead of post hoc.
2. **Build time.** 40s against CAGRA's 4.85s. The prune is 128 sequential passes
   over chunks of 1024 nodes; it should be one kernel.
3. **A two-level entry catalog.** Density is the lever that works and P=16384
   costs 3.5ms of flat scoring; a coarse-to-fine catalog should keep the density
   and drop most of the cost.
4. **Arrival, directly.** 36% of fragments are still never reached at phi>=3 and
   the oracle says entries that reach them are in the pool. No query-independent
   rule finds them and neither does proximity or spread. B (probe then select) is
   the untried source of signal; so is raising the queue ceiling past m=128,
   which is currently where the recall curve ends.
5. C (fused greedy) stays on hold: it makes selection cheaper, and selection is
   not where anything is lost.
