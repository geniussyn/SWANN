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
reach sketch, query-weighted greedy coverage -- and `offline/place_entries.py`
adds four more placements so the pool itself could be compared rather than
assumed.

**Placement does not matter.** Five placements of P=512 entries, with 1-4 ids in
common between them, give the same ceiling at every E (oracle, 1024 val queries):

| placement | 1 seed alone | E=1 | E=2 | E=4 | E=8 |
|---|---|---|---|---|---|
| random | 0.7634 | 0.9167 | 0.9791 | 0.9982 | 0.9994 |
| metis | 0.7635 | 0.9166 | 0.9778 | 0.9981 | 0.9992 |
| kmeans | 0.7635 | 0.9170 | 0.9786 | 0.9984 | 0.9996 |
| farthest | 0.7694 | 0.9159 | 0.9782 | 0.9981 | 0.9993 |
| degree (hubs) | 0.7637 | 0.9159 | 0.9770 | 0.9980 | 0.9994 |

Graph regions, geometric regions, maximal spread and pure hubs are
indistinguishable to four decimals. Where the entries sit is not a lever on this
graph; which of them a query picks still is.

**The sketch maximises its own objective and does not move recall.** Full val,
3490 queries, tau tuned on val (and nearly flat from 0.005 to 1.0):

| structure | arm | recall@10 | select ms | total ms |
|---|---|---|---|---|
| E=1, m=64, B=64 | — | 0.9602 | 0.23 | **17.76** |
| E=2, m=64, B=64 | distance | 0.9638 | 0.29 | 29.83 |
| E=2, m=64, B=64 | **sketch (A)** | 0.9646 | 1.06 | 32.07 |
| E=8, m=16, B=16 | distance | 0.9368 | 0.25 | 26.98 |
| E=8, m=16, B=16 | **sketch (A)** | 0.9401 | 2.36 | 32.84 |

A beats nearest-by-distance by 0.0008 (E=2) to 0.0033 (E=8) recall, and pays
1-2.4ms of selection plus a little more search for it. Every row is dominated by
E=1 with a wider queue.

The design document fixes how to read that: *"proxy F만 좋아지면 영역 또는 도달
모델이 실제 검색과 맞지 않는 것이다."* F was measured directly:

| | F(distance) | F(sketch) | max possible |
|---|---|---|---|
| E=2 | 16.21 | 22.40 (+38%) | 30.15 |
| E=8 | 25.88 | 30.00 (+16%) | 30.15 |

The greedy does its job -- at E=8 it covers 99.5% of all available region weight
-- and recall does not follow. The surrogate is mis-specified, not
under-optimised: **static reach information does not predict which entries
recover a query's missing top-10 on this graph.** No tuning of tau, of the pool,
or of the region count addresses that, and none of them was close to binding.

What survives: the oracle gap is unchanged and large (0.9961 available at
E=2/B=64 against 0.9646 reached), so the information needed to choose entries
exists -- it simply is not in any query-independent description of where the
entries are. That is precisely B's hypothesis, and it is now the only one of the
three with evidence pointing at it.

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

1. **B (probe then select).** The only remaining hypothesis with evidence behind
   it: static information is now measured to carry no usable signal, while the
   per-query headroom is 3 recall points at E=2. Resume already works and is
   tested; frontier scores, survivor selection and the losing-probe merge are
   what is missing.
2. **The multi-branch cost structure.** Every multi-entry row still pays 1.5-1.9x
   the time of the wide-queue row. Shared visited across branches is listed in
   the document as a later optimisation; the measurements say it is on the
   critical path. B cannot win on the Pareto until this is addressed, however
   good its selection is.
3. C (fused greedy) is on hold. It makes selection cheaper, and selection is not
   currently where anything is lost.
