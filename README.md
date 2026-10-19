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

`msmarco-1m` first 100K (d=1024, L2-normalised, IP), exact kNN graph deg=64,
3490 validation queries, k=10, m=16. Nothing is tuned; the test half is untouched.

Baseline arms, total budget fixed at 128 node expansions per query:

| arm | E | B/branch | recall@10 | dist/query | batch ms | QPS |
|---|---|---|---|---|---|---|
| single   | 1 | 128 | 0.7870 | 670 | 5.60 | 624k |
| distance | 2 | 64  | 0.8236 | 1373 | 8.49 | 411k |
| distance | 4 | 32  | 0.8511 | 2806 | 13.46 | 259k |
| distance | 8 | 16  | 0.8690 | 4784 | 17.65 | 198k |
| distance | 16 | 8  | 0.8720 | 5754 | 17.66 | 198k |
| random   | 8 | 16  | 0.8417 | 5250 | 22.34 | 156k |
| random   | 16 | 8  | 0.8188 | 6203 | 24.21 | 144k |

At E=1 the search exhausts after ~19 expansions of its 128: with m=16 the queue
runs out of unexpanded candidates long before the budget does. The binding
constraint at low E is the queue width, not the budget.

H0 oracle, P=512 random pool seeds, 16 expansions per branch:

| | recall@10 |
|---|---|
| one pool seed, alone | 0.6188 |
| best single seed in the pool | 0.8834 |
| best 2 / 4 / 8 seeds (exact DP) | 0.9434 / 0.9718 / 0.9753 |
| the whole 512-seed pool | 0.9753 |

**The gap worth working on: 0.869 measured against 0.975 available**, at the
same E=8 and the same per-branch budget. Choosing eight entries well is worth
about 10 recall points here, and a perfect selector would reach the pool's own
ceiling. Eight seeds already exhaust what 512 can cover, so E beyond 8 buys
nothing — it is selection, not pool size, that is left on the table.

Caveat: all of it is on the kNN stand-in graph, and the pool is random nodes,
not the region representatives A specifies.

## Running it

```bash
python offline/prep_data.py                      # -> /dev/shm/gpuann/msmarco-1m-100k
python tests/test_engine.py                      # 8/8
python bench/entry_arms.py  --out results/...
python offline/oracle.py    --out results/...
```

`GPUANN_DATA` relocates the corpus (default `/dev/shm`, because `/workspace` is
quota-limited). The CUDA extension builds on first import and caches.

## Next

1. A real search graph: cuVS CAGRA (also gives the `cagra.from_graph` native
   baseline the document asks for) or a CAGRA-style prune written here. Every
   recall number above is provisional until this lands.
2. Region metadata for A: METIS k-way on the symmetrised graph, representatives,
   reach sketch.
3. B: probe, frontier scores, resume by state index. The resume half already
   works and is tested.
