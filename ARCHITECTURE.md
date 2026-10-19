# gpuann — algorithms and architecture

Every design choice named here was settled by a measurement, and the number that
settled it is quoted next to it. `README.md` has the full tables; this file is
the structure and the reasoning.

## 0. Names

Four knobs carry the search, and they are named the same in the kernel, the
Python API, the benchmark flags and here:

| | meaning | typical |
|---|---|---|
| **V** | entries per query — branches when they are independent, lineages when they share a state | 16-64 |
| **L** | queue width: how many evaluated nodes a branch keeps alive | 16 (fan-out), 64-256 (shared) |
| **D** | graph out-degree, the index's edge budget | 32-96 |
| **B** | expansion budget per branch | 16 (nominal; `eps` usually stops it sooner) |

`eps`, the retirement slack, is the fifth and the only one ever tuned.

In the kernel the vector dimension is `dim`, because `D` is the degree.

---

## 1. The shape of the system

```
offline                                     online, per query
───────                                     ──────────────────
X  (fp32; fp16 is an ablation)
│
├─ build_graph.py ──────► G, degree[N]      catalog scan    P distances -> E entries
│    exact top-C candidates                 │
│    DiskANN robust prune (alpha)           ├─ init_states  seeds -> queue, visited, cache
│    reverse edges + re-prune               │
│    LID degree allocation                  ├─ advance      expansions; resumable;
│                                           │               retires lineages
├─ place_entries.py ────► pool[P]           │
├─ build_ivf_rabitq.py ─► 1-bit index       └─ merge        dedup -> top-k
│
└─ diagnostics: phi_h, phi-hat_1, H0 oracle, hard_decile
```

The search engine is one CUDA translation unit (`gpuann/csrc/local_search.cu`)
with two entry points, `init_states` and `advance`. All state is in
caller-owned tensors, so a search can be stopped, inspected and resumed, and a
Python reference can be compared to it line for line.

---

## 2. State

Per **(query, branch)**:

| array | type | meaning |
|---|---|---|
| `ids` | int32 [Q,J,L] | queue: best-m evaluated so far, sorted by (distance, id) |
| `dist` | float32 [Q,J,L] | their distances |
| `expanded` | uint8 [Q,J,L] | whether each entry's neighbours have been walked |
| `lineage` | uint8 [Q,J,L] | which seed this entry descends from |
| `visited` | int32 [Q,J,H] | open-addressed; keeps this branch's queue free of duplicates |
| `counters` | int32 [Q,J,6] | expansions, distance evals, hash fails, exhausted, cache hits, claims |

Per **query**, shared by all of its branches:

| array | type | meaning |
|---|---|---|
| `cache_id` / `cache_d` | int32 / float32 [Q,Hc] | distances anyone has computed |
| `bound` | float32 [Q] | best k-th distance any branch has reached |

### Invariants, stated once

- the queue is **best-L over everything evaluated**, ordered by `(distance, id)`;
  the expanded flag and the lineage travel with their entry through every merge;
- a branch stops on **budget spent**, **no retained candidate unexpanded**, or
  **every lineage retired** — never because a round left the queue unchanged;
- the budget counts **node expansions**; distance evaluations are a separate
  counter and are never substituted for it;
- `advance(a)` then `advance(b)` equals `advance(a+b)`;
- a full `visited` table raises; a full distance cache silently recomputes,
  which is safe because a recomputed distance is bitwise equal to a cached one.

---

## 3. The search kernel

One CTA per (query, branch), 128 threads. Per expansion:

1. **choose p** — under round-robin, the nearest unexpanded entry *of the
   lineage whose turn it is*, skipping retired lineages.
2. **expand** — read `G[p]`; for each neighbour the branch's `visited` decides
   whether it is new to this queue, and the query's cache decides whether its
   distance has to be computed at all.
3. **sort** the `next_pow2(D)` candidate slots by `(distance, id)`, bitonic.
4. **merge** the two sorted runs into the best m, one output slot per thread by
   merge-path binary search.

Distances are computed by a **team** of TEAM lanes (4/8/16/32), each loading 16
bytes at a time — 4 floats or 8 halves — with a strided per-lane sum folded by a
halving shuffle tree. `refcpu.py` replays that order exactly, and the kernel is
built `-fmad=false`, so CPU/GPU agreement is asserted **bitwise** rather than
within a tolerance. Sub-warp teams diverge, so every shuffle names its own
team's lane mask.

---

## 4. The algorithms

### 4.1 Multi-seed shared state
S entries share one queue, one visited table and one budget; every entry carries
its seed's lineage. The alternative — S independent branches — has each
re-evaluate the others' nodes: 17.9 distances per expansion against 28.1.

### 4.2 Round-robin over lineages
Each lineage takes turns. Without it the nearest seed's basin takes every
expansion and the extra seeds do nothing: sharing the state with nearest-first
selection recovered **none** of the gain (0.9054 against 0.9046 for one seed),
taking turns recovered all of it (0.9150).

### 4.3 Retirement against a shared bound
A lineage retires when its own best unexpanded candidate lies beyond
`(1 + eps)` times the current k-th best distance. The radius is a property of
the **query**, not the branch — `bound[q]`, updated by `atomicMin` on the float's
bit pattern, since distances are non-negative and integer order is float order.

A branch with a 16-slot queue cannot tell that it is wasting its time; the query
can. Measured at Q=64, V=100, recall 0.9953: 2.53ms using the branch's own
radius, **1.95ms** using the shared one.

`eps` is a **travel allowance**, not a tolerance. The tail of a search is arrival
at a fragment it has not reached, not mining one it has — 47% of the
ground-truth documents arriving after expansion 96 are the first of their
fragment, against 14% before it — so a rule with no slack cuts the hardest-won
documents.

### 4.4 The distance cache
A distance belongs to a (query, node) pair, not to a branch. The per-branch
`visited` still keeps each queue unique; the arithmetic goes through a per-query
table in L2, volatile, claimed with `atomicCAS`, probed for a bounded 32 steps.
At Q=64, V=100 it removes 86% of the distance work (71,963 → 10,212).

No CTA ever waits on another. Cross-CTA spinning on a GPU can deadlock, since
co-residency is not guaranteed, so the probe gives up and recomputes instead.

### 4.5 Exclusive territory
In exclusive mode a node another branch has claimed is **skipped outright**
rather than entered with its cached distance.

The reason is a measurement: of a query's 64 nearest pool entries, 97.7% recover
at least one ground-truth document, but **94% of entry pairs recover overlapping
fragments** (entry similarity predicts disjointness at AUC 0.606). The
candidates funnel into one basin, so no selector can pick a covering subset out
of them — which is why coverage sketches, spread, MMR, catalog density and a
pool hierarchy all returned nothing. The diversity has to come from the
trajectories.

At V=32 it takes recall 0.9734 → 0.9981 while the distance count goes **up** 2.5x,
which is the mechanism working: branches that cannot take the near nodes keep
going outward instead of converging.

### 4.6 Graph construction
1. **candidates** — exact top-C by a chunked sweep. O(N²); NN-descent is where
   this goes when N stops allowing it.
2. **robust prune** — DiskANN's rule: keep v unless an already-kept w satisfies
   `alpha * d(w,v) <= d(u,v)`. Larger alpha keeps more. Swept: 1.0 → varphi_2
   2.704, **1.2 → 1.479**, 1.5 → 2.011, 2.0 → 2.206 (at 2.0 nothing is pruned
   and it is a kNN graph again).
3. **reverse edges** — every kept edge is offered back to its target and each
   node re-prunes the union. Without it a node nothing points at is unreachable.
4. **LID degree allocation** — the Levina–Bickel/Amsaleg MLE over the candidate
   distances the builder already has. Degree by LID **rank** between 0.5x and
   1.5x of the target with the mean pinned, so the index does not grow; edges
   move to where the neighbourhood is locally high-dimensional and a fixed
   degree covers less of it. varphi_2 1.539 → 1.479 at equal edge count.

Degree is itself a recall–QPS knob and should not be matched to anyone else's:
at V=32/eps=0.05, deg 32/48/64/80 gives 0.9684/0.9888/0.9946/0.9966 at
126k/82k/67k/60k QPS.

### 4.7 Entry selection
The catalog is P offline nodes (no query, no GT); online, score the pool and
take the E nearest. Everything cleverer was measured and lost — coverage over a
reach sketch, near-but-spread max-min, MMR with the trade-off swept, denser
catalogs, a graph over the pool.

The bound that explains it: hand the search the query's **exact top-E** and at
V=1 that is worth 0.003 over a pool node at rank 18. **Proximity is worth almost
nothing.** The whole gain appears as E grows (0.9818 at V=8 against 0.9412)
because the exact top-8 contains one member of every fragment by construction —
query-specific information no placement can supply.

The scan still earns its 0.93ms: it buys a cheaper search rather than recall
(3,528 distances against 3,930 for free random entries, 38.1ms against 40.0).

### 4.8 IVF + RaBitQ (separate index, for comparison)
`build_ivf_rabitq.py` / `csrc/rabitq.cu`. Per vector: `sign(R(x - c))` as D bits,
`||x - c||`, and `<o_bar, o_res>`. Then

    <q, x> ~= <q, c> + ||x-c|| * ( <Rq, o_bar> / err )

The rotation is orthogonal, so `Rq` does not depend on the list and is computed
once per query rather than once per probe. `<Rq, o_bar>` needs no multiplies:
with `o_bar = +-1/sqrt(D)` it is `(2 * sum_{set bits} Rq_i - sum_i Rq_i)/sqrt(D)`,
so a nibble table turns the scan into D/4 shared lookups over D/8 bytes of code,
against D multiply-adds over 4D bytes for the exact distance — 128 bytes per
vector instead of 4096 at d=1024. The query is never quantised; that asymmetry
is what makes one bit usable at all.

---

## 5. What is measured, and how

- **fp32 is the comparison**; fp16 is an ablation and is labelled wherever used.
  It matters: CAGRA gains 3.7x from fp16 where we gain 1.6x, so a mixed-dtype
  comparison reverses the result.
- **ground truth is exact fp32**, recomputed here, never read from a file.
- **val tunes, test reports.** Only `eps` was ever tuned, on val.
- **matched recall**, always.
- **batch size is part of the measurement.** Q*V CTAs against the SM count (188
  here) decides whether extra parallelism is free; the crossover is near Q=256.
  A small subset measured as its own batch reports occupancy, not difficulty.
- **nobody is held to anyone else's index size.** Each system sweeps its own
  graph degree and its own search knob.

### Diagnostics
- **varphi_h** — components of a query's exact top-k under mutual h-hop
  reachability: how many pieces the graph cut the answer into.
- **phi-hat_1** — the index-free screen, `min(rank_i(j), rank_j(i)) <= d`, one
  corpus sweep. Its threshold *is* the out-degree budget.
- **H0 oracle** — exact 2^k DP over which entries recover which GT slots: the
  ceiling on entry selection.
- **hard decile** — the lab's definition: lowest mean recall over an effort
  sweep on an independently built selector index.

---

## 6. Limits

| | |
|---|---|
| queue width | L <= 256 (`MAXL`) |
| graph degree | D <= 128 (`MAXD` == block size; the candidate sort needs it) |
| entries per shared state | V <= 128 (init sort buffer), V <= 256 (lineage is uint8) |
| branches (fan-out) | no kernel limit; memory, and Q*E against the SM count |
| visited table | power of two, load factor <= 0.5, overflow raises |
| distance cache | power of two; overflow recomputes rather than failing |
| RaBitQ scan | D <= 1024 and D % 32 == 0 (nibble table is 16 KiB of shared) |
| metric | inner product on normalised vectors, or squared L2 |

### Reproducibility

Everything is bitwise reproducible and bitwise equal to the CPU reference
**except** two flags, both off by default:

- **shared bound** — which branch retires depends on which CTA lowered the bound
  first. Measured drift over 6 runs: ids identical, recall sd 0.00000.
- **exclusive territory** — which branch claims a node depends on CTA order.
  Measured: ids differ every run, recall sd 0.00014.
