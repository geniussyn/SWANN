# gpuann: architecture and algorithms

What is built, how the pieces fit, and what each algorithm actually does. Every
design choice named here was settled by a measurement that is in `README.md`;
this file is the structure, that file is the evidence.

---

## 1. The shape of the system

```
offline                                  online (per query)
───────                                  ──────────────────
X  (fp32 or fp16, L2-normalised)
│
├─ build_graph.py ──► G, degree[N]       catalog scan   P distances, pick entries
│    candidates                          │
│    robust prune                        ├─ init_states   seeds → queue + visited
│    reverse edges                       │
│    LID degree allocation               ├─ advance       expansions, resumable
│                                        │
├─ place_entries.py ──► pool[P]          └─ merge         dedup → top-k
│
├─ phi_h.py / phihat.py ──► diagnostics
└─ hard_decile.py ──► difficulty stratum
```

The search engine is one CUDA translation unit (`gpuann/csrc/local_search.cu`)
with two entry points, `init_states` and `advance`, and all of its state in
caller-owned tensors. Nothing is hidden in the kernel between calls, which is
what makes the search resumable and the tests able to compare against a Python
reference line for line.

---

## 2. Search state

Per **(query, branch)**:

| array | type | meaning |
|---|---|---|
| `ids` | int32 [Q,J,m] | the queue: best-m evaluated so far, sorted by (distance, id) |
| `dist` | float32 [Q,J,m] | their distances |
| `expanded` | uint8 [Q,J,m] | whether each entry's neighbours have been walked |
| `lineage` | uint8 [Q,J,m] | which seed this entry descends from |
| `visited` | int32 [Q,J,H] | open-addressed set: keeps the queue free of duplicates |
| `counters` | int32 [Q,J,6] | expansions, distance evals, hash failures, exhausted, cache hits, cache claims |

Per **query**, shared by all of its branches:

| array | type | meaning |
|---|---|---|
| `cache_id` / `cache_d` | int32 / float32 [Q,Hc] | distances already computed, by any branch |
| `bound` | float32 [Q] | best k-th distance any branch has reached |

Fixed semantics, stated once and not restated by any variant:

- the queue is **best-m over everything evaluated**, ordered by `(distance, id)`;
  the `expanded` flag and the lineage travel with their entry through every merge;
- a branch stops on **budget spent**, **no retained candidate unexpanded**, or
  **every lineage retired** — never because a round left the queue unchanged;
- the budget counts **node expansions**; distance evaluations are a separate
  counter and are never substituted for it;
- `advance(a)` then `advance(b)` equals `advance(a+b)`;
- a full `visited` table raises. It never drops a node quietly.

---

## 3. The search kernel

One CTA per (query, branch), 128 threads. Per expansion:

1. **choose p** — the nearest unexpanded queue entry. Under round-robin, the
   nearest unexpanded entry *of the lineage whose turn it is*.
2. **expand** — read `G[p]`, and for each neighbour: the branch's `visited`
   decides whether it is new to this queue; the query's distance cache decides
   whether its distance has to be computed at all.
3. **sort** the `next_pow2(deg)` candidate slots by `(distance, id)`, bitonic.
4. **merge** the two sorted runs into the best m, one output slot per thread by
   merge-path binary search.

Distances are computed by a **team** of TEAM lanes (4/8/16/32). Each lane loads
16 bytes at a time — 4 floats or 8 halves — and the reduction is a strided
per-lane sum folded by a halving shuffle tree. `refcpu.py` replays that exact
order in float32, and the kernel is built `-fmad=false` so the multiply and the
add round separately; agreement with the CPU reference is therefore asserted
**bitwise**, not within a tolerance.

Sub-warp teams diverge from each other, so every shuffle names its own team's
lane mask rather than the full warp.

---

## 4. Algorithms

### 4.1 Multi-seed shared state

S entries share one queue, one visited table and one budget; every queue entry
carries its seed's lineage. The alternative — S independent branches — makes
each re-evaluate the others' nodes and gives each a budget too small to go deep.

### 4.2 Round-robin expansion

Each lineage takes turns being expanded. Without it the nearest seed's basin
consumes every expansion and the extra seeds do nothing at all: sharing the state
with nearest-first selection recovers *none* of the gain, taking turns recovers
all of it.

### 4.3 Retirement, and the shared bound

A lineage retires when its own best unexpanded candidate lies beyond
`(1 + eps)` times the current k-th best distance. `eps` is a **travel allowance**:
the tail of a search is arrival at a fragment it has not reached, not mining one
it has, so a rule with no slack cuts the hardest-won documents.

The radius is a property of the **query**, not the branch: `bound[q]`, the best
k-th distance any of the query's CTAs has reached, updated by `atomicMin` on the
float's bit pattern (distances are non-negative, so integer order is float
order). A branch with a 16-slot queue cannot tell that it is wasting its time;
the query can.

*This is the only part of the engine that is not reproducible bit for bit*, since
which branch retires depends on which CTA lowered the bound first. Off by default.

### 4.4 The distance cache

A distance belongs to a (query, node) pair, not to a branch. The per-branch
`visited` still keeps each queue unique; the arithmetic goes through a per-query
table in L2, volatile, slots claimed with `atomicCAS`, probed for a bounded 32
steps.

The property that makes it safe: **a recomputed distance is bitwise equal to a
cached one**. A miss, a lost race or a full table costs work and never
correctness — which is why the probe may give up, and why no CTA ever waits on
another (cross-CTA spinning on a GPU can deadlock, since co-residency is not
guaranteed).

### 4.5 Graph construction

1. **candidates** — exact top-C by a chunked sweep. O(N²); fine to ~1M, and the
   place NN-descent goes when it stops being.
2. **robust prune** — DiskANN's rule: keep v unless an already-kept w satisfies
   `alpha * d(w,v) <= d(u,v)`. Larger alpha keeps more. Swept: 1.2 is right.
3. **reverse edges** — every kept edge is offered back to its target, and each
   node re-prunes the union. Without it a node nothing points at is unreachable.
4. **LID degree allocation** — the Levina-Bickel/Amsaleg MLE over the candidate
   distances the builder already has. Degree is assigned by LID *rank* between
   0.5x and 1.5x of the target with the mean pinned, so the index does not grow;
   the edges move to where the neighbourhood is locally high-dimensional and a
   fixed degree covers less of it.

### 4.6 Entry selection

The catalog is P offline nodes (no query, no GT). Online: score the pool, take
the E nearest. Measured against coverage-greedy over a reach sketch (A) and
against near-but-spread max-min: **plain nearest wins both**. The lever that does
work is catalog *density*, which also makes the search itself cheaper.

### 4.7 Diagnostics

- **varphi_h** (`offline/phi_h.py`) — components of a query's exact top-k under
  mutual h-hop reachability. Says how many pieces the graph cut the answer into.
- **phi-hat_1** (`offline/phihat.py`) — the index-free screen: `min(rank_i(j),
  rank_j(i)) <= d`, components. One corpus sweep, no index. Its threshold *is*
  the out-degree budget.
- **H0 oracle** (`offline/oracle.py`) — exact 2^k DP over which entries recover
  which ground-truth slots: the ceiling on entry selection, which no selector can
  pass.
- **hard decile** (`offline/hard_decile.py`) — the lab's definition: lowest mean
  recall over an effort sweep on an independent selector index.

---

## 5. Measurement protocol

- **fp32 is the comparison**, fp16 is an ablation and is labelled wherever used.
- **ground truth is exact fp32**, recomputed here, never taken from a file.
- **val tunes, test reports.** Only `eps` was ever tuned, on val.
- **matched recall**, always: a latency without the recall it bought means nothing.
- **batch size is part of the measurement.** Q×E CTAs against the SM count
  decides whether extra parallelism is free; a small subset measured as its own
  batch reports occupancy, not difficulty.
- **the same graph, or each its own** — stated per table, never mixed silently.

---

## 6. Known limits

| | |
|---|---|
| queue width | m <= 256 (`MAXM`) |
| graph degree | deg <= 128 (`MAXDEG`, == block size, the sort needs it) |
| seeds per shared state | S <= 128 (init sort buffer), S <= 256 (lineage is uint8) |
| branches (fan-out) | no kernel limit; memory, and Q*E against the SM count |
| visited table | power of two, load factor <= 0.5, overflow raises |
| distance cache | power of two; overflow is not an error, it recomputes |
| metric | inner product on normalised vectors, or squared L2 |
