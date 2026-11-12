# SWANN — algorithms and architecture

*(package and directory are still `gpuann`.)*

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
│    flat degree (see 4.6)                  ├─ advance      expansions; resumable;
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
3. **sort** the `next_pow2(deg[p])` candidate slots by `(distance, id)`, bitonic.
4. **merge** the two sorted runs into the best m, one output slot per thread by
   merge-path binary search.

An expansion is bounded by **p's own degree**, not by the adjacency matrix's
width. A pruned row is not full -- 37.5 edges in a 96-wide row on GIST -- and
handed `deg[N]` from `engine.graph_degrees(G)` the kernel stops the graph read,
the cache probes, the expansion rounds and the candidate sort at `deg[p]`. It
skips exactly the slots it was already rejecting, so the output is **bitwise
identical** either way, which is a test rather than a claim. The shared-memory
carve-up follows `max(deg)` instead of the width for the same reason.

It is worth about 1% on the graphs here, and the measurement that says why is
`bench/degree_width.py`: of the 2.5 points a 1.5x-over-wide row costs, the
degree bound recovers 1.8 and leaves 0.7, and at 2x it recovers 6.1 of 13.6.
**The padding tax is mostly the row stride, not the loop bounds** -- bytes moved
for a 384-byte row holding 240 bytes of edges -- and only true compaction (CSR)
reaches that. At 0.7% on the shipped widths it is not worth writing, so CSR is
not here and this number is the reason.

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


**A slot publishes in two steps, and that is not an optimisation.** The table is
never cleared -- each slot carries its generation, so an earlier search's slot
reads as empty -- and not clearing it removed the sentinel that made the
claim/store race benign. A cleared table held `FLT_MAX` in every distance, so a
branch probing a slot between another branch's claim and its store read
`FLT_MAX` and computed the distance itself; an uncleared table holds *an earlier
search's distance* there, and that branch took it. So a slot carries a reserved
bit: the claimer takes it with `CRESV` set, writes the distance, fences, and only
then clears `CRESV`. A reader trusts the distance only when `CRESV` is clear and
treats a reserved slot as "another branch owns this node", which is all exclusive
mode needs and is a miss otherwise.

Measured cost of getting it wrong, at V=32 with claiming off: recall 0.9728 ->
0.7296. With it right the table is what this section claims: recall identical to
four decimal places, distance evaluations 0.37x, throughput 1.66x.
`test_shared_cache_is_free_of_consequence` is the check, and it was failing.

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
Degree is **uniform**, and the allocation that used to be step 4 here is gone.
Local intrinsic dimension -- the Levina–Bickel/Amsaleg MLE over the candidate
distances the builder already has, degree by LID rank between 0.5x and 1.5x of
the target with the mean pinned -- moved varphi_2 1.539 → 1.479 at equal edge
count and moved **recall by +0.0005**, inside the noise. The graphs named
`G_d64.npy` are the ones it built and are 96 wide for a mean degree near 51.

It was not the kernel that condemned it: a 96-wide matrix costs 2.5% more than a
64-wide one holding the same edges, and 0.7% once the kernel bounds an expansion
by `deg[p]` (`bench/degree_width.py`). +0.0005 does not hide behind 0.7%. The
verdict stands on the graph, and the door is open for an allocation with a
better signal than LID.

Degree is itself a recall–QPS knob and should not be matched to anyone else's:
at V=32/eps=0.05, deg 32/48/64/80 gives 0.9684/0.9888/0.9946/0.9966 at
126k/82k/67k/60k QPS.

### 4.6a Reciprocity, and why phi is not a build target

`phi_1` at h=1 is *exactly* graph reciprocity: `offline/phi_h.py` joins two
answers when `reach[a,b] and reach[b,a]`, and a one-hop reach is the neighbour
set. So the build can move it directly, and the robust prune is what spends it
-- the prune keeps a usable detour instead of the edge, which keeps navigation
and drops mutuality. Measured on MS MARCO 1M: ours 59.5% reciprocal at phi
3.083, CAGRA's d=64 82.1% at 2.784. **CAGRA's graph fragments the answer less
than ours and searches worse.**

Rows are stored dense at width 96 for a mean out-degree of 52.1, so 46% of the
allocated slots hold -1 and an edge put in one **costs no memory**.
`offline/reciprocate.py` fills them with the reverse of kept one-way edges,
shortest first, then re-sorts each row: 90.8% reciprocal, phi **2.149**, degree
67.0. That is a larger move than any build variant in the sweep.

Recall does not follow it. Two controls, test split, 2000 queries, V=32 L=128
B=512, full coordination, read at each graph's ceiling:

| graph | deg | edges | recip | phi_1 | ceiling | ms |
|---|---|---|---|---|---|---|
| G_d64 | 52.1 | 52.1M | 59.5% | 3.083 | 0.9886 | 33.9 |
| G_recip52 | 52.1 | 52.1M | 65.5% | 2.698 | **0.9805** | 31.2 |
| G_recip | 67.0 | 67.0M | 90.8% | **2.149** | 0.9917 | 40.2 |
| G_fill_bridge | 66.9 | 66.9M | 61.4% | 2.808 | 0.9916 | 40.5 |
| G_fill_near | 96.0 | 96.0M | 63.0% | 2.202 | 0.9920 | 44.2 |
| G_cagra_d64 | 64.0 | 64.0M | 82.1% | 2.784 | 0.9765 | 36.9 |

- **Matched edges.** `G_recip52` is the union re-pruned to the original degree
  by pure distance. phi falls 3.083 -> 2.698 and the ceiling falls with it,
  0.9886 -> 0.9805. Replacing a robust-prune edge with a shorter reciprocal one
  buys mutuality and loses navigability, and the second is worth more.
- **Matched degree.** `G_recip` (phi 2.149) against `G_fill_bridge` (phi 2.808)
  at 67.0 and 66.9: a 24% gap in phi for **+0.0001** of ceiling, and no point
  above recall 0.96 differs by more than 0.0006.

What moves the ceiling is the edge count, and it is nearly exhausted: 52.1 -> 67
buys +0.0031, 67 -> 96 buys +0.0003 for 43% more edges.

`G_recip` is still worth shipping, on the memory argument rather than the phi
one: the row width does not change, so +0.0031 of ceiling and 1.11x at recall
0.95 (9.97 against 11.10 ms) are free. Its one advantage over bridge-fill is at
the cheap end -- +0.0057 at 7.5 ms -- and it converges to nothing by 0.99.

So phi_1 stays what the paper says it is, an instrument for predicting where
depth saturates, and it is not a thing to build against. `bench/escape_cost.py`
and `results/degree/recip_vs_fill.json` hold the measurements.

### 4.6b Whose graph is the better graph

Not CAGRA's. `bench/graph_head_to_head.py` holds everything but the graph fixed
-- same corpus, queries, entry pool, budget and **the same search kernel** -- so
a difference is the builder and not the engine. At V=32, L=128, B=512, 2000 val
queries, at the top of an `eps` sweep; the sweeps are in
`results/degree/*_head_to_head.json`:

| corpus | d | ours: deg / recall@10 / ms | CAGRA: deg / recall@10 / ms |
|---|---|---|---|
| MS MARCO 1M | 1024 | 51.6 / **0.9825** / **29.8** | 64.0 / 0.9710 / 35.4 |
| FEVER 1M | 1024 | 61.7 / **0.9534** / **41.4** | 64.0 / 0.9234 / 43.7 |
| GloVe 1.18M | 100 | 61.2 / **0.9151** / **10.5** | 64.0 / 0.8810 / 11.2 |
| GIST 1M | 960 | 37.5 / 0.9680 / 14.1 | 64.0 / 0.9682 / 14.1 |

Ours is ahead on recall at every point of the sweep on three of the four, and on
GIST it draws at the top (0.9680 against 0.9682 at 14.1ms) after leading lower
down (0.8578 against 0.8338 at eps=0.05). It does all of it on **fewer edges**:
GIST is the sharpest case at mean degree 37.5 against 64 -- 41% fewer -- and
still touches 5,226 adjacency entries per query against 9,088 for the same
recall, on 22% fewer distance evaluations.

So the 1.6x native CAGRA holds is its **kernel**, not its graph, and the builder
is not where that gap is. Note also that varphi_2 ranks the two graphs the other
way -- 1.479 for ours against 1.328 -- so varphi_2 is a diagnostic and not an
objective to build against.

### 4.7 Entry selection
The catalog is P offline nodes (no query, no GT); online, score the pool and
take the V nearest.

**Which V of the P barely matters.** Coverage over a reach sketch,
near-but-spread max-min, MMR with the trade-off swept and a graph over the pool
were all measured and none beat taking the nearest. The bound that explains it:
hand the search the query's **exact top-V** and at V=1 that is worth 0.003 over
a pool node at rank 18. **Proximity is worth almost nothing.** The whole gain
appears as V grows (0.9818 at V=8 against 0.9412) because the exact top-8
contains one member of every fragment by construction — query-specific
information no placement can supply.

**P itself matters, and the shipped 4096 was too small.** It is a one-sided
knob: at a fixed `eps` a larger catalogue adds recall and costs only the scan
(MS MARCO at eps=0.03, P 4096 -> 32768: recall 0.9792 -> 0.9859 for 1.45 ms of a
27 ms batch). Measured at matched recall, raising it is worth 1-19% on MS MARCO,
18-26% on NQ, 20-22% on OpenAI-1536 and 10-19% on OpenAI-3072
(`bench/catalogue_size.py`).

What sets the optimum is **the operating point**, not d and not N. Within one
corpus the argmin moves two orders of magnitude as the target approaches that
corpus's ceiling, because near the ceiling `eps` buys recall expensively and P
buys it cheaply — GloVe wants 2048 at recall 0.930 and 65536 at 0.965. Reading
it off d or N gives the wrong answer; reading it off the **scan share** gives
the right one, and `offline/choose_P.py` is that rule:

> P* is the largest P whose catalogue scan stays under 20% of the total.

Every measured argmin sits at a scan share of 3-17%, and a larger P starts
losing near 25%. The rule needs the device, the corpus and `eps`, and **no
queries and no ground truth** — timing does not depend on whether the answers
are right, so random corpus vectors stand in as inputs. Validated against the
argmin of a real validation split: exact on MS MARCO, FEVER, GloVe and NQ, two
ladder steps high on GIST at a cost of 5-7%.

At N near 1M that lands on **P = 65536, about N/15**. P is capped at N/8 on
purpose and not for speed: at P=131072 over 995K the catalogue covers 13% of the
corpus and already holds 13% of each query's top-k, which makes the
configuration a hybrid of a graph search and a flat scan and has to be named as
one. The cap costs about 1% on the OpenAI corpora, whose unconstrained argmin is
one step above it.

Two things that look like they should work and do not, both recorded in
`offline/choose_P.py`. Documents cannot stand in for queries when choosing P —
not because their answers are less fragmented (phi_1 3.261 against 3.018, they
are more so) but because **the graph was built to make exactly that set
reachable**: a document's own adjacency list already holds 63.7% of its exact
top-10 against 50.0% for the best single answer node of a real query, so recall
comes out 0.9995-1.0000 at every P and ranks nothing. And a stand-in query is a
corpus node, so the search returns it while its ground truth excludes it, and
recall reads exactly 0.9000 everywhere until that is handled.

The scan still earns its keep on the other axis too: it buys a cheaper search
rather than recall (3,528 distances against 3,930 for free random entries,
38.1ms against 40.0).

### 4.7b Stored precision: fp16 is the operating point, and narrower is open

**fp16 is the default.** Storing the corpus in half costs $-0.0002$ to $-0.0009$
recall and returns $1.57$--$1.76\times$ at matched recall on MS MARCO; against
native CAGRA in fp16 on both sides the ordering does not reverse (3.37x at
0.989, 3.77x at 0.992). fp32 remains the *comparison* precision -- every
head-to-head is run at matched precision and both are reported -- but it is no
longer the shipped operating point.

**Narrower than fp16 works, and the reason it works is not the obvious one.**
`bench/stored_precision.py` quantises the corpus to B bits per dimension with a
fixed random rotation and asymmetric scoring (the query stays fp32), and then
rescores the best N of the final queue against the fp32 vectors. At eps=0.11 on
MS MARCO, where fp16 reaches 0.9947:

| bits | B/vec | raw | rescore 16 | rescore 64 |
|---|---|---|---|---|
| 8 | 1024 | 0.9901 | **0.9952** | 0.9952 |
| 6 | 768 | 0.9754 | **0.9948** | 0.9948 |
| 5 | 640 | 0.9550 | **0.9946** | 0.9948 |
| 4 | 512 | 0.9168 | 0.9913 | **0.9949** |
| 3 | 384 | 0.8249 | 0.9489 | **0.9943** |

The quantiser's ceiling is a **ranking** problem and not a reaching one, down to
3 bits: the walk arrives at the right nodes on a vector a tenth the width and
only orders them wrong. Sixteen rescored candidates fix 5 bits and up, 64 fix 3
and 4, and 16 candidates is 64 KB a query against the search's own megabytes.
The rotation is what makes the raw column survivable at all -- it halves the
error at every width -- and it is free at serve time, one $d \times d$ multiply
of the query folded into the catalogue scan.

Without the rescore the same widths lose $0.005$ at 8 bits and $0.17$ at 3, so
the raw column is the trade a reader would assume and it is the wrong one.

**What is worth implementing is 4 bits, not 7.** A nibble is two values per byte
and 32 per 16-byte load; 3 bits needs a shift-and-mask chain for 25% fewer bytes
than that, and 7 bits needs one for 12.5% fewer than a byte-aligned int8. The
speed is **not** measured -- a dequantised vector moves as many bytes as fp32 and
no narrow kernel exists yet. Fitting $T = F + K \cdot \text{bytes}$ to the
measured fp32 and fp16 points projects $1.8\times$ for int8 and $2.9\times$ for
int4 over fp16, and that projection is optimistic: the per-candidate costs that
are 5% of the time at 4096 B a vector are 33--39% at 416 B (the $d=104$
measurement in section 5), so nearer $2\times$ is the number to plan against.

### 4.7c The rescore: depth 32, and the residual that does not work

The narrow arm stores two copies -- 4-bit codes to walk with (0.48 GiB at
N = 1e6, d = 1024) and an fp16 copy to rank with (1.91 GiB) -- and the second is
why a 4-bit index is *larger* than an fp16 one, 2.62 GiB against 2.15. Two
things were tried against that.

**Depth 64 was 2x too deep.** The knee is at 24 and it is saturated by 32.
Measured at 4 bits, depth 24/32/64/128:

| corpus | 24 | 32 | 64 | 128 |
|---|---|---|---|---|
| MS MARCO 1M, eps 0.16 | 0.9947 | 0.9948 | 0.9948 | 0.9948 |
| FEVER 1M, eps 0.16 | 0.9865 | 0.9866 | 0.9866 | 0.9866 |
| NQ 500K, eps 0.16 | 0.9985 | 0.9985 | 0.9985 | 0.9985 |

Depth 16 is 0.0019--0.0023 short and depth 8 collapses to 0.785 -- the narrow
score's top eight does not contain the answer. The default is now **32**, which
halves what the stage reads (131 -> 65 KB a query) for at most 0.0001. The time
it saves is smaller than the bytes suggest, 2.2 -> 1.5 ms, because the stage is
dominated by the dedup and sort over the whole queue and not by the gather; only
past 64 does the gather take over, and 128 costs 4.7 ms.

**A residual rescore corpus loses to a flat 8-bit one, and the base's clip is
why.** The base codes are already resident, so a 4-bit code of the *residual*
costs 0.48 GiB where fp16 costs 1.91. First attempt, with the shipped base
(clip 2.5 sigma): reconstruction 7.32x worse than flat 8-bit and ceiling 0.9734
against flat 8-bit's 0.9895. The cause is saturation -- 2.5 sigma is the
MSE-optimal clip for a 4-bit code used *alone*, and what it clips, the residual
has to repair at the residual quantiser's own expense. Widening the base to
4 sigma closes most of it:

| base clip | base alone (mse) | + 4-bit residual | vs flat 8-bit |
|---|---|---|---|
| 2.5 | 1.18e-05 | 6.22e-07 | 7.32x |
| 3.0 | 1.52e-05 | 1.85e-07 | 2.17x |
| **4.0** | 2.66e-05 | **1.28e-07** | **1.50x** |
| 6.0 | 5.98e-05 | 2.85e-07 | 3.36x |

and the wider base is **free** when a rescore follows: raw traversal recall
falls 0.9282 -> 0.9063, and after an fp16 rescore the two are 0.9951 and 0.9952.
The traversal only has to reach, which is the same statement as 4 bits being a
traversal format.

End to end on MS MARCO 1M at the ceiling, with the graph charged in:

| arm | footprint | ceiling |
|---|---|---|
| fp16, no rescore | 2.15 GiB | 0.9954 |
| 4-bit(2.5) + fp16 | 2.62 GiB | 0.9951 |
| 4-bit(2.5) + flat 8-bit | 1.67 GiB | 0.9895 |
| **4-bit(4.0) + 4-bit residual** | **1.19 GiB** | 0.9877 |
| 4-bit(2.5) + 4-bit residual | 1.19 GiB | 0.9734 |

So the cascade is the first narrow configuration that is actually *smaller* than
the fp16 arm -- 1.19 against 2.15 GiB -- and it costs 0.0074 of ceiling against
the shipped narrow arm. It is not shipped as the default, because the arm it
has to beat on the memory axis is IVF-RaBitQ at 0.13 GiB and it does not, and
on the recall axis it loses to fp16 at twice the size. It is a point on the
frontier and it is recorded as one. `bench/rescore_residual.py` and
`results/degree/cascade_rescore.json`.

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
| per-node degree | rows front-compacted; `max(deg)` <= the row width |
| entries per shared state | V <= 128 (init sort buffer), V <= 256 (lineage is uint8) |
| branches (fan-out) | no kernel limit; memory, and Q*E against the SM count |
| visited table | power of two, load factor <= 0.5, overflow raises |
| distance cache | power of two; overflow recomputes rather than failing |
| RaBitQ scan | D <= 1024 and D % 32 == 0 (nibble table is 16 KiB of shared) |
| metric | inner product on normalised vectors, or squared L2 |

### Reproducibility

Everything is bitwise reproducible and bitwise equal to the CPU reference
**except** the two coordination flags. Measured on MS MARCO 1M, 2000 val
queries, 4-bit, V=32, L=16, B=16, eps=0.07, 8 runs:

| cache | exclusive | bound | ids differ | recall spread |
|---|---|---|---|---|
| off | off | off | 0/7 | 0.00000 |
| on | off | off | 0/7 | 0.00000 |
| on | **on** | off | 7/7 | 0.00030 |
| on | **on** | **on** | 7/7 | 0.00050 |

The shared distance cache is free of consequence by construction and measures
that way: it changes what is computed and not what is returned. The other two
couple the CTAs on purpose — which branch claims a node, and which lowered the
bound, decide which branch retires — and the drift that buys is ±0.0005 of
recall. The paper's numbers are produced with both on, so that is the figure to
read the fourth decimal against.

**This claim was false until 2026-10-27**, and the way it was false is worth
keeping. `init_kernel` deduped a branch's seed list with the visited table:
whichever team won the `atomicCAS` for a repeated node kept its slot. The table
dedups the search itself correctly that way, but in the seed loop the slot index
*is* the lineage (`c_l[si] = si`), so the surviving *label* was whatever warp
scheduling picked. On any query holding a duplicate seed, `lineage` came out
different run to run while `ids`, `dist`, `visited` and every counter matched,
and `POLICY_ROUND_ROBIN` — the one policy that reads the label to choose a
parent — then ran a different search from the same state.

It took a long time to find because every signal pointed away from it:

- it looked like a bug in `advance`, since that is where the output moved, and
  `advance` from one cloned state is reproducible 40/40;
- `init_states` was checked for reproducibility and passed, because the check
  compared `ids`, `dist`, `expanded`, `visited` and `counters` — the five fields
  that do not move — and not `lineage`;
- it needed a sub-warp team, which reads as a shuffle or mask bug. It is not:
  with `BLOCK = 128` and the duplicate at seed indices 0 and 6, team 4 puts both
  in warp 0 (lanes 0 and 24, same instruction, hardware serialises by lane) and
  team 32 puts them in different `base` iterations with a `__syncthreads()`
  between, so both are deterministic; teams 8 and 16 straddle warps and race;
- it looked like it needed a padded graph, round-robin and L2. None of the
  three is real: comparing every field instead of `ids` alone shows all four
  graph x policy cells diverging identically on the old kernel (0/7, 2/7, 6/7,
  0/7 for teams 4, 8, 16, 32), so the team size is the only axis that matters.
  Those three conditions were artifacts of comparing one field;
- compute-sanitizer initcheck, racecheck, synccheck and memcheck are all clean,
  and correctly so — nothing is read out of bounds or without a barrier. The
  atomic is doing exactly what it was asked to do, on a question that has no
  order-independent answer.

The fix dedups the seed list **by index** instead: a node named twice is kept as
the lowest of its indices, which is a property of the seed list and of nothing
else. It costs the duplicate's distance, which it already skipped, and it
measures free — init is 0.434 → 0.435 ms at V=16 and 3.218 → 3.212 at V=128,
because the scan hides behind the seed distance at d=1024. Seeds from
`select_nearest` are distinct by construction, so no measured result moves.
`test_duplicate_seeds_keep_the_lowest_lineage` asserts the rule over all queries
and all four team sizes; on the old kernel it reports
`team 8 q2: node 5 labelled [6], want 0`.
