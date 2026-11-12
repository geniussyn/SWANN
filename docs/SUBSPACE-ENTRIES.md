# Entry selection from the query's answer subspaces

*Research memo. Started 2026-10-26. Every number here is measured in this tree;
the measurement is named next to it.*

## 0. The proposal, and the one sentence that changes it

A bi-encoder maps a query to one point. An ambiguous query -- "서울의 날씨는",
"allentown pa" -- has answers that sit in several places, each scoring on its
own directions, which is why two answers can both be near `q` and far from each
other. The proposal is to represent those places explicitly: give each vector
`D*N` extra coordinates (`D = 8`, `N` subspaces, so `1024 + 8N` for BGE) and
select entries in the augmented space.

Written out, the augmented score is

```
    s(q, p) = <q, x_p> + <a(q), b_p>,        a(q), b_p in R^r,  r = D*N
```

so the proposal says: **the gap between the catalogue scan and perfect entry
choice is a rank-r additive correction to the [Q, P] score matrix.**

That framing immediately yields the first result, and it is a negative one.

> **Proposition 1 (a linear query head is vacuous).** If `a(q) = W^T q` for some
> `W in R^{d x r}`, then
> `<q, x_p> + <W^T q, b_p> = <q, x_p + W b_p>`.
> The augmented score is an ordinary inner product in the **original** `d`
> dimensions against a re-placed pool `x'_p = x_p + W b_p`. Appending `D*N`
> coordinates buys no representational power whatsoever; it is exactly an entry
> *placement*.

Entry placement is already closed here: ten build- and selection-side
interventions were measured and all lost (ARCHITECTURE.md 4.7 -- coverage over a
reach sketch, near-but-spread max-min, MMR with the trade-off swept, a graph over
the pool, denser catalogues), and at fixed `V` every entry diverted from the
nearest ranks costs recall monotonically. So Proposition 1 says the linear
reading of the proposal has already been refuted, four times over, and the only
live version is one where **`a(q)` is nonlinear in `q`**.

This is the same bottleneck the single-vector lower bounds describe: the family
of top-`V` sets a `d`-dimensional dot-product model can return is capped through
the sign-rank of the relevance matrix (Weller et al., *On the Theoretical
Limitations of Embedding-Based Retrieval*, 2025). Adding `r` dimensions raises
the cap -- but only if the extra coordinates carry information the first `d`
do not, which a linear map of `q` by construction does not.

## 1. What perfect entry choice is worth

The prize has to be known before anything is spent chasing it.
`results/subspace/full_V*.json` holds an oracle arm -- the pool node nearest
each of the query's true answers, deduped, first `V` kept -- against the base
frontier pooled over every `V` and every `eps`:

| oracle V | recall | ms | cheapest base at that recall | speedup |
|---|---|---|---|---|
| 4 | 0.9648 | 17.2 | V=8, eps=0.12, 29.5 ms | 1.71x |
| 8 | 0.9820 | 27.9 | V=16, eps=0.12, 49.7 ms | 1.78x |
| 16 | 0.9937 | 65.7 | V=32, eps=0.3, 126.4 ms | **1.93x** |
| 32 | 0.9960 | 79.8 | unreachable at any V, eps | -- |

So perfect entry choice is worth **1.2-1.9x wall clock at matched recall**,
rising with recall, and it unlocks a recall band (>= 0.996) the base frontier
cannot reach at all. On a GPU that number is not only latency: `V` entries per
query are `V` CTAs, so halving `V` at equal recall doubles the batch that fits
in the 188 SMs before occupancy saturates.

The arithmetic agrees from the other side. `bench/basin_coverage.py`: 32 entries
chosen by distance land on only **12.4 distinct attractors**, a missed answer is
its own local maximum 85.8% of the time, and seeding on the answers themselves
returns recall 1.0000. Three fifths of the entry budget is spent arriving where
another entry already is.

## 2. Why no selector has taken it

The pool's own geometry does not expose the structure. Of a query's 64 nearest
pool entries, 97.7% recover at least one answer, but **94% of entry pairs
recover overlapping fragments**, and entry similarity predicts disjointness at
**AUC 0.606** (ARCHITECTURE.md 4.5). The candidates funnel into one basin, so no
function of the pool-side geometry can pick a covering subset out of them.

Measured again today, directly on the rank question. `bench/subspace_rank.py`
fixes an offline `B in R^{P x r}` and hands `a(q)` oracle freedom per test
query -- an upper bound on realizability -- and asks how much of the oracle entry
set lands in the top `V`:

| r | B from SVD of the val oracle sets | B random orthonormal |
|---|---|---|
| 0 | 0.453 | 0.453 |
| 32 | 0.570 | **0.777** |
| P | 1.000 | 1.000 |

A **random** basis beats the fitted one. The `r` free query-side parameters are
doing all the work; the offline document-side basis carries nothing. Combined
with the spectrum of the oracle relation (64 directions hold only 0.34 of its
Frobenius energy, and removing pool-node popularity changes that to 0.337, so it
is not low-rank), the conclusion is:

> **The `D*N` document coordinates are not where the problem is. Design `a(q)`.**

## 3. The mechanism works, and resolution is the whole story

Two arms of the *same* mechanism were already in the tree and disagreed. Both
drop `q` from a set of neighbours, PCA the residual, probe
`normalize(q +- lambda * axis)` in both signs, and take the nearest pool entry
to each probe. Only the fitting set differs:

- `bench/subspace_seed.py`, axes on the top-50 **corpus** documents a pass-1
  search returned: V=2 recall 0.9212 -> 0.9585, at 1706 extra distances.
- `bench/subspace_select.py`, axes on the top-64 **pool** entries the catalogue
  scan already holds: **null at every V**.

`bench/subspace_res.py` holds everything else fixed and moves only the fitting
set. msmarco-1m, val, 2000 queries, G_d64, L=12, B=64, eps=0.05, 3 axes, k=10:

| pool | V | base | axes on pool top-64 | axes on exact corpus top-10 | oracle ceiling |
|---|---|---|---|---|---|
| nest4k | 4 | 0.9410 | -0.0067 | **+0.0065** (p=2e-2) | +0.0055 |
| nest4k | 8 | 0.9627 | -0.0006 | **+0.0061** (p=1e-5) | +0.0094 |
| nest4k | 16 | 0.9780 | -0.0004 | **+0.0024** (p=2e-3) | +0.0051 |
| nest4k | 32 | 0.9852 | +0.0001 | **+0.0015** (p=6e-3) | +0.0025 |
| nest16k | 8 | 0.9678 | -0.0017 | **+0.0107** (p=9e-19) | +0.0126 |
| nest64k | 8 | 0.9719 | -0.0002 | **+0.0095** (p=2e-20) | +0.0154 |
| nest128k | 8 | 0.9750 | -0.0013 | **+0.0089** (p=6e-21) | +0.0140 |

Three things, and the third is the one that matters.

1. The corpus-resolution arm captures 47-85% of the entry ceiling at V >= 8 and
   **exceeds** it at V=4 (legitimately: the ceiling arm is one particular
   oracle, the pool node nearest each answer, not the true optimum). Distance
   evaluations per query are flat or slightly lower, so the recall is free in
   search work.
2. `m = 10` beats `m = 50` at every V. The subspace is estimated by the nearest
   handful; fifty dilute it.
3. **A denser catalogue does not rescue the pool-resolution arm.** From P=4096 to
   P=131072 the pool arm stays negative at every V. Catalogue density was the one
   entry-side intervention that ever won here ([[swann-catalogue-density]]), and
   it does not buy the axes. So "resolution" is not the pool's sampling density.

### 3b. The circularity, and the measurement that closes it

The exact corpus top-10 **is the ground truth** -- verified, overlap 1.0000
against `gt_val_ids[:, :10]`. So that arm fits the subspace axes on the answers
themselves and then uses them to go and find the answers.

The de-circularising arm fits the axes on the *same* exact neighbours with every
answer removed -- corpus ranks `k+1 .. k+m`, which hold no answer by
construction. nest16k, P=16384, val, 2000 queries, lambda=0.6, 2 entries per
direction:

| V | axes fitted on | answers in the fitting set | vs base | p |
|---|---|---|---|---|
| 8 | corpus top-10 | 10 of 10 | **+0.0099** | 1.6e-18 |
| 8 | corpus top-50 | 10 of 50 | +0.0032 | 8.7e-03 |
| 8 | corpus ranks 11-20 | 0 of 10 | +0.0016 | 2.2e-01 |
| 8 | corpus ranks 11-60 | 0 of 50 | +0.0006 | 8.4e-01 |
| 8 | pool top-64 | 0 of 64 | -0.0053 | 4.7e-07 |
| 4 | corpus top-10 | 10 of 10 | **+0.0093** | 3.7e-08 |
| 4 | corpus ranks 11-20 | 0 of 10 | +0.0011 | 7.2e-01 |
| 4 | corpus ranks 11-60 | 0 of 50 | -0.0013 | 5.7e-02 |

A monotone dose-response in **how much of the answer set is in the fitting set**,
and nothing left once it is removed. So:

> **The local subspace axes carry no information about where the answers are
> beyond the answers that were used to fit them.** The union-of-subspaces picture
> is a correct *description* of `A(q)` and not a *predictor* of it. Local PCA of
> the query's neighbourhood is dead as a source of `a(q)`, at every resolution
> tested: pool top-64, corpus ranks 11-20, corpus ranks 11-60.

It follows that the PRF arm's V=2 gain (0.9212 -> 0.9585) is bought by its 1706
extra distance evaluations, which find real answers, and not by its axes. That
is consistent with its own accounting: it never beat the base frontier at matched
cost at any V >= 4.

This is the sign-rank bound in concrete form, and it makes the framing stronger
rather than weaker. *Which subspaces this query's answers occupy* is not a
function of the query's neighbourhood geometry. If it is anywhere, it is outside
the embedding -- in the query text, in a query log, in a lexical signal -- which
is exactly what Proposition 1 said when it demanded a nonlinearity, and exactly
where GRIP finds it in the sparse domain (the modes are explicit terms, so
`a(q)` is free).

## 4. The ladder, and where the open rung is

| rung | fitting set | cost | result |
|---|---|---|---|
| 1 | top-64 pool entries | free (the scan has them) | null at every V, every P |
| 2 | top-50 of a pass-1 search | 1706 distances | V=2: +0.037 recall, costs more than it saves |
| 3 | exact corpus top-10 | the retrieval problem | reaches/exceeds the ceiling -- but it *is* the answers (3b) |
| 3' | exact corpus ranks 11-60 | the retrieval problem | **null** -- the rung was the answers, not the resolution |

Section 3b removes the rung that looked open. The ladder does not have a cheap
middle, because the expensive middle was not buying axes -- it was buying
answers. What survives is the prize (section 1), the diagnosis that the
bottleneck is query-side (section 2), and Proposition 1's instruction that
`a(q)` has to be nonlinear and cannot come from the pool.

Two cheap-neighbourhood indexes are nonetheless built here, and are worth one
run only because rung 2's accounting deserves to be closed properly rather than
inferred:

- **The 1-bit RaBitQ index** (`build_ivf_rabitq.py`, `csrc/rabitq.cu`). 128 bytes
  per vector at d=1024 against 4096, and the scan is `D/4` shared-memory lookups
  with no multiplies, because `o_bar = +-1/sqrt(D)` turns `<Rq, o_bar>` into a
  nibble table. The query is never quantised, and that asymmetry is what makes
  one bit usable -- which is also the lab's standing result that only asymmetric
  scoring escapes the quality frontier ([[bq-quality-frontier]]).
- **IVF with a few probes**, same index.

The axes themselves are nearly free either way: `local_axes` is an `m x m` Gram
matrix and an `eigh`, `m = 10`, and the probing costs `2c` extra catalogue scans
(`c = 3`), which at tf32 is the one matmul in this system where precision cannot
change an answer ([[swann-tf32-catalogue]]).

## 5. What this is, as a paper

The SWANN paper's related work already names the gap, in its last paragraph: *"A
multi-vector retriever is also the setting where seed placement would have
something to work with, because the extra vectors would define targets and not
only starting points."* This is that paper, with the multi-vector representation
constructed rather than trained.

The contribution is not "multi-vector retrieval" and not "diversified entry
points". It is that **the payoff of resolving a query into its answer subspaces
is an occupancy payoff, not a recall payoff**: fewer entries at equal recall is
fewer CTAs per query, and on a GPU that is batch. Nobody has measured that axis,
because nobody else separates the entry count from the traversal.

## 6. Prior work, and what to take from it

**Take the theory.**
- *On the Theoretical Limitations of Embedding-Based Retrieval* (Weller et al.,
  2025, arXiv 2508.21038). Sign-rank bounds the number of top-k sets a
  `d`-dimensional single-vector model can return; the LIMIT dataset is the
  empirical edge. This is the formal reason a single `q` cannot address a
  multi-subspace answer set, and the reason Proposition 1 bites.
- *Retrieval with Learned Similarities* (Ding et al., WWW'25, arXiv 2407.15462).
  Mixture-of-Logits: `P` pairs of low-rank embeddings with adaptive **gating**, a
  universal-approximation theorem for similarity functions, and top-k retrieval
  with error bounds. This is the proposal's architecture done properly -- the
  gate is the nonlinearity Proposition 1 demands -- and it is the right
  parameterisation to try for `a(q)`.
- Subspace clustering / union-of-subspaces (SSC, Elhamifar & Vidal; LRR) is the
  right name for the model of `A(q)`, and local intrinsic dimension
  (Levina-Bickel, already used by the builder) is the right instrument for `D`.
- Hubness (Radovanović et al., JMLR 2010) explains the funnelling in 4.5: hubs
  absorb the descent, antihubs are the fragments we miss.

**Take the engineering.**
- *MUVERA* (Dhulipala et al., NeurIPS 2024, arXiv 2405.19504). Fixed Dimensional
  Encodings: asymmetric single-vector proxies for multi-vector similarity, with
  eps-approximation guarantees, same recall at 2-5x fewer candidates. This is
  literally "collapse `N` subspaces into `1024 + D*N` dimensions" with a proof,
  and the asymmetry is the same asymmetry RaBitQ uses here.
- *SOAR* (Sun et al., NeurIPS 2023, arXiv 2404.00774). Orthogonality-amplified
  residual loss so that multiple assignments of one point **cover each other's
  failures**, at 10-20% storage. The loss is the one to copy if we ever train the
  `b_p` side -- it is the formal version of "the entries should recover disjoint
  fragments", which is the thing AUC 0.606 says the geometry does not give.
- *GRIP* (arXiv 2610.30425). **Term-based entry point seeding**: precompute the
  documents with the largest weight on each vocabulary term and start the graph
  search from the ones under the query's highest-impact terms; 2.5x over Seismic
  at 95% recall@10 on 138M passages. This is the closest existing relative of the
  whole idea and it is the sparse-domain version: the modes are explicit terms,
  so `a(q)` is free. Our problem is the dense version, where the modes have to be
  estimated -- which is exactly why resolution turned out to be the binding
  constraint.
- *Adaptive entry point selection* (arXiv 2402.04713) gives b-monotonic paths and
  B-MSNET, and proves adaptive entry points beat a fixed central one under more
  general conditions. Formal backing that the entry is worth optimising at all.
- Multi-interest retrieval (MIND, ComiRec, Poly-encoder, SetMIR) has already
  named the two failure modes we would hit: **interest collapse** (the `N`
  embeddings learn the same mode -- our V=32-lands-on-12.4-attractors, in
  training form) and **static dispatch** (a fixed budget per embedding even when
  a query has one mode). SetMIR's per-interest *presence score* is the fix to
  copy: it tells the search how many entries to spend, which is exactly the knob
  that turns recall into occupancy here.

**Do not take.** Diversified retrieval (MMR, xQuAD, subtopic retrieval) is the
obvious-looking relative and is already measured dead on this problem
(ARCHITECTURE.md 4.5, MMR with the trade-off swept). The reason is 4.5's AUC
0.606: diversification needs a disjointness signal, and the pool-side geometry
does not have one.

## 7. Where this leaves the project

Closed, by measurement, in one day:

- the linear reading of `1024 + D*N` (Proposition 1 -- it is entry placement,
  already dead in ten arms);
- the document side as the bottleneck (section 2 -- a random `b_p` basis beats a
  fitted one);
- local PCA of the query's neighbourhood as the source of `a(q)` (section 3b --
  null at every resolution once the answers are removed from the fitting set).

Still standing, and worth the work:

- **the prize**, 1.2-1.9x at matched recall and a recall band the base frontier
  cannot reach, with the occupancy reading on top of it (section 1);
- **supervision as the only remaining source of `a(q)`.** The geometry does not
  hold the signal; a query log might. The experiment is: oracle entry sets from a
  training split as targets, a gated low-rank head in the Mixture-of-Logits form
  (the parameterisation with the universality theorem, and the only one that
  satisfies Proposition 1), SOAR's orthogonality-amplified residual as the loss
  (it is the formal version of "the entries should recover disjoint fragments",
  which AUC 0.606 says the geometry will not give), reported val-tunes /
  test-reports as this lab always does.
- **`N` per query, not fixed.** `varphi_h` already measures how many pieces the
  graph cut a query's answer into. It is the supervision signal for a presence
  score in SetMIR's sense, and it is the knob that turns a recall result into an
  occupancy result: spend entries where a query is actually fragmented.
- **the reporting axis.** Whatever lands, report it as `V` at matched recall and
  batch at matched latency, `Q*V` against the 188 SMs. That is where the
  contribution lives and no prior work measures it.

One more run is owed to the record: rung 2's accounting against the base frontier
pooled over `V` and `eps` at matched recall, so that "PRF costs more than it
saves" is a table and not an inference.

## 8. The allocation question, and the result that reopens everything

*Added 2026-10-26, after the sections above. The user's objection: "then just
allocate V by that weight." It was right, and the sections above had tested the
wrong direction.*

### 8.1 What had actually been tested

`bench/piece_dedup.py` (run here for the first time; it had no saved results) and
`bench/piece_mass.py` only ever moved the budget **outward**, and they chose the
extra pieces **blindly** -- any unseen attractor, answer-bearing or not:

| V=32 | recall | pieces reached | GT-piece coverage | dist/q |
|---|---|---|---|---|
| nearest-V | 0.9854 | 12.36 | 85.5% | 5386 |
| j=1 / 2 / 4 | -0.0002 / -0.0002 / -0.0001 | 13.1 / 13.8 / 15.1 | 86.2 / 86.6 / 87.5% | - |
| j=V (full dedup) | **-0.0081** (p=5e-23) | 31.36 | 92.1% | 4514 |

and the mass distribution looked like it explained the failure: a query's answers
divide **71.5%** (median 80%) into their largest piece and **86.4%** into their
largest two, 25.8% of queries have them all in one piece, 2.97 pieces per query.
Stratified by fragmentation, even the >= 4-piece third (n=646, base 0.9161 at
V=8) moved by at most +0.0014.

### 8.2 The allocation that works

`bench/mass_alloc.py` gives each **answer-holding** piece a quota and fills it
with the nearest entries reaching that piece. val, 2000 queries, nest4k,
M=256, total_B=2048:

| V | nearest-V | equal over GT pieces | mass over GT pieces |
|---|---|---|---|
| 8 | 0.9627 | +0.0188 (p=6e-41) | **+0.0196** (p=2e-47) |
| 16 | 0.9781 | +0.0096 (p=3e-28) | +0.0089 (p=4e-23) |
| 32 | 0.9854 | +0.0047 (p=1e-12) | **+0.0053** (p=3e-13) |

At V=8 that is recall 0.9823 against 2220 distance evaluations; nearest-V needs
V ~= 24 to reach it. **A 3x reduction in V at 5% more distance work**, which on
this GPU is 3x fewer CTAs per query and therefore 3x the batch before occupancy
saturates.

The diagnostic column is the one that matters. *in-mass* is the share of the V
entries that land in a piece holding at least one answer:

| | V=8 | V=16 | V=32 |
|---|---|---|---|
| nearest-V | 64.9% | 62.5% | 60.4% |
| allocated | 98.6% | 97.8% | 96.1% |

**Nearly 40% of the entry budget lands in pieces that hold no answer at all.**
That is the waste, and it is not the waste 4.5 and `basin_coverage.py` named --
those counted *duplication*, entries arriving where another entry already is.
Duplication is cheap to fix and worth nothing (8.1). Misdirection is expensive to
fix and worth 3x in V.

And equal quotas tie with mass-proportional ones at every V. **The lever is the
piece's identity, not its weight.** The mass skew of 8.1 was a true fact that
explained the wrong experiment: it explains why spreading fails, not why
allocating works.

This also corrects section 2's reading. "A random B beat the SVD B, so the
document side carries nothing" was measured against the wrong target -- the pool
node nearest each answer. The target that matters is the *basin* an entry
descends into, which is a property of the query's own greedy walk and appears in
no static factorisation of the score matrix.

### 8.3 What is still an oracle, and why the remaining question is small

The allocation uses the answers to know which pieces hold them, so it is a bound.
But the bound has changed the question into a much smaller one. `climb` is the
query's own greedy walk, so an entry's piece is *where its descent ends for this
query*. The question is no longer "where are the answer subspaces" (the
geometry does not say -- section 3b) but:

> **Will this entry's descent end somewhere an answer's descent ends?**

Three candidate signals, in `bench/basin_predict.py`, reported as AUC for that
binary event over the nearest M=256:

- `<q, x_p>`, the entry's own score -- what nearest-V uses, and 4.5 already warns
  it is weak (entry similarity predicts disjointness at AUC 0.606);
- `<q, .>` after only **t** best-neighbour steps, t small;
- `<q, x_A>` for the full attractor -- the ceiling for any walk-based signal.

If a short climb already ranks entries the way the attractor does, the method is
**oversubscribe-then-prune**: start 4V entries on a small budget, keep the V whose
walks look best, spend the rest there. The engine already supports it exactly --
`advance(a)` then `advance(b)` equals `advance(a+b)`, state is caller-owned, and
the per-branch counters and shared bound are already there. The `eps` retirement
rule (4.3) is the primitive form of this; what is new is starting wide and
concentrating, rather than starting at V and retiring.

No supervision is required for any of it.

## 9. The triage, built and measured

`bench/basin_predict.py` said the signal is the search's own first few expansions.
`bench/triage.py` spends them. No kernel change was needed and no supervision is
involved.

### 9.1 How it is built, and the one wrong way

**Right (`--mode inplace`).** One state with `cV` branches and budget `t + B2`.
`advance(t)`, read `st.dist[:, j, 0]` for each branch, keep the best `V`, and stop
the others by emptying their queues -- `ids = -1`, `expanded = 1`, which is the
engine's own documented exit ("no retained candidate unexpanded"). Then
`advance(B2)`. The cache, the visited tables and the claims all stay consistent,
and nothing is counted twice.

**Wrong (`--mode restart`, and worth recording).** Running a second search from
the survivors' seeds recomputes everything pass 1 did. The obvious fix --
carrying the per-query distance table across by pinning `cache_epoch` -- is
**incorrect**, because that table carries pass 1's **claims** as well as its
distances, and in exclusive mode the survivors are then locked out of exactly the
territory they were selected for. Measured recall **0.004**. Separating the two
would be a kernel change: the cache would need a claim bit that a new generation
can clear while keeping the distance.

### 9.2 What it buys, at fixed total expansions

msmarco-1m, val, 2000 queries, nest4k, L=12, total_B=2048, eps=0.05, fan-out
(so V branches really are V CTAs):

| V | arm | recall | in-mass | dist/q | of which triage |
|---|---|---|---|---|---|
| 8 | nearest-V | 0.9627 | 64.9% | 2111 | - |
| 8 | triage c=2 t=4 | 0.9742 (+0.0115, p=2e-29) | 77.5% | 3169 | 2125 |
| 8 | triage c=4 t=4 | **0.9785** (+0.0158, p=8e-38) | 83.8% | 4781 | 3786 |
| 16 | nearest-V | 0.9775 | 62.5% | 3391 | - |
| 16 | triage c=4 t=4 | **0.9885** (+0.0110, p=1e-38) | 78.0% | 8031 | 6661 |
| 32 | nearest-V | 0.9856 | 60.4% | 5390 | - |
| 32 | triage c=4 t=4 | **0.9923** (+0.0067, p=3e-24) | 72.4% | 13234 | 11474 |

Every arm is a real recall gain and every one is significant. The mechanism is
confirmed end to end: in-mass rises with `c` and `t` exactly as the AUC table
predicts, and recall rises with in-mass.

### 9.3 The cost, and the one thing that can fix it

**79% of the work is the triage** (3786 of 4781 at V=8, c=4, t=4). The reason is
structural: opening a branch evaluates its seed's whole neighbourhood, mean degree
51 here, and oversubscribing by `c` pays that `c` times. The baseline's own
expansions are nearly free by comparison -- 2048 expansions for 2111 distances,
about 1.1 each, because branches converge and the per-query cache absorbs the
repeats (4.4 measures 86%). The triage spends its budget in the one regime where
expansions are expensive: fresh territory.

On distance count, therefore, the triage is at best a wash against the base
frontier swept over `V` and `eps`. It lifts recall at fixed *expansions*, not at
fixed *distances*.

Which is exactly the place the 1-bit codes belong, and `advance` already takes
them (`codes=`). The triage's entire output is **which branches to keep** -- an
index, not a distance -- and that is the one site this lab exempts from "fp32 is
the comparison" ([[swann-tf32-catalogue]]: reduced precision is defensible only
where the output is an index). At d=1024 a code is 128 bytes against 4096 and the
scan is D/4 shared lookups with no multiplies (4.8). That does not reduce the
distance *count*, so it will not show up in dist/q at all -- it has to be read in
**wall clock**, which is why `bench/triage.py` now times both passes and reports
the best of three.

It is a mechanism, not a method. Section 10.

### 9.4 The 1-bit triage does not help, and the reason is specific

`--codes` changes the wall clock by nothing: V=8, c=4, t=4 is 26.92 ms on codes
against 26.25 ms in fp32, and dist/q 3735 against 3791. `advance` uses the codes
as a **margin filter**, not as the arithmetic: `load_codes`' own docstring says
margin 0.20 on MS MARCO "keeps every admitted candidate while letting 27.4% of
the neighbour slots through", so better than a quarter of the slots still get an
exact fp32 distance, and the code scan is added on top. The screen is tuned to
lose nothing, which is right for a search and wrong for a triage.

That is the one concrete thing that could still flip section 10. A triage only
needs the **ranking of its branches**, so it could expand on codes alone with no
fp32 refinement at any margin. That is a small kernel change -- a code-only
expansion mode -- and it is the only lever left that attacks the 79%.

## 10. Verdict: the eps rule was already doing this

The triage has to be judged against the base frontier swept over everything the
baseline can sweep. Over V in {8,16,32,64} x total_B in {1024..8192} x eps in
{0.02, 0.05, 0.12, 0.3, 0.6}, 32 points, the base Pareto frontier runs from
recall 0.9473 at 10.31 ms to **0.9975 at 141.74 ms**.

Against that frontier, the 24 triage arms (fp32 and codes, c in {2,4}, t in
{2,4}, V in {8,16,32}) come out:

| | count | range |
|---|---|---|
| ahead by > 3% | 8 | 1.04x - 1.11x |
| within +-3% | 7 | 0.99x - 1.02x |
| behind by > 3% | 9 | 0.70x - 0.96x |

and the recall band the triage looked like it had to itself in 9.2 is gone: the
base frontier reaches 0.9975.

**The reason is `eps`.** The retirement rule of 4.3 retires a lineage whose best
unexpanded candidate lies beyond `(1 + eps)` times the query's current k-th best
distance. That is the same decision the triage makes -- *is this branch going
anywhere* -- taken **continuously, on branches that are already open, for no
extra work at all**. The triage takes the same decision once, earlier, and pays
`c` times the branch-opening cost to have the option. Starting at the right width
and retiring continuously dominates starting wide and concentrating.

So section 9's gains were real and they were not new: they are what `eps` buys,
bought a second time at a worse price.

### What the work is actually worth

The measurement that stands is the diagnosis, and it is a better result than the
method would have been:

- **40% of the entry budget lands in a basin holding no answer** (in-mass 64.9%
  at V=8 falling to 60.4% at V=32), and fixing that is worth +0.0196 recall at
  V=8 (8.2).
- **An entry's own proximity to the query predicts this at AUC 0.5393** -- a coin
  flip. Four best-neighbour steps predict it at 0.9328 and eight at 0.9995 (8.3).
- Therefore 4.7's "proximity is worth almost nothing" is not a statement about
  catalogues or placement. It is this: **where an entry *is* says nothing about
  where it *goes*, and only where it goes matters.** That is also why `eps`
  works, why ten placement arms failed, and why no static factorisation of the
  score matrix (section 2) could ever have captured it -- the quantity being
  predicted is a property of the query's own trajectory, which exists in no
  offline table.

### Honest gaps

- The triage arms were run at **eps=0.05 only**. Sweeping eps for the triage too
  is the fair completion. Given that the losses reach 0.70x and the wins stop at
  1.11x, and that eps is the dominant knob for both arms, a wash is the expected
  outcome -- but it is not measured.
- One corpus (msmarco-1m), one graph (G_d64), one pool family (nest), val split.
- The code-only expansion mode of 9.4 is unbuilt.

## 11. Is it a graph problem? Yes, and the graph says no

*Added after section 10. The user's reframe: "where it descends to -- isn't that
something a graph algorithm should solve?" It is the right question and it has a
clean answer.*

### 11.1 Why a ball cannot work and a path can

The signal sits 4-8 hops away (AUC 0.9328 at 4 steps, 0.9995 at 8). Two ways to
reach it, and only one is affordable:

- the h-hop **ball** is exponential -- deg 51 here, so the 4-hop ball is ~6.7M
  nodes. Storing a node's neighbourhood offline therefore stops at h=1, which is
  exactly `bench/entry_maxpool.py`, and 1-step AUC 0.6302 is exactly how weak
  that arm was (+0.0069, beaten by a cost-matched dense catalogue at +0.0122);
- the greedy **path** is linear -- it touches t nodes.

So the cost is not the path's length. It is the price of **one step**: choosing
the best neighbour evaluates all `deg` of them, and the triage pays
`cV * t * deg`. Narrowing that one choice to `w` edges would take 79% of the work
down by `deg/w`, and at w=4 that is 12.75x. `bench/ascent_index.py` and
`bench/ascent_diag.py` test it.

### 11.2 The statistic that looked like a green light was an artifact

The first run reported "the top-1 slot takes 90.4% of each visited node's steps".
It averaged the share **per node**, and 68.1% of visited nodes were stepped
through exactly once -- share 1.0 at w=1 by construction. Weighted by steps, and
again restricted to the nodes the walk actually passes through repeatedly:

| w | per-node mean (the artifact) | step weighted | at hubs (>= 10 visits) |
|---|---|---|---|
| 1 | 0.8946 | 0.6824 | **0.3088** |
| 2 | 0.9532 | 0.7844 | 0.4169 |
| 4 | 0.9792 | 0.8694 | 0.5673 |
| 8 | 0.9943 | 0.9479 | 0.7827 |
| 16 | 0.9995 | 0.9925 | 0.9685 |

At the nodes that matter the leading edge wins **31%** of the time, and covering
97% needs 16 of the 96 slots.

### 11.3 Held out, the index does not transfer

Build on 1000 val queries (32 entries, 16 steps: 93,389 steps over 43,974
distinct nodes, 4.4% of the corpus); measure on the other 1000.

| w | per-step agreement | at nodes the build saw | w nearest edges, no log | compounded over 8 steps |
|---|---|---|---|---|
| 1 | 0.0860 | 0.0536 | 0.0804 | 0.0000 |
| 4 | 0.2321 | 0.1713 | 0.2146 | 0.0000 |
| 16 | 0.5294 | 0.4583 | 0.5124 | 0.0062 |
| 32 | 0.7925 | 0.7513 | 0.7859 | 0.1556 |

and end to end (`ascent_index.py`), a narrow walk reaches the full-width
attractor 7.9% of the time at w=1 and 16.1% at w=8, against 99.9% for the full
width at 8 steps.

Three things at once, and each is fatal on its own:

1. **The query log contributes almost nothing.** The learned index beats "keep the
   w nearest edges, no log at all" by 1-2 points at every w. Whatever the index
   knows, the builder's own edge order already knew.
2. **Coverage does not transfer.** Only 40.5% of held-out steps land on a node the
   build had seen, and only 21.4% of the distinct nodes. The walk is query
   specific, so a log of any realistic size covers the wrong nodes.
3. **It is worse exactly where it is used.** Agreement at seen nodes is *lower*
   than overall (0.0536 against 0.0860 at w=1), because the seen nodes are the
   hubs, and the hubs are where the choice is least concentrated (11.2).

### 11.4 The statement this settles

> **Greedy ascent's next step is not a graph property.** It is
> `argmax_{u in N(p)} <u, q>`, and which neighbour wins is decided by `q`, not by
> the graph. The graph supplies the candidate set; the query supplies the answer.

Which closes the loop with section 10 and makes it an identity rather than an
inference: the quantity that predicts an answer-bearing basin at AUC 0.9995 is a
property of the **query's own trajectory**, and a signal that locates the answers
that precisely cannot also be cheap and query-independent -- if it were, it would
*be* the index. Section 10 said this quantity "exists in no static table"; 11.3
measures it.

This is also the honest form of 4.5's AUC 0.606 and 4.7's "proximity is worth
almost nothing". They are not facts about catalogues or placements. They are this
one fact, seen from the entry side.

### 11.5 The one variant left, and why to expect little

The failure above is that the choice depends on `q` and the table does not. The
fix in principle is to **condition the table on direction**: per node, store `b`
(direction, chosen edge) pairs and pick the bucket nearest `q`. The directions
that actually arrive at a given node are restricted, so the conditional
distribution could be far sharper than the 31% marginal.

The reason to expect little: `argmax_{u in N(p)} <u, q>` partitions the sphere
into one cell per neighbour, ~51 of them. A table with `b` buckets resolves the
choice only when `b` approaches the number of distinguishable cells, and storing
that map *is* storing the computation it was meant to avoid. The restriction of
arriving directions is the only reason it is not simply circular, and how much it
buys is one measurement -- but it is a measurement against a known wall, not an
open direction.

## 12. Why the basins are like that: one scalar, and it is not the data

*The user's question: "then why do basins come out that way -- is it in the data
distribution?" It is not. `bench/basin_why.py` holds everything fixed but one
axis at a time, with `pool_random4k`, the one pool every corpus has.*

### 12.1 Four axes

| axis | what moves | cells |
|---|---|---|
| graph | the build rule, same vectors/queries/pool | msmarco-1m under DiskANN prune, CAGRA, HNSW, NN-descent (d64, d128) |
| dimension | the encoder width, **same 100k documents** | qwen-1024 / 2560 / 4096 |
| size | N, same encoder and content | ms-125k / 250k / 500k / msmarco-1m |
| domain | learned text, word vectors, L2 descriptors | msmarco, fever, glove-100, gist-960 |

`att/entry` is the number of distinct basins the nearest M=64 entries reach,
divided by M -- the inverse of "how many entries share a basin". `in_mass` is the
share of the nearest 8 that land in a basin holding an answer.

| cell | att/entry | in_mass8 | AUC(proximity) |
|---|---|---|---|
| qwen-4096 | 0.134 | 0.883 | 0.5700 |
| qwen-2560 | 0.142 | 0.881 | 0.5728 |
| qwen-1024 | 0.146 | 0.873 | 0.5780 |
| gist-960 | 0.252 | 0.639 | 0.5345 |
| ms-125k | 0.285 | 0.716 | 0.5413 |
| ms-250k | 0.323 | 0.670 | 0.5508 |
| msmarco-1m, NN-descent d128 | 0.336 | 0.615 | 0.5574 |
| ms-500k | 0.360 | 0.640 | 0.5469 |
| msmarco-1m, DiskANN prune | 0.404 | 0.576 | 0.5425 |
| fever-1m | 0.408 | 0.575 | 0.5508 |
| msmarco-1m, CAGRA | 0.441 | 0.564 | 0.5463 |
| glove-100 | 0.490 | 0.522 | 0.5531 |
| msmarco-1m, NN-descent d64 | 0.516 | 0.466 | 0.5745 |
| msmarco-1m, HNSW | 0.536 | 0.454 | 0.5377 |

### 12.2 The law

```
    in_mass  =  0.9993 - 1.0299 * (att/entry)        r = -0.972,  rms 0.033
```

Slope -1.03, intercept 0.999. The bare identity `in_mass = 1 - att/entry` fits
with rms 0.035 and max residual 0.109 (the outlier is gist-960, the one L2
descriptor set, which also has the most answer basins at 5.74). Read it:

> **Every additional distinct basin that proximity opens is a basin with no
> answer in it.** Entries that share a basin share an answer-bearing one; an
> entry that strikes out into a basin of its own strikes out.

And it is not confined to nearest-V. Over three selection strategies on
msmarco-1m / nest4k / V=32 / 1000 queries:

| selection | att/entry | in_mass | 1 - att/entry | residual |
|---|---|---|---|---|
| nearest-V | 0.425 | 0.560 | 0.575 | -0.015 |
| one per piece (`piece_dedup`) | 0.982 | **0.101** | 0.018 | +0.083 |
| mass over GT pieces (the oracle of 8.2) | **0.141** | **0.950** | 0.859 | +0.091 |

So `piece_dedup`'s recall collapse (-0.0081, p=5e-23) was it driving att/entry to
0.982 and in-mass to **10%**. And the oracle allocation worth +0.0196 does not
escape the law: it reaches in_mass 0.950 by driving att/entry from 0.425 to
0.141 -- 2.4 entries per basin to 7.1. **It moves along the law, it does not
break it.**

### 12.3 What the axes say

- **Dimension does nothing.** The same 100k documents at 1024, 2560 and 4096 give
  att/entry 0.146, 0.142, 0.134. Whatever this is, it is not high-dimensional
  geometry, which is the first thing one would guess.
- **N is the driver.** At P=4096 fixed, N from 125k to 1M (8x) takes att/entry
  0.285 -> 0.404 and in_mass 0.716 -> 0.576, monotonically. Roughly
  `att/entry ~ N^0.17` on four points. So the controlling ratio is the
  catalogue's density against the corpus, which is exactly the mechanism behind
  [[swann-catalogue-density]], now with a name: a denser catalogue puts more
  entries in each basin.
- **The graph matters, by 1.6x.** att/entry 0.336 (NN-descent d128) to 0.536
  (HNSW), in_mass 0.615 to 0.454. The build rule sets how finely the landscape is
  cut.
- **Domain matters moderately**, 0.252 (gist) to 0.490 (glove) at N ~ 1M.

### 12.4 And the unpredictability is invariant

**AUC(entry proximity) = 0.5540 +- 0.0139 across all fourteen cells**, range
0.5345 to 0.5780, while AUC(8-step climb) is 0.9999 to 1.0000 in every one. Five
graph builders, three encoder widths, an 8x range of N, four domains, two of them
L2 and not learned at all.

So the two halves of the question have different answers:

- **the rate** -- how many entries are wasted -- is one scalar, `att/entry`, set
  by catalogue density against corpus size, then by the graph, then by domain,
  and **not at all** by dimension;
- **the unpredictability** -- which entries are wasted -- does not move. Nothing
  tested changes it. That is section 11's statement again: the graph supplies the
  candidate set and the query supplies the answer.

### 12.5 What this says about ten failed arms

Coverage over a reach sketch, near-but-spread max-min, MMR with the trade-off
swept, denser catalogues, a pool graph (4.5, 4.7) -- every entry-side
intervention this lab has tried is a **diversity** intervention, and by 12.2
every one of them *raises* att/entry. They were all pushing the wrong way along
the law, and the law says the marginal basin is always a miss. Only catalogue
density, the one that worked, pushes the other way.

Which names the untried mirror image, and it needs no basin labels, no climb and
no supervision: pick entries near the query **and near each other**, since mutual
similarity is a free surrogate for sharing a basin. Greedy, with `lam >= 0`
*rewarding* closeness to what is already picked:

```
    pick argmax_p   <q, x_p>  +  lam * max_{p' in picked} <x_p, x_p'>
```

`lam = 0` is nearest-V and `lam < 0` is MMR, which is here as the control that
should lose because 4.7 already measured it losing. `bench/anti_mmr.py`.

### 12.6 The law is a constraint, not a lever

msmarco-1m, val, 2000 queries, nest4k, 256 candidates, total_B=2048:

| V=8, lambda | | recall | att/entry | in_mass |
|---|---|---|---|---|
| -0.50 | MMR (spread) | 0.9567 (-0.0058) | 0.620 | 0.570 |
| **0.00** | **nearest-V** | **0.9625** | 0.534 | 0.649 |
| 0.25 | anti-MMR | 0.9624 (-0.0001) | 0.498 | 0.663 |
| 0.50 | anti-MMR | 0.9621 (-0.0003) | 0.478 | **0.671** |
| 1.00 | anti-MMR | 0.9596 (-0.0028) | 0.465 | 0.671 |
| 2.00 | anti-MMR | 0.9561 (-0.0063) | **0.461** | 0.670 |

and the same shape at V=16 and V=32: att/entry 0.448 -> 0.392 and 0.386 -> 0.348,
in_mass up, recall down -0.0071 and -0.0066 at lambda=2.

Anti-MMR does exactly what 12.2 predicts -- it lowers att/entry and raises
in_mass -- and **recall does not follow**. Both directions lose, and nearest-V
sits at the top of the curve.

The reason is that clustering blindly concentrates on whichever basin the first
pick landed in, and that basin holds an answer only 65% of the time. It amplifies
the first pick instead of correcting it, and pays proximity for the privilege. The
oracle of 8.2 reaches in_mass 0.950 and wins because it concentrates on the
basins that **hold answers** -- which is the AUC 0.54 information again.

So the entry side closes with one statement, and it covers every arm this lab has
run:

> The waste in the entry budget is governed by one scalar, `att/entry`, which
> obeys `in_mass = 1 - att/entry` across corpora, graphs, dimensions, sizes and
> selection strategies. **nearest-V is the optimum of the one-parameter family
> that slides along that identity**: diversity (ten arms, 4.5/4.7) loses by
> raising att/entry, anti-diversity loses by betting on the first pick. Escaping
> the identity requires choosing basins by *content*, and that is precisely the
> quantity a position cannot supply (AUC 0.5540 +- 0.0139 over fourteen cells)
> and a walk can (0.9999+) -- at the price of the walk.

Catalogue density is the one intervention that ever won here
([[swann-catalogue-density]]) and it is the only one that lowers `att/entry`
without spending proximity: it does not move the entries, it supplies more of
them per basin. 12.3 is why it works and why its payoff is set by the
scan-to-search cost ratio.

## 13. The expansion rule is not the entry rule, and the gap is reachability

Section 12 closed entry selection. The obvious next suspicion was that the same
critique applies *inside* the search, because the kernel chooses what to expand by
the same criterion -- 3/step-1, the nearest unexpanded node of the lineage whose
turn it is -- and the budget there is 2048 expansions rather than 8-32 entries.

`bench/expand_value.py`, msmarco-1m val, 400 queries, V=8, L=64, B=64, eps=0.05,
recall 0.9580. The queue holds 512 slots and the search leaves **431 of them
(84.2%) unexpanded**, so it declines plenty of choices. For each declined slot,
does its own similarity to the query predict that expanding it would reach an
answer?

| label | base rate | AUC self (what the kernel uses) | AUC 1-step lookahead |
|---|---|---|---|
| a neighbour is an answer | 0.1721 | **0.7853** | 1.0000 |
| a neighbour is a MISSED answer | 0.0012 | **0.6948** | 0.9361 |

(The lookahead column is partly circular -- its score *is* the max neighbour
similarity, and an answer is a global top-10 node -- so the column that matters is
"AUC self".)

**The expansion rule is not broken.** 0.785 against the entry side's 0.554, and
the reason is clean:

> **Proximity works locally and fails globally.** Inside the queue every node is
> already in the right region, so proximity resolves them. Among entries the
> candidates are spread over basins, and proximity does not resolve which basin.

And the base rate settles it regardless of any rule: **0.12% of the declined
slots have a missed answer among their neighbours** -- only 69 of 400 queries have
even one such slot. There is nothing for a better expansion rule to pick.

> **The missed answers are not adjacent to the queue at all.**

Which is the same thing [[swann-catalogue-density]] measured from the other side:
handed its single best entry alone, with no shared bound and no rivals, a descent
reaches a missed-and-alone answer only 46.4% of the time, because the descent
minimises distance to the query and so leaves the isolated piece rather than
carrying it back.

### 13.1 Where the remaining work is, and where the subspaces finally fit

| axis | state |
|---|---|
| entry selection | closed -- the identity of 12.2, nearest-V at its optimum |
| expansion order | closed -- the rule is already good (0.785), and the opportunity is 0.12% |
| **edges** | **open** -- the missed answers are unreachable, so the problem is the build |

And this is where the subspace structure is usable after all. It is not usable at
search time -- sections 3b, 11 and 12 close that thoroughly -- but at **build**
time the circularity that killed it disappears, because a query log's answers are
*known*, not inferred:

- a query's answers divide into 2.6-5.7 pieces (12.1), and those pieces are what
  the graph cut apart;
- so the intervention is **edges between the pieces**, and the piece structure is
  the supervision for which edges;
- nothing has to be predicted from `q` at query time, which is the constraint
  Proposition 1 and section 11 could not satisfy.

This is the line the bibliography already names: `ngfix` edits a built graph from
a query log and `roargraph` builds query-aware edges, and `/root/NGFix` is in the
tree. `varphi_h` and `phi-hat_1` are already the instruments for it, and 4.6b
already establishes that this builder's graph beats CAGRA's, so the question is
well posed: **how far can inter-piece edges drive `varphi` at a fixed edge
budget, and what does that buy in recall?**

The next measurement is the ceiling: link each query's answer pieces to each
other and see where recall goes. If it is large, the entry-side work of sections
8-13 has handed the build side its objective.

## 14. The supervised head: built, and where its loss goes

Section 13.1 left supervision as the only untried source of `a(q)`, and the user
proposed a diffusion model for it. The agreed order was to build the cheapest
member of the family first, because a diffusion sampler distils to a feed-forward
head for serving anyway, and because the dial is already calibrated at both ends:
in_mass 0.560 for nearest-V, 0.950 for the oracle.

### 14.1 The data, which turned out to exist

`/workspace/SYN/emb/msmarco-1m-trainpool__vanilla_queries.npy` holds **59,242**
queries from the same encoder as val/test -- BAAI/bge-m3, cls, d=1024, revision
5617a9f6 -- over the same corpus, and none of the first 5,000 appear in val or
test by hash. The docs there are fp16 and `X.npy` is the re-normalised fp32 cast
(per-row cosine >= 0.99993), so the training queries live in the search's space.
`bench/head_data.py` recomputes their exact top-10 on the GPU (4.9 s) and their
piece structure by the same greedy climb used everywhere else (4.8 s). The
distributions match the evaluation almost exactly:

| | pieces/query | largest piece's share | single-piece | >= 4 pieces |
|---|---|---|---|---|
| train, 59,242 | 2.85 | 73.1% | 27.4% | 29.1% |
| val, 2,000 (8.1) | 2.97 | 71.5% | 25.8% | 32.3% |

### 14.2 The head, and the baseline that makes it readable

`bench/head_train.py`. `out_j = normalize(q + delta_j)` for j < N, zero-init, so
the untrained head is `q` itself; a presence logit per output (SetMIR's score, 6);
and **no Hungarian matching** -- the pieces have a canonical order by how many
answers they hold, so output j targets the j-th largest piece and the targets are
distinct by construction, which is also what stops the outputs collapsing onto
each other.

The number that makes the loss readable is what `out_j = q` scores, per output:

| | overall | j=0 | j=1 | j=2 | j=3 |
|---|---|---|---|---|---|
| identity (`out_j = q`) | 0.3604 | 0.2901 | 0.3774 | 0.4247 | 0.4565 |
| trained, 40 epochs | **0.1241** | 0.0975 | 0.1611 | 0.1747 | 0.1620 |
| live fraction | - | 1.000 | 0.726 | 0.466 | 0.291 |

**Every output beats the identity substantially, including the minor pieces.**
So there is learnable signal in the query for where the *other* answer pieces
are -- which section 3b showed the neighbourhood's geometry does not contain.
Pairwise cosine between the N outputs is 0.872, so they did not collapse.

### 14.3 And it loses recall. The two controls say why

test, 2,000 queries, V=8, nest4k, total_B=2048, eps=0.05:

| arm | recall | att/entry | in_mass |
|---|---|---|---|
| nearest-V | 0.9658 | 0.526 | 0.655 |
| head (predicted pieces) | 0.9620 (-0.0038) | 0.469 | 0.706 |
| **TRUE piece centroids** | **0.9725 (+0.0067)** | 0.438 | 0.795 |
| **TRUE attractors (8.2)** | **0.9830 (+0.0172)** | 0.362 | 0.986 |

val agrees to within 0.0007 on every row. Two separate losses, and they are
cleanly separated by the controls:

1. **The pipeline costs 58% of the prize even with exact inputs.** Aiming at a
   true piece centroid and taking the nearest pool entry gives +0.0067 against
   the attractor oracle's +0.0172, and in_mass 0.795 against 0.986. The centroid
   is a point in embedding space; the thing that matters is a basin, and
   proximity maps the one to the other badly -- which is 12's att/entry ~0.5
   again, basin boundaries sitting below the spacing of the entries.
2. **Prediction error eats the rest.** The head's outputs sit at cosine ~0.876
   from their targets, which is not close enough to land in the right basin.

Note that the head *does* break 12's identity, and by more than the oracle: its
residual over `1 - att/entry` is +0.171, against +0.091 for the oracle and -0.015
for nearest-V. It is the best selector yet by the in-mass measure. It still loses
recall, which is the third time in this document that in-mass has moved the right
way without recall following (the others are anti-MMR in 12.6 and the triage in
9.2 on cost).

### 14.4 What that implies, and v2

The oracle's target was never a vector. It is a **set of pool entries**. So the
embedding-space detour is avoidable: score the pool directly and train on which
entries reach an answer-bearing basin. `bench/head_pool_data.py` caches that
supervision for all 59,242 training queries (their nearest 64 entries and the
basin label for each), and `bench/head_pool.py` learns

```
    score_p = <q, x_p>  +  alpha * max_j <out_j, x_p>
```

with `alpha` and `delta` both zero-init, so the untrained model **is** the
catalogue scan and any departure is deliberate. The `max` is what keeps this past
Proposition 1 -- an additive linear term in `q` is a pool re-placement and nothing
more -- and it is one matmul of a width the scan already pays for.

The ceiling to beat is in 14.3 and it is +0.0172 at in_mass 0.986.
