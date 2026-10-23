# SWANN

Intra-query parallelism for graph-based approximate nearest neighbour search on
GPUs.

A graph index answers a query by walking, and a walk is a chain: what to expand
next is known only once the current candidates have been scored. One query
therefore exposes almost no parallelism, and the mapping every GPU graph index
uses — one thread block per query — inherits that. One block is 0.04% of a
modern device; filling it takes thousands of simultaneous queries.

SWANN splits one query into `V` trajectories that advance at the same time, and
keeps them apart. They share a distance table so a node is scored once for the
whole query, share a bound so every branch prunes against the best answer any
of them has found, and claim nodes so no two branches walk the same ground.

## Install

Needs a CUDA GPU, PyTorch with CUDA, and a compiler. The CUDA extension is
built on first import against your torch and driver, which takes a minute or
two; it is cached afterwards.

```bash
pip install -e .
python examples/quickstart.py
```

## Use

```python
import swann

index = swann.Index.build(X, metric="ip")   # X: (n, d) float32, rows normalised for "ip"
ids, dist = index.search(Q, k=10)           # Q: (nq, d) float32
```

`ids` and `dist` are `(nq, k)` numpy arrays. For L2 corpora pass
`metric="l2"` and leave the vectors unnormalised.

Persist the graph and reattach it to vectors you already hold:

```python
index.save("my_index")
index = swann.Index.load("my_index", X)
```

## Knobs

The defaults are the configuration every number in the paper was measured at.

| argument | default | what it does |
| --- | --- | --- |
| `branches` | 32 | trajectories per query, and thread blocks per query. This is the parallelism: a batch of `2256 / branches` fills one RTX PRO 6000. Raising it raises the recall ceiling; lowering it suits large batches. |
| `stop_eps` | 0.05 | stop a branch once its best candidate is worse than the query's current k-th answer by this fraction. The main accuracy/throughput dial. |
| `queue` | 12 | width of each branch's candidate queue. Below about 8 recall collapses. |
| `budget` | 64 | cap on expansions per branch; `stop_eps` usually binds first. |
| `team` | 16 | threads cooperating on one distance. 16 suits `d >= 512`, 32 suits wider still. |

```python
ids, dist = index.search(Q, k=10, branches=128, stop_eps=0.2)
```

## Half precision

Halving the corpus halves the bytes a walk reads, which is most of its cost at
width:

```python
index = swann.Index(X_fp16, graph, metric="ip")
```

Queries stay fp32 regardless — the kernel holds the query vector in shared
memory and reads it as floats, so a half query is rejected rather than merely
pointless.

## Serving

Each search verifies that no branch overflowed its visited table, which reads a
counter back to the host and so synchronises. The check costs a tenth of a
single-query search and nothing at batch. A serving loop that has already
validated its configuration can turn it off and inspect the counter when
convenient:

```python
swann.overflow_guard(False)
```

With it off the search issues no host synchronisation, which also makes the
whole path capturable as a CUDA graph.

## Limits

- **Batch size.** The kernel indexes queries by the grid's `y` dimension, which
  CUDA caps at 65535. Split larger batches.
- **Dimension.** Per-block shared memory holds the query vector, so `d = 4096`
  spends 16 KiB of the 48 KiB a block gets by default and the queue width has
  to give way.
- **Build cost.** The graph build is an exact candidate sweep: minutes for a
  million rows, against seconds for an IVF index. It is paid once, offline, and
  has no incremental path — a corpus under constant ingest is the case where
  this trade does not hold.

## Where it helps, and where it does not

Measured against cuVS CAGRA and IVF-RaBitQ on ten corpora at matched recall and
matched precision: on the two narrowest (`d = 100`, `d = 128`) CAGRA is faster;
from `d = 1536` upward SWANN leads on every corpus, by 1.7–3.8× at recall 0.95
and up to 9.2× at 0.99. Below a batch of a few hundred queries a partitioned
index is ahead, because one query already makes it scan whole lists and fill
the device.

## Reproducing the paper

The measurements are a separate concern from the library, and they need the
baselines installed: cuVS for CAGRA, IVF-FLAT and IVF-PQ, the IVF-RaBitQ
extension in `swann/csrc`, GGNN, and hnswlib. Point `CUVS_PY` at the cuVS
environment if it is not importable directly.

```bash
# 1. prepare corpora into $ROOT (one directory each, with X.npy and a manifest)
python3 offline/prep_annbench.py --h5 gist-960-euclidean.hdf5 --name gist-960 \
    --out /dev/shm/swann/gist-960

# 2. build the graph each corpus is searched over
python3 offline/build_graph.py --data /dev/shm/swann/gist-960 \
    --D 96 --cand 128 --alpha 1.2 --out G_d64.npy

# 3. every measurement, in order
CUVS_PY=/opt/cuvsenv bash scripts/reproduce.sh

# 4. every figure
bash scripts/figures.sh
```

`reproduce.sh` refuses to start a timing while anything else holds the GPU. That
is not politeness: a concurrent job inflates every number in the run, and the
inflation is invisible afterwards.

Ground truth is recomputed by the preparation scripts as exact fp32 search on
the same vectors. We do not use the neighbour lists distributed with a corpus,
because they were produced by other code at another precision.

## Layout

```
swann/index.py     the public Index and SearchConfig
swann/build.py     graph construction
swann/_engine.py   search state and the kernel launches
swann/_select.py   seed selection from the catalogue
swann/csrc/        the CUDA kernels
bench/             one harness per system, plus the figure scripts
offline/           corpus preparation and graph construction
scripts/           the two runners that drive the above
```

## Citation

```bibtex
@article{swann,
  title  = {Intra-Query Parallelism for Graph-Based {ANN} Search on {GPUs}},
  author = {Kim, Sieun and Jang, Jihyo and Kim, Kyungmin and
            Park, Junhwi and Min, Jaehwi and Lee, Sang-Won},
  year   = {2026}
}
```
