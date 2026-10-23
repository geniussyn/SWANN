"""SWANN: intra-query parallelism for graph-based ANN search on GPUs.

A graph walk is sequential, so one query exposes almost no parallelism and the
usual one-block-per-query mapping leaves a GPU idle. SWANN splits a query into
trajectories that advance at the same time and keeps them apart: they share a
distance table, share a bound on the answer, and claim nodes so no two branches
walk the same ground.

    import swann
    index = swann.Index.build(X, metric="ip")
    ids, dist = index.search(Q, k=10)
"""
from ._engine import overflow_guard
from .index import Index, SearchConfig

__all__ = ["Index", "SearchConfig", "overflow_guard"]
__version__ = "0.1.0"
