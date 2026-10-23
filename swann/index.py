"""The public interface: build an index over vectors, then search it.

Two objects and three calls. Everything the paper discusses is reachable
through keyword arguments, but the defaults are the configuration we report,
so the short form is also the one that was measured.

    import swann
    index = swann.Index.build(X, metric="ip")      # X: (n, d) float32
    ids, dist = index.search(Q, k=10)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import _engine as E
from . import _select as S

METRICS = {"ip": E.METRIC_IP, "l2": E.METRIC_L2}


@dataclass
class SearchConfig:
    """Knobs, with the values every number in the paper was measured at.

    branches      how many trajectories one query is split into. This is the
                  parallelism: a query occupies `branches` thread blocks, and a
                  batch of `2256 / branches` fills one RTX PRO 6000.
    queue         width of each branch's candidate queue.
    budget        cap on expansions per branch; `stop_eps` usually binds first.
    stop_eps      stop a branch once its best candidate is worse than the
                  query's current k-th answer by this fraction. Negative
                  disables it and `budget` governs instead.
    team          threads cooperating on one distance. 16 suits d >= 512.
    """
    branches: int = 32
    queue: int = 12
    budget: int = 64
    stop_eps: float = 0.05
    team: int = 16
    k: int = 10


class Index:
    """A proximity graph plus the vectors it indexes, resident on one GPU."""

    def __init__(self, vectors: torch.Tensor, graph: torch.Tensor,
                 metric: str = "ip", catalogue: torch.Tensor | None = None,
                 catalogue_size: int = 4096, seed: int = 0):
        if vectors.device.type != "cuda":
            raise ValueError("vectors must already be on the GPU")
        if vectors.dtype not in (torch.float32, torch.float16):
            raise ValueError("vectors must be float32 or float16")
        if metric not in METRICS:
            raise ValueError(f"metric must be one of {sorted(METRICS)}")
        self.X = vectors.contiguous()
        self.G = graph.contiguous()
        self.metric = metric
        self._m = METRICS[metric]
        if catalogue is None:
            n = self.X.shape[0]
            rng = np.random.default_rng(seed)
            pick = np.sort(rng.permutation(n)[:min(catalogue_size, n)])
            catalogue = torch.from_numpy(pick.astype(np.int64)).cuda()
        self.catalogue = catalogue
        # the catalogue's own vectors, kept in fp32 so entry selection is not
        # the thing that loses precision when the corpus is halved
        self._cat = S.pool_vectors(self.X.float(), self.catalogue)

    # -- construction ----------------------------------------------------
    @classmethod
    def build(cls, vectors, metric: str = "ip", degree: int = 96,
              candidates: int = 128, alpha: float = 1.2, **kw) -> "Index":
        """Build a graph over `vectors`. Minutes for a million rows."""
        from .build import build_graph
        X = _as_cuda(vectors)
        G = build_graph(X, metric=metric, degree=degree,
                        candidates=candidates, alpha=alpha)
        return cls(X, G, metric=metric, **kw)

    # -- search ----------------------------------------------------------
    def search(self, queries, k: int = 10, config: SearchConfig | None = None,
               **overrides):
        """Return (ids, distances), each (nq, k), as numpy arrays.

        Queries stay in fp32 even when the corpus is halved: the kernel holds
        the query vector in shared memory and reads it as floats.
        """
        cfg = config or SearchConfig(k=k)
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise TypeError(f"unknown search option {key!r}")
            setattr(cfg, key, value)
        Q = _as_cuda(queries).float()
        if Q.shape[0] > 65535:
            raise ValueError(
                f"batch of {Q.shape[0]} exceeds the 65535 the kernel's grid "
                "allows; split it")
        entries = S.select_nearest(Q, self._cat, cfg.branches, metric=self._m)
        seeds = self.catalogue[entries].to(torch.int32)
        ids, dist, _ = E.search(
            self.X, self.G, Q, seeds, k=k, L=cfg.queue, B=cfg.budget,
            metric=self._m, team_size=cfg.team, stop_eps=cfg.stop_eps,
            shared_cache=True, shared_bound=True, exclusive=True,
            elide_visited=True)
        return ids.cpu().numpy(), dist.cpu().numpy()

    # -- persistence -----------------------------------------------------
    def save(self, path) -> None:
        """Write the graph and the catalogue. The vectors are yours already."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "graph.npy", self.G.cpu().numpy())
        np.save(path / "catalogue.npy", self.catalogue.cpu().numpy())
        (path / "index.json").write_text(json.dumps(
            {"metric": self.metric, "n": int(self.X.shape[0]),
             "d": int(self.X.shape[1]), "degree": int(self.G.shape[1])},
            indent=2))

    @classmethod
    def load(cls, path, vectors) -> "Index":
        path = Path(path)
        meta = json.loads((path / "index.json").read_text())
        G = torch.from_numpy(np.load(path / "graph.npy")).cuda()
        cat = torch.from_numpy(np.load(path / "catalogue.npy")).cuda()
        return cls(_as_cuda(vectors), G, metric=meta["metric"], catalogue=cat)

    def __repr__(self) -> str:
        n, d = self.X.shape
        return (f"Index(n={n:,}, d={d}, degree={self.G.shape[1]}, "
                f"metric={self.metric!r}, dtype={self.X.dtype})")


def _as_cuda(a) -> torch.Tensor:
    if isinstance(a, np.ndarray):
        a = torch.from_numpy(np.ascontiguousarray(a))
    if a.device.type != "cuda":
        a = a.cuda()
    return a.contiguous()
