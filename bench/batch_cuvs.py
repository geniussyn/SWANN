"""Batch-size scaling for the cuVS systems, in the cuVS process.

Throughput and latency are measured. The traffic and arithmetic columns are
modelled from each index's structure, because cuVS exposes no counter and the
profiler is unavailable here:

  IVF-FLAT   the centroid scan, then the probed lists read as whole fp32 vectors
  IVF-PQ     the centroid scan, then the probed lists read as PQ codes; with
             refinement, plus the real vectors of the candidates it re-ranks
  CAGRA      a graph walk whose iteration count cuVS chooses internally. There
             is no honest way to count its distance evaluations from outside,
             so CAGRA carries throughput and latency only and is absent from the
             traffic panels rather than being given a guess.
"""
import argparse, json, time
from pathlib import Path

import numpy as np


def make_batch(Q0, nq, draw, rng):
    """A batch of nq queries drawn from the test set, not its first nq rows.

    Taking a prefix makes batch size and query difficulty the same variable:
    batch=1 would be one particular query's cost and batch=10000 the whole set's
    average. Each draw samples uniformly instead, and small batches are measured
    over several draws, so what the curve shows is the batch effect alone.
    """
    idx = rng.choice(Q0.shape[0], size=min(nq, Q0.shape[0]), replace=False)
    if nq > Q0.shape[0]:
        extra = rng.choice(Q0.shape[0], size=nq - Q0.shape[0], replace=True)
        idx = np.concatenate([idx, extra])
    return idx


def draws_for(nq):
    """More draws where one draw says least."""
    return 16 if nq <= 1 else 8 if nq <= 10 else 4 if nq <= 100 else 2 if nq <= 1000 else 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--settings", type=Path, required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    from pylibraft.common import device_ndarray
    from cuvs.neighbors import ivf_flat, ivf_pq, cagra, refine as cuvs_refine
    try:
        from cuvs.common import Resources
    except ImportError:
        from pylibraft.common import DeviceResources as Resources

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = "inner_product" if ip else "sqeuclidean"
    X = np.load(a.data / "X.npy")
    n, d = X.shape
    Q0 = np.load(a.data / "Q_test.npy")
    Xd = device_ndarray(X)
    settings = json.loads(a.settings.read_text())[a.corpus]
    res = Resources()

    rows = []
    cache = {}

    def timed(make_fn, nq):
        """Median over draws of the best over repeats.

        The repeats remove scheduling noise within one batch; the draws remove
        the difference between one sample of the query set and another, which
        at batch=1 is the whole measurement.
        """
        rng = np.random.default_rng(0)
        per_draw = []
        for dr in range(draws_for(nq)):
            qi = make_batch(Q0, nq, dr, rng)
            Qd = device_ndarray(np.ascontiguousarray(Q0[qi]))
            fn = make_fn(Qd)
            best = None
            for rep in range(a.repeats + 1):
                t = time.perf_counter()
                fn()
                ms = (time.perf_counter() - t) * 1e3
                if rep:
                    best = ms if best is None else min(best, ms)
            per_draw.append(best)
        per_draw.sort()
        return per_draw[len(per_draw) // 2]

    for target, per_sys in settings.items():
        # ---- CAGRA
        s = per_sys.get("CAGRA")
        if s:
            arm = s.get("arm", "built")
            key = ("cagra", arm)
            if key not in cache:
                if arm == "from_graph":
                    G = np.load(a.data / a.graph).astype(np.uint32)
                    cache[key] = cagra.from_graph(device_ndarray(np.ascontiguousarray(G)),
                                                  Xd, metric=metric, resources=res)
                else:
                    ip_ = cagra.IndexParams(metric=metric,
                                            graph_degree=man.get("deg", 64))
                    cache[key] = cagra.build(ip_, Xd)
            idx = cache[key]
            sp = cagra.SearchParams(itopk_size=int(s["itopk"]),
                                    search_width=int(s.get("width", 1)))
            for nq in a.batches:
                ms = timed(lambda Qd: (lambda: cagra.search(sp, idx, Qd, a.k)), nq)
                rows.append(dict(system="CAGRA", target=float(target), batch=nq,
                                 latency_ms=ms, qps=nq / (ms / 1e3),
                                 draws=draws_for(nq), source="none"))
                print(f"{a.corpus} CAGRA t={target} b={nq:<6} {ms:9.3f} ms "
                      f"{nq/(ms/1e3):10,.0f} qps")

        # ---- IVF-FLAT and IVF-PQ
        for sysname in ("IVF-FLAT", "IVF-PQ (wo/ref)", "IVF-PQ (w/ref)"):
            s = per_sys.get(sysname)
            if not s:
                continue
            lists, pr = int(s["lists"]), int(s["n_probes"])
            flat = sysname == "IVF-FLAT"
            key = ("flat" if flat else "pq", lists)
            if key not in cache:
                cache[key] = (ivf_flat.build(ivf_flat.IndexParams(n_lists=lists,
                                                                  metric=metric), Xd)
                              if flat else
                              ivf_pq.build(ivf_pq.IndexParams(n_lists=lists,
                                                              metric=metric,
                                                              pq_bits=8), Xd))
            idx = cache[key]
            sp = (ivf_flat.SearchParams(n_probes=pr) if flat
                  else ivf_pq.SearchParams(n_probes=pr))
            mult = float(s.get("refine", 0.0) or 0.0)
            kk = a.k if mult == 0 else int(a.k * mult)
            per_list = n / lists
            code = d * 4 if flat else max(1, d // 4)
            bpq = lists * d * 4 + pr * per_list * code
            fpq = (lists + pr * per_list) * d * 2
            if mult:
                bpq += kk * d * 4
                fpq += kk * d * 2
            for nq in a.batches:
                def make(Qd):
                    def run():
                        if flat:
                            _, i = ivf_flat.search(sp, idx, Qd, kk)
                        else:
                            _, i = ivf_pq.search(sp, idx, Qd, kk)
                        if mult:
                            cuvs_refine(Xd, Qd, i, a.k, metric=metric)
                    return run

                ms = timed(make, nq)
                rows.append(dict(system=sysname, target=float(target), batch=nq,
                                 latency_ms=ms, qps=nq / (ms / 1e3),
                                 bytes_per_query=bpq, flops_per_query=fpq,
                                 gb_per_s=bpq * nq / 1e9 / (ms / 1e3),
                                 draws=draws_for(nq), source="modelled"))
                print(f"{a.corpus} {sysname:16} t={target} b={nq:<6} {ms:9.3f} ms "
                      f"{nq/(ms/1e3):10,.0f} qps")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(corpus=a.corpus, n=n, d=d, rows=rows), indent=2))
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
