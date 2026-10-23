"""Batch-size scaling for the systems that live in the torch process.

Throughput and latency are measured. The traffic and arithmetic columns are
not measured the same way for both arms and the difference matters:

  SWANN       counted. The kernel reports how many distances it evaluated, and
              every one of them read a whole vector, so the bytes follow from a
              tally the hardware actually produced.
  IVF-RaBitQ  modelled. cuVS exposes no counter and the profiler is unavailable
              on this machine, so the bytes are what the structure forces: the
              centroid scan every query pays, plus the probed lists read at
              bits_per_dim. A real implementation reads at least this.

Latency is the batch's own wall clock, which is what a caller waits, not a
per-query division of a large batch.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch


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
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 10, 100, 1000, 10000])
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from swann import _engine as engine, _select as select_a
    from swann._pools import load_pool

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    n, = (man["n"],)
    Xf = np.load(a.data / "X.npy")
    d = Xf.shape[1]
    X32 = torch.from_numpy(Xf).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q0 = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X32, pt)
    settings = json.loads(a.settings.read_text())[a.corpus]

    rows = []
    for target, per_sys in settings.items():
        for sysname in ("SWANN fp32", "SWANN fp16"):
            s = per_sys.get(sysname)
            if not s:
                continue
            half = sysname.endswith("fp16")
            X = X32.half() if half else X32
            V, B, L = int(s["V_per_branch"]), int(s["B"]), int(s["L"])
            eps = float(s["stop_eps"])
            for nq in a.batches:
                rng = np.random.default_rng(0)
                per_draw, de_draw = [], []
                for dr in range(draws_for(nq)):
                    idx = make_batch(Q0, nq, dr, rng)
                    # Only the corpus is halved. The kernel keeps the query in
                    # fp32 in shared memory, so a half query is not merely
                    # pointless here -- it is rejected.
                    Qs = Q0[torch.from_numpy(idx).cuda()].contiguous()
                    seeds = pt[select_a.select_nearest(Qs, Vp, V, metric=metric)].to(torch.int32)
                    best, de = None, None
                    for rep in range(a.repeats + 1):
                        torch.cuda.synchronize(); t = time.perf_counter()
                        r = engine.search(X, G, Qs, seeds, k=a.k, L=L, B=B, metric=metric,
                                          team_size=a.team, stop_eps=eps,
                                          shared_cache=True, shared_bound=True,
                                          exclusive=True, elide_visited=True)
                        torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                        if rep:
                            best = ms if best is None else min(best, ms)
                        if de is None and isinstance(r, tuple) and len(r) > 2:
                            st = r[2]
                            if hasattr(st, "dist_evals"):
                                de = float(st.dist_evals.sum().item()) / nq
                    per_draw.append(best)
                    if de is not None:
                        de_draw.append(de)
                    del Qs, seeds
                # the median draw, so one unusually easy or hard sample of the
                # query set does not become the reported number
                per_draw.sort()
                best = per_draw[len(per_draw) // 2]
                de = (sorted(de_draw)[len(de_draw) // 2] if de_draw
                      else float(s.get("dist_per_q", float("nan"))))
                esz = 2 if half else 4
                bpq = de * d * esz
                rows.append(dict(system=sysname, target=float(target), batch=nq,
                                 latency_ms=best, qps=nq / (best / 1e3),
                                 dist_per_query=de, bytes_per_query=bpq,
                                 flops_per_query=de * d * 2,
                                 gb_per_s=bpq * nq / 1e9 / (best / 1e3),
                                 draws=draws_for(nq), source="counted"))
                print(f"{a.corpus} {sysname:11} t={target} b={nq:<6} "
                      f"{best:9.3f} ms {nq/(best/1e3):10,.0f} qps "
                      f"{bpq*nq/1e9/(best/1e3):7.0f} GB/s")
            if half:
                del X
                torch.cuda.empty_cache()

    # --- IVF-RaBitQ, in the same process because the extension is a torch one
    try:
        sys.path.insert(0, "bench")
        from cuvs_rabitq import ext
        m = ext()
    except Exception as e:
        print(f"(RaBitQ extension unavailable: {type(e).__name__})")
        m = None
    if m is not None:
        MODE = {"IVF-RaBitQ (LUT)": 0, "IVF-RaBitQ (Bitwise)": 1}
        built = {}
        for target, per_sys in settings.items():
            for sysname, md in MODE.items():
                s = per_sys.get(sysname)
                if not s:
                    continue
                bits, lists = int(s["bits"]), int(s["lists"])
                pr = int(s["n_probes"])
                if (bits, lists) not in built:
                    built[(bits, lists)] = m.build(X32, lists, bits, 25, ip)
                idx = built[(bits, lists)]
                per_list = n / lists
                # centroids every query, then the probed lists at bits/dim
                bpq = lists * d * 4 + pr * per_list * d * bits / 8
                fpq = (lists + pr * per_list) * d * 2
                for nq in a.batches:
                    rng = np.random.default_rng(0)
                    per_draw = []
                    for dr in range(draws_for(nq)):
                        qi = make_batch(Q0, nq, dr, rng)
                        Q = Q0[torch.from_numpy(qi).cuda()].contiguous()
                        best = None
                        for rep in range(a.repeats + 1):
                            torch.cuda.synchronize(); t = time.perf_counter()
                            m.search(idx, Q, a.k, pr, md)
                            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                            if rep:
                                best = ms if best is None else min(best, ms)
                        per_draw.append(best)
                        per_draw.sort()
                    best = per_draw[len(per_draw) // 2]
                    rows.append(dict(system=sysname, target=float(target), batch=nq,
                                     latency_ms=best, qps=nq / (best / 1e3),
                                     bytes_per_query=bpq, flops_per_query=fpq,
                                     gb_per_s=bpq * nq / 1e9 / (best / 1e3),
                                     draws=draws_for(nq), source="modelled"))
                    print(f"{a.corpus} {sysname:21} t={target} b={nq:<6} "
                          f"{best:9.3f} ms {nq/(best/1e3):10,.0f} qps")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(corpus=a.corpus, n=n, d=d, rows=rows), indent=2))
    print(f"wrote {a.out}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
