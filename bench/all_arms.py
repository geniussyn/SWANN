"""Every stored format on one frontier, against CAGRA and IVF-RaBitQ.

Six lines, each swept over its own knob and read at matched recall:

  SWANN fp32 / fp16 / 4-bit+rescore   eps, selection charged in
  native cuVS CAGRA                   itopk_size x search_width x algorithm
  cuVS IVF-RaBitQ at 7 and 1 bits     probes, to all 4096 lists

The 4-bit line is the kernel, not a simulation: the corpus is packed two values
to a byte with a per-vector scale after a fixed rotation (engine.pack4), the
query stays fp32 and is rotated once a batch, and the queue is rescored against
fp16 before the answer is returned (engine.rerank). The rescore is what the line
needs to exist -- raw 4-bit tops out near 0.919 -- and it costs about 2.4 ms of
19 at d = 1024.

The CAGRA and RaBitQ frontiers are read from the JSONs bench/cagra_native.py and
bench/cuvs_rabitq.py wrote on the same queries; neither is re-run here.

Measured, batch ms at matched recall. NOTE: the table below was recorded
before selection was moved inside the timed region, so every SWANN cell in it
is short by the selection cost of its catalogue (3.4 ms at P = 65536,
Q = 2000, d = 1024). Re-run before quoting.

  MS MARCO 1M, d=1024, 2000 val queries
    recall   fp32    fp16   4-bit+rr64    CAGRA   RaBitQ-7   RaBitQ-1
    0.950      --      --         --      27.35      45.22       --
    0.985   23.18   12.10      10.87     103.16     109.59       --
    0.990   30.55   16.02      13.42     147.66     159.69       --
    0.993   38.44   21.01      16.52     221.51         --       --
    ceiling 0.9955  0.9954     0.9951    0.9955     0.9929   0.8015

  NQ 500K, d=1024, 1726 val queries
    0.985      --      --         --      35.87      28.44       --
    0.994   19.41   10.28       9.14      80.36      77.15       --
    0.996   25.77   12.98      10.83     131.91         --       --
    ceiling 0.9984  0.9982     0.9985    0.9981     0.9942   0.8147

  FEVER 1M, d=1024, 2000 val queries
    0.970   37.80   19.19      14.74     146.00      55.73       --
    0.980   54.94   28.18      19.90     207.83      74.98       --
    0.985   73.71   36.37      25.15     274.57      95.26       --
    ceiling 0.9873  0.9871     0.9865    0.9875     0.9924   0.7536

So 4-bit leads everywhere above 0.97 and the 1-bit baseline reaches none of
these targets: its ceiling is 0.75-0.82 whatever the probe count, which is the
sign code's own limit and not a tuning failure. RaBitQ-7 is the one baseline
that competes below 0.99 and it is ahead of CAGRA on two of the three.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def pareto(pts):
    out, best = [], float("inf")
    for rc, t in sorted(set(pts), key=lambda p: (-p[0], p[1])):
        if t < best:
            out.append((rc, t)); best = t
    return sorted(out)


def time_at(front, rc):
    if not front or rc < front[0][0] or rc > front[-1][0]:
        return None
    for (r0, t0), (r1, t1) in zip(front, front[1:]):
        if r0 <= rc <= r1:
            return t0 if r1 == r0 else t0 + (t1 - t0) * (rc - r0) / (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_ptf32.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--configs", nargs="*", default=["32:16:16"],
                    help="V:L:B triples. One frontier per stored width is built "
                         "over all of them crossed with --epss, because no single "
                         "V spans 0.9 to 0.999: a small V cannot reach the top "
                         "and a large one cannot be made cheap enough for the "
                         "bottom.")
    ap.add_argument("--epss", type=float, nargs="+",
                    default=[0.02, 0.04, 0.07, 0.11, 0.16, 0.24])
    ap.add_argument("--pools", nargs="+", default=["pool_auto"],
                    help="catalogue files under <data>/pools, without .npy. The "
                         "catalogue is a knob like V and eps -- its scan costs the "
                         "same whatever the query needs, while the search it seeds "
                         "spans two orders of magnitude -- so the frontier is built "
                         "over all of them with each point charged its own "
                         "selection, and a small catalogue wins below recall 0.99.")
    ap.add_argument("--bits", type=int, nargs="*", default=[4],
                    help="packed widths to measure with the kernel: 4 and/or 2")
    ap.add_argument("--depth", type=int, default=32, help="rescore depth; the knee is "
                    "at 24 and it saturates by 32 (see engine.rerank)")
    ap.add_argument("--sign-depths", type=int, nargs="*", default=[],
                    help="BBQ on SWANN: traverse on 1-bit sign codes (engine.pack_sign, "
                         "xbits=1), then rerank the best R of the queue against fp32. "
                         "One arm per R")
    ap.add_argument("--sign-configs", nargs="*", default=["32:16:8", "32:16:12",
                    "32:16:16", "32:32:24", "32:32:32", "32:64:48", "32:64:64"],
                    help="V:L:B for the 1-bit arms; they need a deeper walk than fp")
    ap.add_argument("--sign-epss", type=float, nargs="+", default=[0.11])
    ap.add_argument("--sign-team", type=int, default=4)
    ap.add_argument("--sign-fp", nargs="+", default=["gpu"], choices=["gpu", "host"],
                    help="where the fp32 rerank vectors live: gpu memory, or pinned "
                         "host RAM gathered per batch and copied over PCIe (timed)")
    ap.add_argument("--skip-fp", action="store_true",
                    help="skip the SWANN fp32/fp16/narrow arms")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cagra", type=Path, nargs="*", default=[])
    ap.add_argument("--rabitq", nargs="*", default=[],
                    help="label=path pairs, e.g. 'RaBitQ 7-bit=results/.../r7.json'")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.95, 0.97, 0.985, 0.99, 0.993])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Xh = Xg.half().contiguous()
    Q = np.ascontiguousarray(np.load(a.data / f"Q_{a.split}.npy")[: a.nq])
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq]
    Qg = torch.from_numpy(Q).cuda()
    Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
    dg = engine.graph_degrees(Gg)
    POOLS = [(nm, np.load(a.data / "pools" / f"{nm}.npy"
                          if (a.data / "pools" / f"{nm}.npy").exists()
                          else a.data / f"{nm}.npy").astype(np.int32))
             for nm in a.pools]
    N, d = Xg.shape
    CFG = [tuple(int(x) for x in c.split(":")) for c in a.configs]
    SCFG = [tuple(int(x) for x in c.split(":")) for c in a.sign_configs] \
        if a.sign_depths else []
    ALLV = sorted({c[0] for c in CFG + SCFG})

    # Entry selection is part of answering the query, so it is timed with the
    # search and not before it: the [Q, P] scan costs 3.2 ms at P = 65536,
    # Q = 2000, d = 1024, which is more than the whole search at low recall.
    # Taking it outside the loop (which this script used to do) subtracts a
    # constant from every SWANN point and from no baseline point.
    #
    # Warm the shape before timing it: cuBLAS picks a kernel on the first call
    # for a given [Q, P, d], and a cold call measures 90 ms where a warm one
    # measures 0.4.
    def selection(catX, pool_t):
        seeds_for, sel_ms = {}, {}
        for V in ALLV:
            if V in seeds_for:
                continue
            for _ in range(2):
                select_a.select_nearest(Qg, catX, V, metric=metric, precision="tf32")
            best = None
            for _ in range(a.repeats):
                torch.cuda.synchronize(); t = time.perf_counter()
                nr = select_a.select_nearest(Qg, catX, V, metric=metric,
                                             precision="tf32")
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if best is None or ms < best:
                    best = ms
            seeds_for[V] = pool_t[nr].unsqueeze(-1).contiguous()
            sel_ms[V] = best
        return seeds_for, sel_ms

    print(f"{man['corpus']} n={N:,} d={d} {a.split} q={len(Q)}  "
          f"P={', '.join(f'{len(pl):,}' for _, pl in POOLS)}  "
          f"configs {a.configs}")
    print(f"stored: fp32 {Xg.numel()*4/2**30:.2f} GiB   fp16 "
          f"{Xg.numel()*2/2**30:.2f} GiB"
          + "".join(f"   {b}-bit {Xg.numel()*b/8/2**30:.2f}" for b in a.bits)
          + f"   (+{Xh.numel()*2/2**30:.2f} kept for the rescore)\n")

    arms, rows, keep = {}, [], []
    specs = [] if a.skip_fp else [("SWANN fp32", Xg, Qg, None, None, 4),
                                  ("SWANN fp16", Xh, Qg, None, None, 4)]
    for b in ([] if a.skip_fp else a.bits):
        Pb, scb, rotb = engine.pack_narrow(Xg, b)
        keep.append((Pb, scb))
        specs.append((f"SWANN {b}-bit+rr{a.depth}", Pb,
                      engine.rotate_queries(Qg, rotb), scb, a.depth, b))
    sspecs = []
    if a.sign_depths:
        Ps, scs, _ = engine.pack_sign(Xg)
        Xhost = None
        if "host" in a.sign_fp:
            Xhost = torch.from_numpy(np.load(a.data / "X.npy")).pin_memory()
        for where in a.sign_fp:
            for R in a.sign_depths:
                sspecs.append((f"BBQ-SWANN rr{R} fp@{where}", R, where))
        print(f"1-bit codes {Ps.numel()/2**30:.3f} GiB on GPU; fp32 rerank vectors on "
              + "/".join(a.sign_fp))

    def rerank_host(st, R):
        """engine.rerank's narrowing, then the R fp32 rows per query fetched
        from pinned host memory: ids to host, gather on the CPU, one H2D copy."""
        ids = st.ids.reshape(st.ids.shape[0], -1)
        dist = st.dist.reshape(st.dist.shape[0], -1)
        I = ids.long()
        srt, order = torch.sort(I, dim=1)
        dup = torch.zeros_like(srt, dtype=torch.bool)
        dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
        bad = torch.zeros_like(dup); bad.scatter_(1, order, dup)
        live = (I >= 0) & ~bad
        Dm = torch.where(live, dist, torch.full_like(dist, float("inf")))
        nd = min(R, Dm.shape[1])
        keep = torch.topk(Dm, nd, dim=1, largest=False).indices
        Ik = torch.gather(I, 1, keep); alive = torch.gather(live, 1, keep)
        flat = Ik.clamp(min=0).reshape(-1).cpu()
        rows = torch.empty((flat.numel(), d), dtype=torch.float32).pin_memory() \
            if not hasattr(rerank_host, "buf") or rerank_host.buf.shape[0] < flat.numel() \
            else rerank_host.buf
        rerank_host.buf = rows
        torch.index_select(Xhost, 0, flat, out=rows[: flat.numel()])
        Vg = rows[: flat.numel()].to("cuda", non_blocking=True).view(Ik.shape[0], nd, d)
        dd = 1.0 - torch.bmm(Vg, Qg.unsqueeze(-1)).squeeze(-1)
        dd = torch.where(alive, dd, torch.full_like(dd, float("inf")))
        sel = torch.topk(dd, min(10, nd), dim=1, largest=False).indices
        return torch.gather(Ik, 1, sel).to(torch.int32)

    print(f"{'arm':22s} {'P':>8s} {'V:L:B':>10s} {'eps':>6s} {'recall':>8s} "
          f"{'ms':>8s} {'bytes/q':>11s}")
    pts_by_arm = {tag: [] for tag, *_ in specs}
    pts_by_arm.update({tag: [] for tag, *_ in sspecs})
    for pname, pool in POOLS:
      pool_t = torch.from_numpy(pool).cuda()
      catX = select_a.pool_vectors(Xg, pool_t)
      seeds_for, sel_ms = selection(catX, pool_t)
      print(f"  {pname} (P={len(pool):,}): selection "
            + "  ".join(f"V{V}={sel_ms[V]:.2f}" for V in sorted(sel_ms)))
      for tag, X, Qq, xs, dep, bw in specs:
        pts = pts_by_arm[tag]
        for (V, L, B) in CFG:
            for eps in a.epss:
                best = None
                for r in range(a.repeats + 1):
                    torch.cuda.synchronize(); t = time.perf_counter()
                    ids, dist, st = engine.search(
                        X, Gg, Qq, seeds_for[V], k=10, L=L, B=B, metric=metric,
                        policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                        shared_cache=True, exclusive=True, shared_bound=True,
                        xscale=xs, xbits=bw)
                    fin = engine.rerank(st, Xh, Qg, k=10, depth=dep,
                                        metric=metric) if dep else ids
                    torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                    if r and (best is None or ms < best[0]):
                        best = (ms, fin, st)
                rc = recall_at_k(best[1].cpu().numpy(), gt)
                search_ms, total_ms = best[0], best[0] + sel_ms[V]
                # The work axis, which the throughput axis hides: a narrower
                # vector may buy no time at all and still move far fewer bytes.
                # Counted, not derived -- the kernel tallies its own distance
                # evaluations and each reads one whole stored vector.
                nd = float(best[2].dist_evals.sum()) / len(Q)
                ne = float(best[2].expansions.sum()) / len(Q)
                sb = d * bw / 8 if dep else (d * 4 if X.dtype == torch.float32
                                             else d * 2)
                vec_b = nd * sb
                graph_b = ne * float(dg[0].float().mean() if dg else Gg.shape[1]) * 4
                rr_b = (dep * d * 2) if dep else 0.0     # rescore reads fp16
                pts.append((rc, total_ms))
                rows.append(dict(arm=tag, P=int(len(pool)), pool=pname,
                                 V=V, L=L, B=B, eps=eps, recall=rc,
                                 ms=total_ms, search_ms=search_ms,
                                 select_ms=sel_ms[V],
                                 dist_evals=nd, expansions=ne,
                                 bytes_per_vec=int(sb),
                                 vec_bytes_q=vec_b, graph_bytes_q=graph_b,
                                 rescore_bytes_q=rr_b,
                                 bytes_q=vec_b + graph_b + rr_b,
                                 mflop_q=2.0 * nd * d / 1e6))
                print(f"{tag:22s} {len(pool):8d} {f'{V}:{L}:{B}':>10s} "
                      f"{eps:6.3f} {rc:8.4f} {total_ms:8.2f} "
                      f"{(vec_b + graph_b + rr_b)/1e6:8.2f} MB")
      for tag, R, where in sspecs:
        pts = pts_by_arm[tag]
        for (V, L, B) in SCFG:
            for eps in a.sign_epss:
                best = None
                for r in range(a.repeats + 1):
                    torch.cuda.synchronize(); t = time.perf_counter()
                    ids, dist, st = engine.search(
                        Ps, Gg, Qg, seeds_for[V], k=10, L=L, B=B, metric=metric,
                        policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                        shared_cache=True, exclusive=True, shared_bound=True,
                        xscale=scs, xbits=1, team_size=a.sign_team)
                    fin = engine.rerank(st, Xg, Qg, k=10, depth=R, metric=metric) \
                        if where == "gpu" else rerank_host(st, R)
                    torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                    if r and (best is None or ms < best[0]):
                        best = (ms, fin, st)
                rc = recall_at_k(best[1].cpu().numpy(), gt)
                search_ms, total_ms = best[0], best[0] + sel_ms[V]
                nd = float(best[2].dist_evals.sum()) / len(Q)
                ne = float(best[2].expansions.sum()) / len(Q)
                vec_b = nd * d / 8
                graph_b = ne * float(dg[0].float().mean() if dg else Gg.shape[1]) * 4
                rr_b = R * d * 4
                pts.append((rc, total_ms))
                rows.append(dict(arm=tag, P=int(len(pool)), pool=pname,
                                 V=V, L=L, B=B, eps=eps, recall=rc,
                                 ms=total_ms, search_ms=search_ms,
                                 select_ms=sel_ms[V], depth=R, fp_where=where,
                                 dist_evals=nd, expansions=ne,
                                 bytes_per_vec=int(d / 8),
                                 vec_bytes_q=vec_b, graph_bytes_q=graph_b,
                                 rescore_bytes_q=rr_b,
                                 bytes_q=vec_b + graph_b + rr_b,
                                 mflop_q=2.0 * nd * d / 1e6))
                print(f"{tag:22s} {len(pool):8d} {f'{V}:{L}:{B}':>10s} "
                      f"{eps:6.3f} {rc:8.4f} {total_ms:8.2f} "
                      f"{(vec_b + graph_b + rr_b)/1e6:8.2f} MB", flush=True)
      del catX, pool_t
      torch.cuda.empty_cache()
    # One frontier per arm over every catalogue: P is a knob, so the arm is read
    # at whichever P is cheapest for the recall asked, and the winning P is in
    # the row the point came from.
    for tag in pts_by_arm:
        arms[tag] = pareto(pts_by_arm[tag])
    print()
    best_P = {}
    for r in rows:
        key = (r["arm"], round(r["recall"], 4))
        if key not in best_P or r["ms"] < best_P[key][0]:
            best_P[key] = (r["ms"], r["P"])

    for f in a.cagra:
        j = json.loads(Path(f).read_text())
        arms.setdefault("native CAGRA", [])
        arms["native CAGRA"] = pareto(
            [(r["recall"], r["batch_ms"]) for r in j["rows"]] + arms["native CAGRA"])
    for spec in a.rabitq:
        label, path = spec.split("=", 1)
        j = json.loads(Path(path).read_text())
        arms[label] = pareto([(r["recall"], r["batch_ms"]) for r in j["rows"]])

    names = list(arms)
    print("batch ms at matched recall (-- = outside that arm's range)")
    print(f"{'recall':>8s}  " + "  ".join(f"{n:>22s}" for n in names))
    for t in a.targets:
        cells = []
        for n in names:
            v = time_at(arms[n], t)
            cells.append(f"{'--':>22s}" if v is None else f"{v:22.2f}")
        print(f"{t:8.4f}  " + "  ".join(cells))
    print("\nceiling reached by each arm")
    for n in names:
        print(f"  {n:22s} {arms[n][-1][0]:.4f} at {arms[n][-1][1]:.1f} ms")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, nq=len(Q),
                                         P=len(pool), configs=a.configs,
                                         depth=a.depth, swann_rows=rows,
                                         frontiers={k: v for k, v in arms.items()}),
                                    indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
