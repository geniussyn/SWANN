"""SWANN against native CAGRA at 1M, at matched recall and matched precision.

Each side on its own graph and its own knobs: SWANN over eps at two catalogue
sizes, CAGRA's frontier read from the JSON bench/cagra_native.py wrote on the
same 2000 queries. Selection is charged to SWANN, because an entry policy that
is free only in the search kernel is not free.

CAGRA's frontier has to be dense before it is interpolated: with itopk stepping
64/128/256/512 the gaps straddle the targets and the interpolation flatters us.
Run bench/cagra_native.py twice, once with the coarse ladder and once with
--itopk 192 256 320 384 448 512 640 768 896 1024, and pass both files. Note that
single_cta raises above itopk 512 (shared memory), so multi_cta carries the top
of the range.

Measured at matched recall and matched precision, each side on its own graph and
its own knobs, selection charged to SWANN. "P=auto" is the catalogue
offline/choose_P.py wrote; P=4096 is what the system shipped before.

  fp32                   d      N     Q   recall   CAGRA ms   P=4096    P=auto
  GloVe                104  1.18M  2000   0.950       13.63    1.12x     1.15x
  GloVe                                   0.960       16.76    1.14x     1.20x
  GIST                 960   1.00M  500   0.980       15.87    1.47x       --
  GIST                                    0.990       21.76    1.58x     1.57x
  MS MARCO            1024   1.00M 2000   0.985      103.16    3.36x     3.93x
  MS MARCO                                0.989      132.44    3.69x     4.08x
  MS MARCO                                0.992      189.34    4.13x     4.83x
  NQ                  1024    500K 1726   0.994       80.36    3.08x     3.81x
  NQ                                      0.996      131.91    4.32x     5.13x
  NQ                                      0.997      173.00    5.02x     5.67x
  FEVER               1024   1.00M 2000   0.970      146.00    2.90x     3.56x
  FEVER                                   0.980      207.83    2.82x     3.49x
  OpenAI-1536 (DBp.)  1536    995K 2000   0.997      418.16    7.47x     9.17x
  OpenAI-3072 (DBp.)  3072    995K 2000   0.9967    1384.41   13.84x    16.57x

  fp16 (both sides)                       recall   CAGRA ms   P=4096    P=auto
  MS MARCO                                0.989       63.31    3.22x     3.37x
  MS MARCO                                0.992       85.55    3.47x     3.77x
  NQ                                      0.996       58.64    3.56x     3.82x
  NQ                                      0.997       69.14    3.48x     3.95x
  FEVER                                   0.970       65.52    2.53x     2.84x
  FEVER                                   0.980       89.10    2.48x     2.79x

The margin grows with the dimension and with the recall target, and the
catalogue is worth 0.9-1.3x of it. GIST is measured at Q=500, which is its whole
validation split and below the batch crossover, so it understates.

For scale, bench/brute_force_floor.py gives an exact tf32 matmul at 64.4 ms for
recall 0.9998 on the same batch -- so above about 0.985 native CAGRA at this size
is slower than brute force, and SWANN is the only one of the three that is not.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a
from gpuann.pools import load_pool


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


ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--Ps", type=int, nargs="+", default=[4096, 0],
                help="0 means the catalogue offline/choose_P.py wrote (pool_auto)")
ap.add_argument("--epss", type=float, nargs="+",
                default=[0.02, 0.04, 0.06, 0.09, 0.13, 0.18, 0.26])
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--nq", type=int, default=2000)
ap.add_argument("--split", default=None)
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--cagra-fp32", type=Path, nargs="+", default=[Path("results/degree/cagra_1m_fp32.json"), Path("results/degree/cagra_1m_fp32_dense.json")])
ap.add_argument("--cagra-fp16", type=Path, nargs="+", default=[Path("results/degree/cagra_1m_fp16.json"), Path("results/degree/cagra_1m_fp16_dense.json")])
ap.add_argument("--targets", type=float, nargs="+",
                default=[0.95, 0.965, 0.978, 0.985, 0.989, 0.992])
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Xh = Xg.half().contiguous()
split = a.split or ("val" if (a.data / "Q_val.npy").exists() else "test")
Q = np.ascontiguousarray(np.load(a.data / f"Q_{split}.npy")[: a.nq])
gt = np.load(a.data / f"gt_{split}_ids.npy")[: a.nq]
Qg = torch.from_numpy(Q).cuda()
Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
dg = engine.graph_degrees(Gg)

cag = {}
for tag, f in (("fp32", a.cagra_fp32), ("fp16", a.cagra_fp16)):
    pts = []
    for one in f:
        j = json.loads(Path(one).read_text())
        pts += [(r["recall"], r["batch_ms"]) for r in j["rows"]]
    cag[tag] = pareto(pts)
    print(f"CAGRA {tag} frontier ({len(cag[tag])} pts): " +
          "  ".join(f"{r:.4f}@{t:.1f}" for r, t in cag[tag]))
print()

ours = {}
for prec, X in (("fp32", Xg), ("fp16", Xh)):
    for P in a.Ps:
        if P == 0:                      # the catalogue offline/choose_P.py chose
            pool = np.load(a.data / "pools" / "pool_auto.npy").astype(np.int32)
        else:
            pool = load_pool(a.data, "random", P)
        pool_t = torch.from_numpy(pool).cuda()
        catX = select_a.pool_vectors(Xg, pool_t)
        pts = []
        for eps in a.epss:
            sb = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                near = select_a.select_nearest(Qg, catX, a.V, metric=metric,
                                               precision="tf32")
                torch.cuda.synchronize(); sel = (time.perf_counter() - t) * 1e3
                if r and (sb is None or sel < sb[0]):
                    sb = (sel, near)
            sel_ms, near = sb
            seeds = pool_t[near].unsqueeze(-1).contiguous()
            gb = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                ids, dist, st = engine.search(
                    X, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                    shared_cache=True, exclusive=True, shared_bound=True)
                torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
                if r and (gb is None or ms < gb[0]):
                    gb = (ms, ids)
            srch_ms, ids = gb
            rc = recall_at_k(ids.cpu().numpy(), gt)
            pts.append((rc, sel_ms + srch_ms))
            del ids, dist, st
        ours[(prec, P)] = pareto(pts)
        print(f"SWANN {prec} P={len(pool):<6d}: " +
              "  ".join(f"{r:.4f}@{t:.1f}" for r, t in ours[(prec, P)]))
        del catX, pool_t
        torch.cuda.empty_cache()

for prec in ("fp32", "fp16"):
    print(f"\n=== {prec}: SWANN against native CAGRA at matched recall "
          f"(1M, Q={a.nq}, selection charged to SWANN) ===")
    hdr = "  ".join(("P=auto " if p == 0 else f"P={p:<6d}") for p in a.Ps)
    print(f"{'recall':>8s} {'CAGRA ms':>9s}  " +
          "  ".join(f"{'ms':>7s} {'x':>7s}" for _ in a.Ps))
    for tgt in a.targets:
        tc = time_at(cag[prec], tgt)
        if tc is None:
            continue
        cells = []
        for P in a.Ps:
            to = time_at(ours[(prec, P)], tgt)
            cells.append(f"{'--':>7s} {'--':>7s}" if to is None
                         else f"{to:7.2f} {tc/to:6.2f}x")
        print(f"{tgt:8.4f} {tc:9.2f}  " + "  ".join(cells))
    print(f"         (columns: {hdr})")
