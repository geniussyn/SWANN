"""Choose the entry catalogue size P offline: no queries, no ground truth.

P is a real knob. Raising it from the shipped 4096 is worth 1-19% of end-to-end
time on MS MARCO 1M, 20-22% on OpenAI-1536 and 10-19% on OpenAI-3072, all at
matched recall (bench/catalogue_size.py). But the sweep that finds it measures
recall, recall needs queries, and on a corpus whose only query split is the test
split that would make P a parameter tuned on test.

So P is not swept here. It is read off a budget, and the budget is query-free:

    P* = the largest P whose catalogue scan stays under SHARE_CAP of the total

Two measured facts are what make that the right rule rather than a shortcut.

**P is a one-sided knob.** At a fixed eps, a larger catalogue adds recall and
costs only the scan: on MS MARCO at eps=0.03, going from P=4096 to P=32768 takes
recall 0.9792 to 0.9859 for 1.45 ms of a 27 ms batch. It never trades recall
away. So the only question is how much scan to buy, which is a budget question
and not a search.

**What sets the optimum is the operating point, not d and not N.** Within one
corpus the argmin moves by two orders of magnitude as the target approaches the
corpus ceiling, because near the ceiling eps buys recall very expensively and P
buys it cheaply:

    corpus         d       N      target   argmin P
    GloVe        104   1.18M      0.930      2048
    GloVe        104   1.18M      0.950      8192
    GloVe        104   1.18M      0.965     65536
    MS MARCO    1024   1.00M      0.980      2048
    MS MARCO    1024   1.00M      0.985      8192
    MS MARCO    1024   1.00M      0.990     32768..131072
    MS MARCO    1024   1.00M      0.994    262144
    NQ          1024    500K      0.996     16384
    NQ          1024    500K      0.997     65536
    OpenAI-1536 1536    995K      0.997    131072
    OpenAI-3072 3072    995K      0.998    131072

Reading that by d or by N gives the wrong answer -- GloVe at d=104 wants 65536 at
its own high-recall end, more than MS MARCO at d=1024 wants at 0.985. Reading it
by the scan share gives the right one: every argmin above sits at a scan share
between 3% and 17%, and the point where a larger P starts losing is where the
share passes about 20% (GloVe at P=131072: 26% share, and 11% worse than
P=65536).

So the rule takes the operating point -- eps, which this system already tunes on
a validation split -- times the search, and spends up to SHARE_CAP of the total
on the catalogue. Both terms are timed here without a single correct answer,
because timing does not depend on recall: random corpus vectors stand in as
query inputs. The optimum is flat besides (at 1M and recall 0.990 the total
varies 8% across a 16x range of P), so landing one ladder step off costs a few
percent.

Why queries cannot be simulated. The obvious trick -- hold out M corpus
documents and use each as a stand-in query, as `gpuann/graph_build.py` does for
the fragmentation index -- does not work for this, and the reason is worth
recording because it is not obvious and it looks like it works. Document
stand-ins are not easier because their answers are less fragmented: measured on
MS MARCO 1M, phi_1 of a document's own top-10 is 3.261 (sd 2.033) against 3.018
(sd 2.040) for real queries, so they are if anything *more* fragmented. They are
easier because **the graph was built to make exactly that set reachable**. A
document's own adjacency list already contains 63.7% of its exact top-10, and
17.2% of documents have 9 of 10 in it; for a real query the best single answer
node covers 50.0% and only 1.0% of queries have 9 of 10. Handed its own
neighbourhood to retrieve, the search recovers it whatever P is: measured recall
0.9995-1.0000 at every P and every eps, which ranks nothing.

(There is also a trap underneath that one. A stand-in query is a corpus node, so
the search returns it as its own nearest neighbour while its ground truth
excludes it, and recall comes out as exactly 0.9000 for every configuration --
nine of ten, every time. That is not a result.)

    python offline/choose_P.py --data /dev/shm/gpuann/openai-3072
    python offline/choose_P.py --data /dev/shm/gpuann/msmarco-1m --validate val
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                 # noqa: E402

LADDER = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144]
SHARE_CAP = 0.20        # of selection + search; the point where a larger P starts
                        # losing is a share near 0.25 (GloVe at P=131072: 26%)
REF_P = 8192            # the scan is linear in P, so one timing fixes the rest


def snap(P, N):
    """The nearest ladder step in log space, capped at an eighth of the corpus.

    Nearest and not downward: the budget lands between two steps as often as
    not, and rounding down cost GloVe a step and 11% (it wants 65536, the budget
    computed 63950, and rounding down gave 32768).

    The N // 8 cap is deliberate and is not about speed. At P = 131072 over a
    995K corpus the catalogue covers 13% of the corpus and already holds 13% of
    each query's top-k, so the configuration is a hybrid of a graph search and a
    flat scan and has to be named as one. An eighth keeps the held share near
    3%, where the question does not arise. On the OpenAI corpora the
    unconstrained argmin is one step above this cap, so the cap costs something
    there and the reason it is paid is honesty about what is being measured.
    """
    cap = min(P, N // 8)
    ok = [x for x in LADDER if x <= cap]
    if not ok:
        return LADDER[0]
    if len(ok) == len(LADDER) or ok[-1] == cap:
        return ok[-1]
    hi = LADDER[len(ok)]
    lo = ok[-1]
    return hi if (hi <= N // 8 and abs(np.log(hi / P)) < abs(np.log(lo / P))) else lo


def recall_at_k(got, gt):
    k = gt.shape[1]
    return float(np.mean([len(set(g[g >= 0].tolist()) & set(t.tolist())) / k
                          for g, t in zip(got, gt)]))


def timed(fn, repeats=3):
    best = None
    for r in range(repeats + 1):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = fn()
        torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
        if r and (best is None or ms < best[0]):
            best = (ms, out)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--graph", default=None)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--eps", type=float, default=0.10,
                    help="the operating point the cost check is made at")
    ap.add_argument("--Q", type=int, default=2000, help="batch the choice is for")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--validate", default=None,
                    help="a real query split: sweep the ladder and report its argmin")
    ap.add_argument("--targets", type=float, nargs="+", default=None)
    ap.add_argument("--write", action="store_true",
                    help="write the chosen catalogue as pools/pool_auto.npy and "
                         "record the choice in P_choice.json, so a bench can take "
                         "it with --pool auto instead of carrying P on the command "
                         "line")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    graph = a.graph or ("G_ptf32.npy" if (a.data / "G_ptf32.npy").exists()
                        else "G_d64.npy")
    Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / graph))).cuda()
    dg = engine.graph_degrees(Gg)
    N, d = Xg.shape
    Xs = Xg.half().contiguous() if a.half else Xg
    print(f"{man['corpus']}  n={N:,}  d={d}  graph={graph}"
          + ("  X=fp16" if a.half else "  X=fp32"))

    # Time both halves at a reference P. No ground truth is read and none is
    # needed: random corpus vectors stand in as query inputs because timing does
    # not depend on whether the answers are right.
    g = torch.Generator(device="cpu").manual_seed(0)
    qi = torch.randperm(N, generator=g)[: a.Q].to(Xg.device)
    Qg = Xg[qi].contiguous()
    refP = min(REF_P, N // 8)
    pool = np.sort(np.random.default_rng(1).choice(N, size=refP, replace=False))
    pool_t = torch.from_numpy(pool.astype(np.int32)).cuda()
    catX = select_a.pool_vectors(Xg, pool_t)
    ref_scan, near = timed(lambda: select_a.select_nearest(
        Qg, catX, a.V, metric=metric, precision="tf32"))
    seeds = pool_t[near].unsqueeze(-1).contiguous()
    srch_ms, _ = timed(lambda: engine.search(
        Xs, Gg, Qg, seeds, k=10, L=a.L, B=a.B, metric=metric,
        policy=engine.POLICY_BEST, stop_eps=a.eps, deg=dg,
        shared_cache=True, exclusive=True, shared_bound=True))
    # the scan is linear in P, so the budget solves in closed form
    budget_ms = srch_ms * SHARE_CAP / (1.0 - SHARE_CAP)
    P = snap(refP * budget_ms / ref_scan, N)
    scan_at_P = ref_scan * P / refP
    share = scan_at_P / (scan_at_P + srch_ms)
    print(f"timed at Q={a.Q}, eps={a.eps}: search {srch_ms:8.2f} ms, "
          f"scan {ref_scan:6.3f} ms at the reference P={refP:,}")
    print(f"budget: {100*SHARE_CAP:.0f}% of the total is {budget_ms:.2f} ms of scan")
    print(f"  -> P = {P:,}   (P/N = {P/N:.4f}, N/P = {N/P:.0f}, "
          f"scan {scan_at_P:.2f} ms, share {100*share:.1f}%)\n")
    out = dict(corpus=man["corpus"], n=int(N), d=int(d), graph=graph, V=a.V,
               L=a.L, B=a.B, eps=a.eps, Q=a.Q, half=bool(a.half),
               ref_P=int(refP), ref_scan_ms=ref_scan, search_ms=srch_ms,
               share_cap=SHARE_CAP, chosen_P=int(P),
               scan_ms_at_P=scan_at_P, scan_share=share)
    del catX, pool_t
    torch.cuda.empty_cache()

    if a.validate:
        print(f"\nvalidation against the real {a.validate} split "
              f"(not used to choose P, only to check the rule)")
        Qv = np.ascontiguousarray(np.load(a.data / f"Q_{a.validate}.npy")[: a.Q])
        gtv = np.load(a.data / f"gt_{a.validate}_ids.npy")[: a.Q]
        Qvg = torch.from_numpy(Qv).cuda()
        rows = []
        for Pv in [p for p in LADDER if p <= N // 8]:
            pl = np.sort(np.random.default_rng(1).choice(N, size=Pv, replace=False))
            pt = torch.from_numpy(pl.astype(np.int32)).cuda()
            cX = select_a.pool_vectors(Xg, pt)
            for eps in (a.eps / 3, a.eps, a.eps * 2):
                s_ms, nr = timed(lambda: select_a.select_nearest(
                    Qvg, cX, a.V, metric=metric, precision="tf32"))
                sd = pt[nr].unsqueeze(-1).contiguous()
                g_ms, ids = timed(lambda: engine.search(
                    Xs, Gg, Qvg, sd, k=10, L=a.L, B=a.B, metric=metric,
                    policy=engine.POLICY_BEST, stop_eps=eps, deg=dg,
                    shared_cache=True, exclusive=True, shared_bound=True)[0])
                rc = recall_at_k(ids.cpu().numpy(), gtv)
                rows.append(dict(P=Pv, eps=eps, recall=rc, scan_ms=s_ms,
                                 search_ms=g_ms, total_ms=s_ms + g_ms))
            del cX, pt
            torch.cuda.empty_cache()
        tg = a.targets or [round(max(r["recall"] for r in rows) - x, 4)
                           for x in (0.006, 0.003)]
        print(f"  {'target':>8s}  " + "  ".join(f"{p:>9d}" for p in sorted({r['P'] for r in rows})))
        for t in tg:
            cells, bp, bt = [], None, float("inf")
            for Pv in sorted({r["P"] for r in rows}):
                pts = sorted({(r["recall"], r["total_ms"]) for r in rows if r["P"] == Pv})
                f, best = [], float("inf")
                for rc, ms in sorted(pts, key=lambda p: (-p[0], p[1])):
                    if ms < best:
                        f.append((rc, ms)); best = ms
                f = sorted(f)
                v = None
                if f and f[0][0] <= t <= f[-1][0]:
                    for (r0, t0), (r1, t1) in zip(f, f[1:]):
                        if r0 <= t <= r1:
                            v = t0 if r1 == r0 else t0 + (t1-t0)*(t-r0)/(r1-r0)
                            break
                cells.append("       --" if v is None else f"{v:9.2f}")
                if v is not None and v < bt:
                    bt, bp = v, Pv
            print(f"  {t:8.4f}  " + "  ".join(cells) +
                  (f"   argmin P={bp:,}" + ("  == budget" if bp == P else
                                            f"  (budget chose {P:,})") if bp else ""))
        out["validate_split"] = a.validate
        out["validate_rows"] = rows

    if a.write:
        pdir = a.data / "pools"
        pdir.mkdir(parents=True, exist_ok=True)
        chosen = np.sort(np.random.default_rng(1).choice(N, size=P, replace=False))
        np.save(pdir / "pool_auto.npy", chosen.astype(np.int32))
        (a.data / "P_choice.json").write_text(json.dumps(
            {k: out[k] for k in ("corpus", "n", "d", "graph", "V", "L", "B", "eps",
                                 "Q", "half", "ref_P", "ref_scan_ms", "search_ms",
                                 "share_cap", "chosen_P", "scan_ms_at_P",
                                 "scan_share")}, indent=2))
        print(f"wrote {pdir / 'pool_auto.npy'} ({P:,} entries) and "
              f"{a.data / 'P_choice.json'}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
