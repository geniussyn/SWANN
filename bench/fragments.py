"""Where the recall goes: fragments never reached, or fragments reached and
under-mined.

A query's exact top-k is cut by the serving graph into phi_h pieces. A search
can lose a document two ways, and they want opposite fixes:

  missed fragment    no member of that piece came back at all. More entries, or
                     better-placed entries, is the only thing that helps.
  shallow fragment   the piece was reached but only partly recovered. More
                     budget or a wider queue helps; more entries does not.

Reported together and split by phi, because the average over all queries hides
exactly the queries multi-entry exists for.
"""
import argparse, json
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                           # noqa: E402
from gpuann.pools import load_pool                                  # noqa: E402
from bench.entry_arms import Arms                                   # noqa: E402


def fragment_stats(got, gt, frag):
    """Per query: fragments present, fragments recovered, recall inside them."""
    nq, k = gt.shape
    out = np.zeros((nq, 4))
    for q in range(nq):
        ret = set(int(v) for v in got[q] if v >= 0)
        labels = frag[q]
        nf = int(labels.max()) + 1
        hit_f = inside_hit = inside_tot = 0
        for f in range(nf):
            slots = np.flatnonzero(labels == f)
            found = sum(1 for s in slots if int(gt[q, s]) in ret)
            if found:
                hit_f += 1
                inside_hit += found
                inside_tot += len(slots)
        out[q] = (nf, hit_f, inside_hit, inside_tot)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--graph", default="G.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--pool", default="metis")
    ap.add_argument("--P", type=int, default=512)
    ap.add_argument("--R", type=int, default=128)
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--configs", nargs="+", default=["distance:1:64:128", "distance:8:16:16"],
                    help="arm:V:L:B[:shared-policy]  e.g. distance:8:64:128:rr")
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--bound", action="store_true")
    ap.add_argument("--exclusive", action="store_true")
    ap.add_argument("--stop-eps", type=float, default=-1.0)
    ap.add_argument("--team", type=int, default=32)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    phi = np.load(a.data / f"phi{a.h}_{a.split}_{tag}.npy")
    frag = np.load(a.data / f"frag{a.h}_{a.split}_{tag}.npy")
    if a.nq:
        Q, gt, phi, frag = Q[: a.nq], gt[: a.nq], phi[: a.nq], frag[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, a.R, a.tau, 32, 24)

    buckets = [("phi=1", phi == 1), ("phi=2", phi == 2), ("phi>=3", phi >= 3), ("all", phi >= 1)]
    print(f"graph={a.graph} h={a.h} q={len(phi)}  mean phi={phi.mean():.3f}  "
          f"one piece {100*(phi==1).mean():.1f}%\n")
    rows = []
    for cfg in a.configs:
        parts = cfg.split(":")
        arm, V, L, B = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        shared = len(parts) > 4
        policy = engine.POLICY_ROUND_ROBIN if shared and parts[4] == "rr" else engine.POLICY_BEST
        seeds, sel_ms, _ = arms.seeds(arm, V)
        if shared:
            seeds = seeds.unsqueeze(1)
        ids, _, st = engine.search(Xg, Gg, Qg, seeds, k=a.k, L=L, B=B, policy=policy,
                                   team_size=a.team, stop_eps=a.stop_eps,
                                   shared_cache=a.cache, shared_bound=a.bound,
                                   exclusive=a.exclusive)
        st.check()
        got = ids.cpu().numpy()
        s = fragment_stats(got, gt, frag)
        print(f"--- {arm} V={V} L={L} B={B}"
              + (f" shared/{parts[4]}" if shared else " split") + " ---")
        print(f"{'queries':10s} {'n':>6s} {'recall@10':>10s} {'fragments hit':>14s} "
              f"{'recall in hit':>14s}")
        for name, sel in buckets:
            if not sel.any():
                continue
            ss = s[sel]
            rec = (ss[:, 2] / a.k).mean()
            hitr = (ss[:, 1] / ss[:, 0]).mean()
            inr = (ss[:, 2] / np.maximum(ss[:, 3], 1)).mean()
            print(f"{name:10s} {int(sel.sum()):6d} {rec:10.4f} {hitr:14.4f} {inr:14.4f}")
            rows.append(dict(config=cfg, bucket=name, n=int(sel.sum()), recall=float(rec),
                             frag_hit_rate=float(hitr), recall_in_hit=float(inr),
                             cache=bool(st.cache_id.numel() > 0),
                             bound=bool(st.bound.numel() > 0),
                             exclusive=bool(st.exclusive), stop_eps=a.stop_eps,
                             J=int(st.ids.shape[1])))
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(graph=a.graph, h=a.h, split=a.split, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
