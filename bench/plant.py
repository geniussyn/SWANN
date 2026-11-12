"""Take the answers we actually lose, and plant an entry in the basin of each.

The question is whether a lost answer needs to be travelled to or merely stood
on. A greedy climb stops at a local maximum, and an answer that is its own local
maximum is delivered by any entry under it -- no travel, just coverage. The real
search is not a pure climb (a queue of width L carries candidates worse than the
current best, which is how it crosses saddles), so this has to be measured on
real failures rather than argued: for every missed, alone-in-its-piece answer,
is it its own local maximum, how wide is its basin, and does replacing one of
the 32 selected entries with a catalogue entry from that basin recover it?
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_map import basins                                   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--max-cases", type=int, default=120,
                    help="failures to work through; each needs a full-corpus basin map")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    frag = np.load(a.data / f"frag1_{a.split}_{tag}.npy")[: a.nq]
    N = X.shape[0]

    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    sel_all = pt[select_a.select_nearest(Q, Vp, a.V, metric=metric)]
    ids, _, _ = engine.search(X, G, Q, sel_all.to(torch.int32), k=a.k, L=a.L, B=a.B,
                              metric=metric, team_size=a.team, stop_eps=a.stop_eps,
                              shared_cache=True, shared_bound=True,
                              exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])
    sizes = np.zeros_like(frag)
    for qq in range(len(frag)):
        c = np.bincount(frag[qq]); sizes[qq] = c[frag[qq]]
    qi, ki = np.nonzero((~hit) & (sizes == 1))
    print(f"{a.data.name} {a.graph}  pool {a.pool}(P={a.P}) V={a.V} eps={a.stop_eps}\n"
          f"baseline recall {hit.mean():.4f}; "
          f"{len(qi)} missed, alone-in-piece answers in {a.nq} queries\n"
          f"working through the first {min(a.max_cases, len(qi))}\n")

    own_max = 0; basin_sz = []; recovered = 0; planted = 0; no_entry = 0
    spacing = N // a.P
    for n in range(min(a.max_cases, len(qi))):
        qq, t = int(qi[n]), int(gt[qi[n], ki[n]])
        qv = Q[qq].view(1, -1).contiguous()
        att, s = basins(X, G, qv.view(-1), ip)
        at = int(att[t])
        bs = int((att == at).sum())
        basin_sz.append(bs)
        if at == t:
            own_max += 1
        cand = pt[(att[pt] == at).nonzero().view(-1)]
        if cand.numel() == 0:
            no_entry += 1
            continue
        planted += 1
        seeds = sel_all[qq].clone()
        seeds[-1] = cand[0]                        # give up the farthest entry for it
        out, _, _ = engine.search(X, G, qv, seeds.view(1, -1, 1).to(torch.int32),
                                  k=a.k, L=a.L, B=a.B, metric=metric,
                                  team_size=a.team, stop_eps=a.stop_eps,
                                  shared_cache=True, shared_bound=True,
                                  exclusive=True, elide_visited=True)
        if t in set(int(v) for v in out[0].cpu().numpy() if v >= 0):
            recovered += 1
        del att, s

    m = min(a.max_cases, len(qi))
    b = np.array(basin_sz)
    print(f"the lost answer is its own local maximum       {own_max:5}/{m}  "
          f"{100*own_max/m:5.1f}%")
    print(f"its basin, in nodes                p25 {np.percentile(b,25):,.0f}   "
          f"p50 {np.percentile(b,50):,.0f}   p90 {np.percentile(b,90):,.0f}")
    print(f"  (a P={a.P} catalogue lands one entry every {spacing:,} nodes)")
    print(f"basin contains no catalogue entry at all       {no_entry:5}/{m}")
    print(f"\nplanting one entry from that basin, giving up the farthest of the 32:")
    print(f"  answer recovered                             {recovered:5}/{planted}  "
          f"{100*recovered/max(planted,1):5.1f}%")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, pool=a.pool, P=a.P, cases=m,
            own_max=own_max, basin_p50=float(np.percentile(b, 50)),
            no_entry=no_entry, planted=planted, recovered=recovered), indent=2))


if __name__ == "__main__":
    main()
