"""When is a query done? Measure first, then pick a rule.

The search cannot see fragments, so a stopping rule has to be built out of what
it can see: whether the answer changed, how far its frontier is, which seed
lineages are still producing. To choose between those, this replays a search one
expansion at a time and records, per step:

  hit       a ground-truth document entered the top-k at this step
  newfrag   that document was the first of its fragment to arrive -- the search
            reached somewhere it had not been
  changed   the top-k changed at all
  radius    the current k-th best distance
  frontier  the best unexpanded candidate's distance, per lineage

From that, any rule can be evaluated offline without rerunning the search, and
the oracle -- stop exactly at each query's last useful expansion -- bounds all of
them. The number that matters is not recall but recall per expansion kept.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--graph", default="G_lid.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--h", type=int, default=2)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--V", type=int, default=4)
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--nq", type=int, default=1024)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    frag = np.load(a.data / f"frag{a.h}_{a.split}_{tag}.npy")
    Q, gt, frag = Q[: a.nq], gt[: a.nq], frag[: a.nq]
    nq = Q.shape[0]

    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda(); Qg = torch.from_numpy(Q).cuda()
    Xs = Xg.half().contiguous() if a.half else Xg
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.1, 32, 24)
    seeds, _, _ = arms.seeds("distance", a.V)
    st = engine.init_states(Xs, Qg, seeds.unsqueeze(1), m=a.L, max_expansions=a.steps,
                            deg=G.shape[1], team_size=16, policy=engine.POLICY_ROUND_ROBIN)

    gt_t = torch.from_numpy(gt.astype(np.int64)).cuda()              # [q, k]
    frag_t = torch.from_numpy(frag.astype(np.int64)).cuda()
    nfrag = int(frag.max()) + 1
    found = torch.zeros((nq, a.k), dtype=torch.bool, device="cuda")  # gt slot in top-k
    fragseen = torch.zeros((nq, nfrag), dtype=torch.bool, device="cuda")

    hits, newfrags, changed, radii, frontier = [], [], [], [], []
    pq_hit, pq_changed = [], []            # per query, for simulating rules offline
    prev = torch.full((nq, a.k), -1, dtype=torch.int64, device="cuda")
    for t in range(a.steps):
        engine.advance(Xs, Gg, Qg, st, 1)
        top = st.ids[:, 0, : a.k].long()                             # queue is sorted
        hit = (top.unsqueeze(-1) == gt_t.unsqueeze(1)).any(dim=1)    # [q, k] gt slot present
        new = hit & ~found
        found |= hit
        # a fragment is "new" when the first of its members arrives
        fnew = torch.zeros((nq, nfrag), dtype=torch.bool, device="cuda")
        fnew.scatter_(1, frag_t.clamp(min=0), new)
        fresh = fnew & ~fragseen
        fragseen |= fnew
        ch = (top != prev).any(dim=1)
        hits.append(int(new.sum()))
        newfrags.append(int(fresh.sum()))
        changed.append(int(ch.sum()))
        pq_hit.append(new.sum(dim=1).to(torch.int8).clone())
        pq_changed.append(ch.clone())
        prev = top.clone()
        radii.append(st.dist[:, 0, a.k - 1].clone())
        unexp = (st.expanded[:, 0] == 0) & (st.ids[:, 0] >= 0)
        d = torch.where(unexp, st.dist[:, 0], torch.full_like(st.dist[:, 0], float("inf")))
        frontier.append(d.min(dim=1).values.clone())
        # per-lineage frontier: the best unexpanded candidate of each seed's descent
        lin = st.lineage[:, 0].long()
        pl = torch.full((nq, a.V), float("inf"), device="cuda")
        pl.scatter_reduce_(1, lin.clamp(max=a.V - 1), d, reduce="amin")
        frontier[-1] = pl

    hits = np.array(hits); newfrags = np.array(newfrags); changed = np.array(changed)
    cum_hit = np.cumsum(hits) / (nq * a.k)
    print(f"graph={a.graph} q={nq} L={a.L} V={a.V} steps={a.steps}\n")
    print(f"{'expansions':>10s} {'recall@10':>10s} {'new gt':>8s} {'new fragments':>14s} "
          f"{'queries whose topk moved':>25s}")
    for t in (8, 16, 24, 32, 48, 64, 96, 128, 192, a.steps - 1):
        if t < a.steps:
            print(f"{t+1:10d} {cum_hit[t]:10.4f} {hits[t]:8d} {newfrags[t]:14d} "
                  f"{changed[t]:25d}")

    tail = slice(96, a.steps)
    print(f"\nafter expansion 96: {hits[tail].sum()} new ground-truth documents, "
          f"{newfrags[tail].sum()} of them the first of a fragment "
          f"({100*newfrags[tail].sum()/max(hits[tail].sum(),1):.0f}%)")
    print(f"up to expansion 96: {hits[:96].sum()} new, {newfrags[:96].sum()} first-of-fragment "
          f"({100*newfrags[:96].sum()/max(hits[:96].sum(),1):.0f}%)")

    # ---- simulate stopping rules on the recorded trace ------------------------
    H = torch.stack(pq_hit).cpu().numpy().astype(np.int32)           # [T, Q]
    C = torch.stack(pq_changed).cpu().numpy()                        # [T, Q]
    R = torch.stack(radii).cpu().numpy()                             # [T, Q]
    F = torch.stack(frontier).cpu().numpy()                          # [T, Q, S]
    T = H.shape[0]
    total = nq * a.k

    def score(stop_at):
        """stop_at[q] = expansions used. Recall counts hits strictly before it."""
        got = np.array([H[: stop_at[q], q].sum() for q in range(nq)])
        return got.sum() / total, stop_at.mean()

    print(f"\n{'rule':34s} {'recall@10':>10s} {'mean expansions':>16s} {'vs exhaust':>11s}")
    base_stop = np.full(nq, T)
    br, bx = score(base_stop)
    print(f"{'exhaust the queue (current)':34s} {br:10.4f} {bx:16.1f} {'--':>11s}")

    # oracle: stop right after each query's last useful expansion
    last = np.array([np.max(np.flatnonzero(H[:, q])) + 1 if H[:, q].any() else 1
                     for q in range(nq)])
    orr, orx = score(last)
    print(f"{'oracle (last useful expansion)':34s} {orr:10.4f} {orx:16.1f} "
          f"{orx/bx:10.2f}x")

    rows = []
    for Tw in (4, 8, 16, 32, 64):
        stop = np.full(nq, T)
        for q in range(nq):
            run = 0
            for t in range(T):
                run = 0 if C[t, q] else run + 1
                if run >= Tw:
                    stop[q] = t + 1
                    break
        r, x = score(stop)
        print(f"{'top-k idle for ' + str(Tw) + ' expansions':34s} {r:10.4f} {x:16.1f} "
              f"{x/bx:10.2f}x")
        rows.append(dict(rule=f"idle{Tw}", recall=float(r), expansions=float(x)))

    # frontier rule: every lineage's best unexpanded candidate is worse than
    # radius * (1 + eps) -- nothing live can still beat the answer
    for eps in (0.0, 0.05, 0.1, 0.25, 0.5):
        live = F <= (R[:, :, None] * (1.0 + eps))
        done = ~live.any(axis=2)                                     # [T, Q]
        stop = np.full(nq, T)
        for q in range(nq):
            w = np.flatnonzero(done[:, q])
            if w.size:
                stop[q] = int(w[0]) + 1
        r, x = score(stop)
        print(f"{'all lineages beyond radius*' + str(1 + eps):34s} {r:10.4f} {x:16.1f} "
              f"{x/bx:10.2f}x")
        rows.append(dict(rule=f"radius{eps}", recall=float(r), expansions=float(x)))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.out.with_suffix(".npz"), hits=hits, newfrags=newfrags, changed=changed,
                 per_query_hit=H, per_query_changed=C, radius=R, frontier=F)
        a.out.write_text(json.dumps(dict(graph=a.graph, nq=int(nq), L=a.L, V=a.V,
                                         steps=a.steps, recall_curve=cum_hit.tolist(),
                                         rules=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
