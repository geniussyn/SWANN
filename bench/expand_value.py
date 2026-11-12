"""Does the EXPANSION rule have the entry rule's problem? It is the same rule.

The entry side is closed (docs/SUBSPACE-ENTRIES.md 8-12). Its one fact: an
entry's similarity to the query predicts whether expanding from it reaches an
answer at AUC 0.5540 +- 0.0139 over fourteen cells, and only walking tells you
(0.9999 after 8 steps). ARCHITECTURE.md 4.5 drew the conclusion already -- "the
diversity has to come from the trajectories" -- and 4.7's "proximity is worth
almost nothing" was that same fact.

But the kernel picks what to expand by **the same criterion**: 4.3/3-step-1, the
nearest unexpanded node of the lineage whose turn it is. Nearest-first. If
proximity is uninformative about where a descent goes, there is no reason it is
informative about which queue node is worth expanding -- and the budget there is
not 8-32 entries, it is 2048 expansions.

Measured on the queues the search actually builds. For every node that sat in a
queue unexpanded, two labels:

  has-gt     a ground-truth answer is among its graph neighbours, so expanding it
             evaluates that answer and 4.7's argument then guarantees the answer
             survives to the output.
  has-new    the same, restricted to answers the search did NOT find, which is
             the label that corresponds to recall actually lost.

and three scores: the node's own similarity to the query (what the kernel uses),
its best neighbour's similarity (one step of lookahead), and the similarity after
an 8-step climb (the entry-side ceiling, for comparability).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def auc(score, label, valid):
    out = []
    for s, y, v in zip(score, label, valid):
        s, y = s[v], y[v]
        p, n = y.sum(), (~y).sum()
        if p == 0 or n == 0:
            continue
        r = np.argsort(np.argsort(s)) + 1
        out.append((r[y].sum() - p * (p + 1) / 2) / (p * n))
    return (float(np.mean(out)) if out else float("nan")), len(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="val")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=8)
    ap.add_argument("--L", type=int, default=64, help="wide queue, so there is a choice")
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")[: a.nq]).cuda()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    NQ = Q.shape[0]
    pl = torch.from_numpy(load_pool(a.data, a.pool, a.P).astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pl)
    sel = select_a.select_nearest(Q, Vp, a.V, engine.METRIC_IP)

    ids, _, st = engine.search(X, G, Q, pl[sel].to(torch.int32), k=a.k, L=a.L,
                               B=a.B, metric=engine.METRIC_IP, team_size=a.team,
                               stop_eps=a.stop_eps, shared_cache=True,
                               shared_bound=True, exclusive=True,
                               elide_visited=True)
    got = ids.cpu().numpy()
    found = [set(int(v) for v in got[q] if v >= 0) for q in range(NQ)]
    gts = [set(int(t) for t in r) for r in gt]
    recall = float(np.mean([len(gts[q] & found[q]) / a.k for q in range(NQ)]))

    # every queue slot the search left UNEXPANDED: the choices it declined
    qi = st.ids.reshape(NQ, -1)                     # [NQ, J*L]
    un = (st.expanded.reshape(NQ, -1) == 0) & (qi >= 0)
    print(f"{a.data.name} {a.split} {NQ} queries, V={a.V} L={a.L} B={a.B} "
          f"eps={a.stop_eps}: recall {recall:.4f}")
    print(f"  queue slots per query: {qi.shape[1]}, left unexpanded "
          f"{un.float().sum(1).mean():.1f} ({un.float().mean():.1%})\n")

    nodes = qi.clamp(min=0).to(torch.long)
    gtt = torch.from_numpy(gt.astype(np.int64)).cuda()
    newmask = torch.zeros(NQ, a.k, dtype=torch.bool, device=X.device)
    for q in range(NQ):
        newmask[q] = torch.tensor([int(t) not in found[q] for t in gt[q]],
                                  device=X.device)
    # the lookahead is [NQ, slots, D, d] if materialised -- 75 GiB here -- so the
    # slot axis is chunked and only the per-slot max is kept.
    S = nodes.shape[1]
    has_gt = torch.zeros(NQ, S, dtype=torch.bool, device=X.device)
    has_new = torch.zeros(NQ, S, dtype=torch.bool, device=X.device)
    s_look = torch.empty(NQ, S, device=X.device)
    step = max(1, (1 << 28) // (G.shape[1] * X.shape[1] * NQ))
    for s0 in range(0, S, step):
        e0 = min(s0 + step, S)
        nb = G[nodes[:, s0:e0].reshape(-1)].long().view(NQ, e0 - s0, G.shape[1])
        hit = (nb.unsqueeze(3) == gtt.view(NQ, 1, 1, a.k)).any(2)    # [NQ, c, k]
        has_gt[:, s0:e0] = hit.any(2)
        has_new[:, s0:e0] = (hit & newmask.unsqueeze(1)).any(2)
        v = X[nb.clamp(min=0).reshape(-1)].view(NQ, e0 - s0, G.shape[1], -1)
        sn = (v * Q.view(NQ, 1, 1, -1)).sum(3)
        sn = torch.where(nb >= 0, sn, torch.full((1,), -1e30, device=X.device))
        s_look[:, s0:e0] = sn.max(2).values
        del nb, hit, v, sn
    has_gt = has_gt.cpu().numpy()
    has_new = has_new.cpu().numpy()
    s_self = (X[nodes.reshape(-1)].view(NQ, -1, X.shape[1])
              * Q.unsqueeze(1)).sum(2)
    s_c8 = (X[climb(X, G, nodes, Q, steps=8).reshape(-1)]
            .view(NQ, -1, X.shape[1]) * Q.unsqueeze(1)).sum(2)
    uc = un.cpu().numpy()
    print(f"  {'label':34} {'base rate':>10} "
          f"{'AUC self (the kernel)':>23} {'AUC 1-step':>11} {'AUC climb8':>11}")
    rows = []
    for name, lab in (("neighbour is an answer", has_gt),
                      ("neighbour is a MISSED answer", has_new)):
        br = lab[uc].mean()
        v1, n1 = auc(s_self.cpu().numpy(), lab, uc)
        v2, _ = auc(s_look.cpu().numpy(), lab, uc)
        v3, _ = auc(s_c8.cpu().numpy(), lab, uc)
        print(f"  {name:34} {br:10.4f} {v1:23.4f} {v2:11.4f} {v3:11.4f}  "
              f"(n={n1})")
        rows.append(dict(label=name, base_rate=float(br), auc_self=v1,
                         auc_lookahead=v2, auc_climb8=v3, queries=n1))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, split=a.split, nq=NQ,
                                         V=a.V, L=a.L, B=a.B, eps=a.stop_eps,
                                         recall=recall, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
