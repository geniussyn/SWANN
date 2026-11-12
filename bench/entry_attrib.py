"""Which entries actually collect fragments, and what do they look like?

Entry *selection* was measured and found not to matter: five placements land on
the same recall, and predicting "will this entry bring a new piece" scores AUC
0.606. That is a statement about prediction before the search. This asks the
other question -- after the search, the branches that recovered a piece nobody
else recovered, what distinguishes them? If a property shows up here it is a
candidate for a selector; if nothing does, the count really is all there is.

Credit is per branch, which is exact in branch mode: one entry per CTA, so a
branch's own best-k is attributable to its own entry.
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
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/glove-100"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--split", default="test")
    ap.add_argument("--h", type=int, default=1)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=128)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--arm", default="distance")
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--deciles", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
    tag = Path(a.graph).stem
    X = np.load(a.data / "X.npy"); G = np.load(a.data / a.graph)
    Q = np.load(a.data / f"Q_{a.split}.npy"); gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    frag = np.load(a.data / f"frag{a.h}_{a.split}_{tag}.npy")
    if a.nq:
        Q, gt, frag = Q[: a.nq], gt[: a.nq], frag[: a.nq]
    Xg = torch.from_numpy(X).cuda(); Gg = torch.from_numpy(G).cuda()
    Qg = torch.from_numpy(Q).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    arms = Arms(Xg, Qg, pool, a.data, a.pool, 128, 0.05, 32, 24)
    seeds, _, _ = arms.seeds(a.arm, a.V)                 # [Q, V] on cpu
    seeds_g = seeds.to(torch.int32).cuda()

    ids, _, st = engine.search(Xg, Gg, Qg, seeds_g, k=a.k, L=a.L, B=a.B,
                               metric=metric, team_size=a.team,
                               stop_eps=a.stop_eps, shared_cache=True,
                               shared_bound=True, exclusive=True)
    st.check()
    bi, _ = engine.branch_topk(st, a.k)                  # [Q, J, k]
    bi = bi.cpu().numpy()
    nq, J, _ = bi.shape
    print(f"{a.data.name} {a.graph} V={a.V} J={J} q={nq}  mean phi_{a.h}="
          f"{(frag.max(1)+1).mean():.3f}")

    # ---- per-entry features ----
    sd = seeds.numpy().astype(np.int64)                  # [Q, V]
    with torch.no_grad():
        E = Xg[torch.from_numpy(sd).cuda().reshape(-1)].view(nq, a.V, -1)
        q3 = Qg[:nq].unsqueeze(1)
        if metric == engine.METRIC_IP:
            dq = (1.0 - (q3 * E).sum(-1))
        else:
            dq = ((q3 - E) ** 2).sum(-1)
        dq = dq.cpu().numpy()                            # [Q, V] query->entry
        # entry isolation: distance to its nearest other entry of the same query
        if metric == engine.METRIC_IP:
            ee = 1.0 - torch.bmm(E, E.transpose(1, 2))
        else:
            ee = ((E.unsqueeze(2) - E.unsqueeze(1)) ** 2).sum(-1)
        ee += torch.eye(a.V, device=ee.device).unsqueeze(0) * 1e9
        iso = ee.min(dim=2).values.cpu().numpy()         # [Q, V]
        del E, ee
    deg = (G >= 0).sum(1)[sd]                            # [Q, V] entry out-degree
    rank = np.argsort(np.argsort(dq, axis=1), axis=1)    # 0 = nearest entry

    # ---- credit ----
    hit = np.zeros((nq, J), dtype=bool)        # branch recovered >=1 gt doc
    uniq = np.zeros((nq, J), dtype=bool)       # ... a fragment nobody else got
    nfrag_hit = np.zeros((nq, J), dtype=np.int32)
    for q in range(nq):
        lab = frag[q]; g = gt[q]
        slot_of = {int(g[s]): s for s in range(a.k)}
        per = [set() for _ in range(J)]
        for j in range(J):
            for v in bi[q, j]:
                s = slot_of.get(int(v))
                if s is not None:
                    per[j].add(int(lab[s]))
        cnt = {}
        for j in range(J):
            for f in per[j]:
                cnt[f] = cnt.get(f, 0) + 1
        for j in range(J):
            nfrag_hit[q, j] = len(per[j])
            hit[q, j] = len(per[j]) > 0
            uniq[q, j] = any(cnt[f] == 1 for f in per[j])

    print(f"\nbranches that recovered anything : {hit.mean():.4f}")
    print(f"branches with a unique fragment  : {uniq.mean():.4f}")
    print(f"fragments per contributing branch: {nfrag_hit[hit].mean():.3f}")

    rows = []
    hitf, uniqf, nff = hit.reshape(-1), uniq.reshape(-1), nfrag_hit.reshape(-1)

    def bucketise(name, feat, lo_is_first=True):
        qs = np.quantile(feat, np.linspace(0, 1, a.deciles + 1))
        print(f"\n-- {name} --")
        print(f"{'bucket':>14} {'n':>8} {'hit':>8} {'unique':>8} {'frags':>7}")
        for i in range(a.deciles):
            sel = (feat >= qs[i]) & (feat <= qs[i + 1] if i == a.deciles - 1 else feat < qs[i + 1])
            if not sel.any():
                continue
            h, u = hitf[sel].mean(), uniqf[sel].mean()
            fr = nff[sel][hitf[sel]].mean() if hitf[sel].any() else 0.0
            print(f"{qs[i]:7.3f}-{qs[i+1]:6.3f} {int(sel.sum()):8d} {h:8.4f} {u:8.4f} {fr:7.3f}")
            rows.append(dict(feature=name, lo=float(qs[i]), hi=float(qs[i + 1]),
                             n=int(sel.sum()), hit=float(h), unique=float(u),
                             frags=float(fr)))
    bucketise("distance query -> entry", dq.reshape(-1))
    bucketise("rank of entry by distance", rank.reshape(-1).astype(float))
    bucketise("entry isolation (nearest other entry)", iso.reshape(-1))
    bucketise("entry out-degree", deg.reshape(-1).astype(float))

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(
            data=a.data.name, graph=a.graph, V=a.V, h=a.h, nq=nq,
            hit=float(hit.mean()), unique=float(uniq.mean()),
            frags_per_contrib=float(nfrag_hit[hit].mean()), rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
