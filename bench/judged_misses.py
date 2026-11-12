"""The answers that actually matter and that we still lose.

Recall against the human judgements runs higher than recall against the exact
top-k, so the passages we drop are mostly ones nobody called relevant. The
residue is the interesting part: queries whose judged passage the encoder did
rank in its own top-k, and that the search still fails to return. Those are the
only failures with a cost outside the metric, and there are few enough to read.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402

S = "/tmp/claude-0/-root/5bf4d04d-280c-4ff8-a5aa-31202385739e/scratchpad"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--docs", type=Path,
                    default=Path("/workspace/SYN/corpora/msmarco-1m_docs.txt"))
    ap.add_argument("--queries", type=Path,
                    default=Path("/workspace/SYN/corpora/msmarco-1m_queries.jsonl"))
    ap.add_argument("--map", type=Path, default=Path(f"{S}/test2syn.npy"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--chars", type=int, default=100)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / "Q_test.npy")).cuda()
    gt = np.load(a.data / "gt_test_ids.npy")[:, : a.k]
    NQ = Q.shape[0]
    recs = [json.loads(l) for l in open(a.queries)]
    idx = np.load(a.map)
    qtext = [recs[i]["text"] for i in idx]
    rel = [set(int(d) for d, v in (recs[i].get("rel") or {}).items() if int(v) > 0)
           for i in idx]
    gts = [set(int(t) for t in r) for r in gt]

    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    order = select_a.select_nearest(Q, Vp, a.P, metric=metric)
    ids, _, _ = engine.search(X, G, Q, pt[order[:, : a.V]].to(torch.int32), k=a.k,
                              L=a.L, B=a.B, metric=metric, team_size=a.team,
                              stop_eps=a.stop_eps, shared_cache=True,
                              shared_bound=True, exclusive=True, elide_visited=True)
    got = ids.cpu().numpy()
    sets = [{int(v) for v in got[q] if v >= 0} for q in range(NQ)]

    reachable = [q for q in range(NQ) if rel[q] & gts[q]]
    bad = [q for q in reachable if not (rel[q] & gts[q] & sets[q])]
    print(f"{a.data.name}  V={a.V} P={a.P:,} eps={a.stop_eps}\n"
          f"{len(reachable):,} queries whose judged passage the encoder ranked "
          f"in its own top-{a.k}\n"
          f"{len(bad)} of them the search does not return "
          f"({100*len(bad)/len(reachable):.2f}%)\n")

    # where the judged passage sits in the encoder's own ranking, and in ours
    pos_in_gt, ent_rank = [], []
    ordnp = order.cpu().numpy()
    want = set()
    for q in bad:
        d = sorted(rel[q] & gts[q])[0]
        want.add(d); want.add(int(gt[q, 0]))
        pos_in_gt.append(int(np.where(gt[q] == d)[0][0]))
    with torch.no_grad():
        wl = torch.tensor(sorted(want), device=X.device)
        s = (X[wl] @ Vp.T) if ip else -torch.cdist(X[wl], Vp)
        bslot = {int(w): int(e) for w, e in zip(sorted(want), s.argmax(1).cpu().numpy())}
        del s
    texts = {d: None for d in want} | {int(pool[v]): None for v in bslot.values()}
    with open(a.docs) as f:
        for n, line in enumerate(f):
            if n in texts:
                texts[n] = line.strip()

    def clip(t):
        t = (t or "?").replace("\t", " ")
        return t[: a.chars] + ("..." if len(t) > a.chars else "")

    print(f"the judged passage's rank inside the encoder's top-{a.k}: "
          f"mean {np.mean(pos_in_gt):.1f} (0 = the encoder's own first choice)\n")
    rows = []
    for q in bad[: a.show]:
        d = sorted(rel[q] & gts[q])[0]
        slot = bslot[d]
        r = int(np.where(ordnp[q] == slot)[0][0])
        cols = torch.cat([torch.arange(a.V, device=X.device),
                          torch.tensor([r], device=X.device)]).view(1, -1)
        out, _, _ = engine.search(X, G, Q[q].view(1, -1).contiguous(),
                                  pt[torch.gather(order[q].view(1, -1), 1, cols)]
                                  .to(torch.int32), k=a.k, L=a.L, B=a.B,
                                  metric=metric, team_size=a.team,
                                  stop_eps=a.stop_eps, shared_cache=True,
                                  shared_bound=True, exclusive=True, elide_visited=True)
        ok = d in {int(v) for v in out[0].cpu().numpy() if v >= 0}
        print(f"  query: {clip(qtext[q])}")
        print(f"  judged: [{d}] encoder rank {int(np.where(gt[q]==d)[0][0])}  "
              f"{clip(texts.get(d))}")
        print(f"  nearest entry to it: rank {r} of {a.P:,} -> adding it recovers: {ok}")
        print(f"          {clip(texts.get(int(pool[slot])))}\n")
        rows.append(dict(q=int(q), query=qtext[q], judged=int(d),
                         gt_rank=int(np.where(gt[q] == d)[0][0]), entry_rank=r,
                         recovers=bool(ok)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(reachable=len(reachable), missed=len(bad),
                                         rows=rows), indent=2))


if __name__ == "__main__":
    main()
