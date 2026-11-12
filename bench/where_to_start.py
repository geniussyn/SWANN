"""For the queries that drag the mean down: which entry would have worked, and
what is it about?

Everything so far says the signal is not computable from the query, and the way
to stop guessing is to look at the cases. For each query with low recall, take an
answer it lost, find the catalogue entry nearest that answer, verify by running
the search again with that entry added, and print all three texts side by side
with the entry's rank in the query's own ordering. If the entry that was needed
turns out to be about something a reader would not have derived from the query,
that is the finding; if it is obviously on-topic and merely ranked 300th, that is
a different finding.
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
    ap.add_argument("--qtexts", type=Path, default=Path(f"{S}/t_texts.json"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, default=32)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--B", type=int, default=64)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--cases", type=int, default=12)
    ap.add_argument("--chars", type=int, default=108)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    metric = engine.METRIC_IP if ip else engine.METRIC_L2
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")
                         if False else np.load(a.data / "Q_test.npy")).cuda()
    gt = np.load(a.data / "gt_test_ids.npy")[:, : a.k]
    NQ = Q.shape[0]
    qtexts = json.loads(a.qtexts.read_text())
    assert len(qtexts) == NQ, f"{len(qtexts)} texts for {NQ} queries"
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)
    order = select_a.select_nearest(Q, Vp, a.P, metric=metric)         # full ordering
    near = order[:, : a.V]

    def search(seed_cols, qsel=None):
        o = order if qsel is None else order[qsel]
        q = Q if qsel is None else Q[qsel]
        ids, _, _ = engine.search(X, G, q, pt[torch.gather(o, 1, seed_cols)]
                                  .to(torch.int32), k=a.k, L=a.L, B=a.B,
                                  metric=metric, team_size=a.team,
                                  stop_eps=a.stop_eps, shared_cache=True,
                                  shared_bound=True, exclusive=True,
                                  elide_visited=True)
        return ids.cpu().numpy()

    got = search(torch.arange(a.V, device=X.device).view(1, -1).expand(NQ, -1).contiguous())
    hit = np.array([[int(t) in set(int(v) for v in g if v >= 0) for t in row]
                    for g, row in zip(gt, got)])
    pq = hit.mean(1)
    print(f"{a.data.name} {a.graph}  V={a.V} P={a.P:,} eps={a.stop_eps}  "
          f"recall {pq.mean():.4f}")
    print(f"the worst {a.cases} queries carry "
          f"{(1 - pq).sum():.0f} missing answers in total; "
          f"{int((pq < 0.5).sum())} queries are below 0.5\n")

    docs = {}
    want = set()
    worst = np.argsort(pq)[: a.cases]
    # entry nearest each missed answer, and its rank in the query's ordering
    rank_of = {}
    for q in worst:
        for i in range(a.k):
            if not hit[q, i]:
                want.add(int(gt[q, i]))
    with torch.no_grad():
        ids_ = torch.tensor(sorted(want), device=X.device)
        s = (X[ids_] @ Vp.T) if ip else -torch.cdist(X[ids_], Vp)
        best = s.argmax(1).cpu().numpy()
        bestsim = s.max(1).values.cpu().numpy()
    ent_of = {int(d): int(e) for d, e in zip(sorted(want), best)}
    sim_of = {int(d): float(v) for d, v in zip(sorted(want), bestsim)}
    inv = {}
    for q in worst:
        inv[int(q)] = {int(pool[int(order[q, r])]): r for r in range(0)}  # filled below
    ordnp = order.cpu().numpy()
    pos = {int(q): {int(v): r for r, v in enumerate(ordnp[q])} for q in worst}

    for d in want:
        docs[d] = None
    with open(a.docs) as f:
        for n, line in enumerate(f):
            if n in docs:
                docs[n] = line.strip()
            if n in want:
                pass
    # the entries' own texts too
    ent_ids = {int(pool[ent_of[d]]) for d in want}
    etext = {i: None for i in ent_ids}
    with open(a.docs) as f:
        for n, line in enumerate(f):
            if n in etext:
                etext[n] = line.strip()

    def clip(t):
        t = (t or "?").replace("\t", " ")
        return t[: a.chars] + ("..." if len(t) > a.chars else "")

    print("for each case: the query, an answer it lost, and the catalogue entry\n"
          "nearest that answer -- with that entry's rank in the query's own\n"
          "ordering, and whether adding it recovers the answer.\n")
    rows = []
    for q in worst:
        qi = int(q)
        missed = [i for i in range(a.k) if not hit[qi, i]]
        if not missed:
            continue
        i = missed[0]
        d = int(gt[qi, i])
        e_slot = ent_of[d]
        e_doc = int(pool[e_slot])
        r = pos[qi].get(e_slot, -1)
        cols = torch.cat([torch.arange(a.V, device=X.device),
                          torch.tensor([r], device=X.device)]).view(1, -1)
        out = search(cols, qsel=torch.tensor([qi], device=X.device))
        ok = d in {int(v) for v in out[0] if v >= 0}
        print(f"  recall {pq[qi]:.2f}  query: {clip(qtexts[qi])}")
        print(f"            lost  : [{d}] {clip(docs.get(d))}")
        print(f"            entry : [{e_doc}] rank {r} of {a.P:,}, "
              f"sim to answer {sim_of[d]:.3f}  -> recovers: {ok}")
        print(f"                    {clip(etext.get(e_doc))}\n")
        rows.append(dict(q=qi, recall=float(pq[qi]), query=qtexts[qi],
                         lost_doc=d, lost_text=docs.get(d), entry_doc=e_doc,
                         entry_text=etext.get(e_doc), entry_rank=int(r),
                         sim_entry_answer=sim_of[d], recovers=bool(ok)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
