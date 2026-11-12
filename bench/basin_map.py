"""The whole corpus, sorted into basins, for one query.

A greedy search does not travel between pieces; it climbs to whatever local
maximum its entry falls under and stops. So the answers it misses are not
unreachable -- they are often the local maximum of their own basin, which means
an entry landing anywhere in that basin delivers them where they stand, with no
travel at all. What decides whether that happens is the basin's size against the
catalogue's spacing: with P entries over N nodes, an entry lands roughly every
N/P nodes, and a basin smaller than that is one no catalogue of that size
reliably covers.

This computes it exactly rather than sampling. For a query, each node's single
best neighbour defines a function; following it to a fixed point assigns every
node to an attractor, and counting gives every basin's size.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine                                            # noqa: E402


def basins(X, G, q, ip=True, chunk=200_000):
    """attractor[u] for every node u, by pointer doubling over best-neighbour."""
    N = X.shape[0]
    s = torch.empty(N, device=X.device)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        s[a:b] = (X[a:b] @ q) if ip else -torch.cdist(X[a:b], q.view(1, -1)).view(-1)
    nxt = torch.empty(N, dtype=torch.int32, device=X.device)
    for a in range(0, N, chunk):
        b = min(a + chunk, N)
        nb = G[a:b].long()
        valid = nb >= 0
        sn = torch.where(valid, s[nb.clamp(min=0)], torch.full_like(s[nb.clamp(min=0)], -1e30))
        best = sn.argmax(1)
        bid = torch.gather(nb, 1, best.view(-1, 1)).view(-1)
        bsc = torch.gather(sn, 1, best.view(-1, 1)).view(-1)
        here = torch.arange(a, b, device=X.device)
        nxt[a:b] = torch.where(bsc > s[a:b], bid, here).to(torch.int32)
    # pointer doubling: every chain is monotone in s, so it terminates
    att = nxt.clone()
    for _ in range(40):
        nx = nxt[att.long()]
        if torch.equal(nx, att):
            break
        att = nx
    return att.long(), s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--qid", default="1030451")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--queries", type=Path,
                    default=Path("/workspace/SYN/corpora/msmarco-1m_queries.jsonl"))
    ap.add_argument("--qemb", type=Path,
                    default=Path("/workspace/SYN/emb/msmarco-1m__vanilla_queries.npy"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    N = X.shape[0]
    row = text = rel = None
    for i, l in enumerate(open(a.queries)):
        r = json.loads(l)
        if r["qid"] == a.qid:
            row, text, rel = i, r["text"], r.get("rel", {}); break
    q = torch.from_numpy(np.load(a.qemb, mmap_mode="r")[row].astype(np.float32)).cuda()
    q = q / q.norm()

    att, s = basins(X, G, q, ip)
    top = torch.topk(s, a.k).indices.cpu().numpy()
    uniq, cnt = torch.unique(att, return_counts=True)
    size = torch.zeros(N, dtype=torch.long, device=X.device)
    size[uniq] = cnt
    spacing = {4096: N // 4096, 16384: N // 16384, 65536: N // 65536, 131072: N // 131072}

    print(f'query {a.qid}: "{text}"   corpus {N:,}, graph {a.graph}')
    print(f"{int((att == torch.arange(N, device=X.device)).sum()):,} local maxima over the "
          f"whole corpus; a catalogue of 4,096 lands one entry every "
          f"{spacing[4096]:,} nodes\n")
    print(f"{'rank':>4} {'doc':>8} {'sim':>7} {'its attractor':>14} {'attr sim':>9} "
          f"{'basin':>9} {'covered by P=':>14}")
    rows = []
    for r, u in enumerate(top):
        at = int(att[int(u)])
        bs = int(size[at])
        need = [str(P) for P, sp in spacing.items() if bs >= sp]
        tick = "own max" if at == int(u) else f"{at}"
        print(f"{r:4} {int(u):8} {float(s[int(u)]):7.4f} {tick:>14} "
              f"{float(s[at]):9.4f} {bs:9,} "
              f"{(','.join(need) if need else 'none of them'):>14}")
        rows.append(dict(rank=r, doc=int(u), sim=float(s[int(u)]), attractor=at,
                         attractor_sim=float(s[at]), basin=bs,
                         is_own_max=at == int(u),
                         judged_relevant=str(int(u)) in (rel or {})))
    nat = len({r["attractor"] for r in rows})
    print(f"\nthe top-{a.k} falls into {nat} basins; "
          f"{sum(r['is_own_max'] for r in rows)} of the {a.k} are their own local maximum")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(qid=a.qid, text=text, N=N, rows=rows,
                                         n_basins=nat), indent=2))


if __name__ == "__main__":
    main()
