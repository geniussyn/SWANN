"""Is a quantised baseline's ceiling its tuning range, or the method?

Two different claims get reported as one. A search that stops at recall 0.975
may be probing too few lists, or it may be returning the wrong candidates
however many it probes. The separation is mechanical:

  candidate recall   is the true top-k inside the returned candidate set at all?
                     At n_probes = n_lists nothing is skipped, so whatever is
                     missing here is the quantiser and not the search.
  reranked recall    rescoring those candidates against the real fp32 vectors,
                     how many survive into the top-k? This is the ceiling a
                     deployment could buy with refinement.

A gap between the two is fixable with rerank budget; a candidate recall below
one at full probe is not fixable at all.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cuvs_rabitq import ext                                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--bits", type=int, nargs="+", default=[3, 8])
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[64, 256, 1024, 2048, 4096])
    ap.add_argument("--cand", type=int, nargs="+", default=[10, 100, 1000],
                    help="candidate list sizes to return before reranking")
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--qchunk", type=int, default=0,
                    help="split the query batch; a full probe over a wide "
                         "candidate list silently drops the tail of a large "
                         "batch, so the ladder is re-checked in chunks")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    m = ext()
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda().contiguous()
    Q = torch.from_numpy(np.load(a.data / f"Q_{a.split}.npy")).cuda().contiguous()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[:, : a.k]
    gts = [set(int(t) for t in r) for r in gt]
    NQ = Q.shape[0]

    def rerank(cand, k):
        """Exact fp32 rescoring of each query's candidates, top-k out."""
        out = np.empty((NQ, k), np.int64)
        for s in range(0, NQ, 256):
            e = min(s + 256, NQ)
            c = torch.from_numpy(cand[s:e].astype(np.int64)).cuda()
            bad = c < 0
            c = c.clamp(min=0)
            V = X[c]                                   # [b, cand, d]
            q = Q[s:e].unsqueeze(2)                    # [b, d, 1]
            sc = torch.bmm(V, q).squeeze(2)            # inner product
            if not ip:
                sc = -((V - Q[s:e].unsqueeze(1)) ** 2).sum(2)
            sc = sc.masked_fill(bad, float("-inf"))
            top = sc.topk(min(k, sc.shape[1]), dim=1).indices
            out[s:e, : top.shape[1]] = torch.gather(c, 1, top).cpu().numpy()
        return out

    rows = []
    print(f"{man['corpus']} n={man['n']:,} d={X.shape[1]} lists={a.lists} "
          f"nq={NQ}  (probes={a.lists} is a full scan)\n")
    print(f"{'bits':>5}{'probes':>8}{'cand':>7}{'cand recall':>13}"
          f"{'reranked':>11}{'search ms':>11}{'rerank ms':>11}")
    for bits in a.bits:
        t = time.perf_counter()
        idx = m.build(X, a.lists, bits, 25, ip)
        torch.cuda.synchronize()
        bs = time.perf_counter() - t
        for pr in a.probes:
            for nc in a.cand:
                torch.cuda.synchronize(); s = time.perf_counter()
                if a.qchunk:
                    parts = []
                    for qs in range(0, NQ, a.qchunk):
                        nbp, _ = m.search(idx, Q[qs:qs + a.qchunk].contiguous(),
                                          nc, pr, a.mode)
                        parts.append(nbp.cpu().numpy())
                    cand = np.concatenate(parts, 0)
                    torch.cuda.synchronize(); ms = (time.perf_counter() - s) * 1e3
                else:
                    nb, _ = m.search(idx, Q, nc, pr, a.mode)
                    torch.cuda.synchronize(); ms = (time.perf_counter() - s) * 1e3
                    cand = nb.cpu().numpy()
                empty = int((cand < 0).all(1).sum())
                if empty:
                    print(f"      {empty} of {NQ} queries returned nothing")
                crec = float(np.mean([len(gts[q] & {int(v) for v in cand[q] if v >= 0}) / a.k
                                      for q in range(NQ)]))
                t2 = time.perf_counter()
                rr = rerank(cand, a.k)
                torch.cuda.synchronize()
                rms = (time.perf_counter() - t2) * 1e3
                rrec = float(np.mean([len(gts[q] & {int(v) for v in rr[q]}) / a.k
                                      for q in range(NQ)]))
                print(f"{bits:>5}{pr:>8}{nc:>7}{crec:>13.4f}{rrec:>11.4f}"
                      f"{ms:>11.1f}{rms:>11.1f}")
                rows.append(dict(system="IVF-RaBitQ", bits=bits, n_probes=pr,
                                 cand=nc, lists=a.lists, candidate_recall=crec,
                                 reranked_recall=rrec, search_ms=ms,
                                 rerank_ms=rms, full_scan=(pr >= a.lists),
                                 empty_queries=empty, qchunk=a.qchunk,
                                 build_s=bs, qps=NQ / (ms / 1e3)))
        del idx
        torch.cuda.empty_cache()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k,
                                         nq=int(NQ), lists=a.lists, rows=rows),
                                    indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
