"""IVF-RaBitQ with the rescore SWANN's narrow arm gets, so the two are like for like.

Our 4-bit line retrieves on a quantised corpus and then rescores the best of its
queue against the full-precision vectors. Comparing that against a RaBitQ line
with no rescore flatters us, and the amount it flatters us is not small: taken to
all 4096 lists RaBitQ scans everything, so what stops it at recall 0.80 at one
bit is the code's ranking error and nothing else -- the same error a rescore
removes for us.

So this gives RaBitQ the same stage: ask it for `depth` neighbours, rescore them
against the fp32 corpus, keep the best k. Both the search and the rescore are
timed, as they are for SWANN.

    PYTHONPATH=/opt/cuvsenv python bench/rabitq_rerank.py --bits 1 --depth 64
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.cuvs_rabitq import ext                                   # noqa: E402


@torch.no_grad()
def rescore(ids, Xf, Qg, k, ip, chunk=256):
    out = torch.full((ids.shape[0], k), -1, dtype=torch.int64, device=ids.device)
    for s in range(0, ids.shape[0], chunk):
        e = min(s + chunk, ids.shape[0])
        I = ids[s:e].long()
        ok = I >= 0
        V = Xf[I.clamp(min=0)]
        q = Qg[s:e].unsqueeze(1)
        d = (1.0 - (V * q).sum(-1)) if ip else ((V - q) ** 2).sum(-1)
        d = torch.where(ok, d, torch.full_like(d, float("inf")))
        sel = torch.topk(d, min(k, d.shape[1]), dim=1, largest=False).indices
        got = torch.gather(I, 1, sel)
        out[s:e, : got.shape[1]] = torch.where(
            torch.gather(d, 1, sel).isinf(), torch.full_like(got, -1), got)
        del V, d
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--split", default="val")
    ap.add_argument("--nq", type=int, default=2000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--bits", type=int, default=1)
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[32, 128, 512, 1024, 2048, 4096])
    ap.add_argument("--depths", type=int, nargs="+", default=[10, 64, 256])
    ap.add_argument("--mode", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    m = ext()
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda().contiguous()
    Q = torch.from_numpy(np.ascontiguousarray(
        np.load(a.data / f"Q_{a.split}.npy")[: a.nq])).cuda().contiguous()
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")[: a.nq, : a.k]
    gts = [set(int(t) for t in r) for r in gt]
    t = time.perf_counter()
    idx = m.build(X, a.lists, a.bits, a.kmeans_iters, ip)
    torch.cuda.synchronize()
    print(f"{man['corpus']} n={man['n']:,} d={X.shape[1]}  IVF-RaBitQ "
          f"{a.bits}-bit, {a.lists} lists, built in {time.perf_counter()-t:.1f}s")
    print(f"{a.split} q={Q.shape[0]}  rescore against the fp32 corpus\n")
    print(f"{'probes':>7} {'depth':>6} {'recall@' + str(a.k):>10} {'search ms':>10} "
          f"{'rescore ms':>11} {'total ms':>9}")
    rows = []
    for pr in a.probes:
        for dep in a.depths:
            best = None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); s = time.perf_counter()
                nb, _ = m.search(idx, Q, dep, pr, a.mode)
                torch.cuda.synchronize(); s_ms = (time.perf_counter() - s) * 1e3
                s2 = time.perf_counter()
                got = rescore(nb, X, Q, a.k, ip) if dep > a.k else nb
                torch.cuda.synchronize(); r_ms = (time.perf_counter() - s2) * 1e3
                if r and (best is None or s_ms + r_ms < best[0] + best[1]):
                    best = (s_ms, r_ms, got)
            s_ms, r_ms, got = best
            g = got.cpu().numpy()
            rec = float(np.mean([len(gts[q] & {int(v) for v in g[q][: a.k] if v >= 0})
                                 / a.k for q in range(Q.shape[0])]))
            rows.append(dict(bits=a.bits, n_probes=pr, depth=dep, recall=rec,
                             search_ms=s_ms, rescore_ms=r_ms,
                             batch_ms=s_ms + r_ms,
                             qps=Q.shape[0] / ((s_ms + r_ms) / 1e3)))
            print(f"{pr:7} {dep:6} {rec:10.4f} {s_ms:10.2f} {r_ms:11.2f} "
                  f"{s_ms + r_ms:9.2f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split,
                                         nq=int(Q.shape[0]), k=a.k, bits=a.bits,
                                         lists=a.lists, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
