"""Ground truth and piece structure for the 59,242 msmarco training queries.

The one-step head (docs/SUBSPACE-ENTRIES.md 13.1, and the plan agreed with the
user) needs (query, answer-pieces) pairs, and val/test hold only 3,490 each.
`/workspace/SYN/emb/msmarco-1m-trainpool__vanilla_queries.npy` holds **59,242**
queries from the same encoder (BAAI/bge-m3, cls, d=1024, revision 5617a9f6) over
the same corpus, and none of them appear in val or test (checked by hash over the
first 5,000).

The docs there are fp16 and `/dev/shm/gpuann/msmarco-1m/X.npy` is the fp32 cast
re-normalised -- per-row cosine >= 0.99993, max abs diff 1.7e-5 -- so the training
queries live in the search's space once cast and re-normalised the same way. X is
the authority, as everywhere else in this tree.

Ground truth is recomputed exactly here and never read from a file, which is the
lab's standing rule. The pieces are the attractors of the query's own greedy walk
(bench/basin_coverage.climb), because that is what an entry's basin means
everywhere else in this work.

Writes two small int32 arrays; the piece centroids are cheap to rebuild from them
and are not cached.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.basin_coverage import climb                               # noqa: E402


def exact_topk(X, Q, k, chunk=512):
    out = []
    for s in range(0, Q.shape[0], chunk):
        sc = Q[s:s + chunk] @ X.T
        out.append(torch.topk(sc, k, dim=1).indices.to(torch.int32))
        del sc
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--queries", type=Path,
                    default=Path("/workspace/SYN/emb/"
                                 "msmarco-1m-trainpool__vanilla_queries.npy"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--nq", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, default=Path("/dev/shm/gpuann/head"))
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    assert man["metric"].startswith("ip"), man["metric"]
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    Qn = np.load(a.queries)
    if a.nq:
        Qn = Qn[: a.nq]
    Q = torch.from_numpy(Qn.astype(np.float32)).cuda()
    Q = torch.nn.functional.normalize(Q, dim=1)
    print(f"{a.queries.name}: {Q.shape[0]:,} queries, d={Q.shape[1]}, "
          f"corpus {X.shape[0]:,}\n")

    t0 = time.perf_counter()
    gt = exact_topk(X, Q, a.k)
    torch.cuda.synchronize()
    print(f"exact top-{a.k} recomputed in {time.perf_counter()-t0:.1f}s")

    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    t0 = time.perf_counter()
    att = climb(X, G, gt.to(torch.int64), Q).to(torch.int32)
    torch.cuda.synchronize()
    print(f"pieces (greedy attractors of the answers) in "
          f"{time.perf_counter()-t0:.1f}s")

    an = att.cpu().numpy()
    npieces = np.array([len(np.unique(r)) for r in an])
    top1 = np.array([np.sort(np.unique(r, return_counts=True)[1])[-1] for r in an])
    print(f"\npieces per query: mean {npieces.mean():.2f}  median "
          f"{np.median(npieces):.0f}  max {npieces.max()}")
    print(f"largest piece's share of the {a.k} answers: mean "
          f"{(top1/a.k).mean():.1%}")
    print(f"single-piece queries: {(npieces==1).mean():.1%}   "
          f">=4 pieces: {(npieces>=4).mean():.1%}")
    print("  (val, 2000 queries, nest4k: 2.97 pieces, 71.5% largest, 25.8% "
          "single, 32.3% >=4 -- docs 8.1)")

    a.out.mkdir(parents=True, exist_ok=True)
    np.save(a.out / "gt_train.npy", gt.cpu().numpy())
    np.save(a.out / "att_train.npy", an)
    (a.out / "manifest.json").write_text(json.dumps(dict(
        queries=str(a.queries), nq=int(Q.shape[0]), k=a.k, graph=a.graph,
        corpus=str(a.data), encoder="BAAI/bge-m3 cls d1024",
        pieces_mean=float(npieces.mean()), top1_share=float((top1/a.k).mean()),
        single=float((npieces == 1).mean()), ge4=float((npieces >= 4).mean())),
        indent=2))
    print(f"\nwrote {a.out}/gt_train.npy, att_train.npy, manifest.json")


if __name__ == "__main__":
    main()
