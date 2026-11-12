"""How few bits the stored vectors need, and what rescoring buys back.

The search is bound by the bytes it reads per candidate, so a narrower stored
vector is the largest lever left after fp16. The question is what it costs in
recall, and the answer is: almost nothing, once the final queue is rescored.

Quantise the corpus to B bits per dimension and dequantise it back, then run the
ordinary kernel. The query stays fp32, so this is asymmetric scoring -- the only
kind the lab's BQ work found usable -- and the only thing measured is the error
the stored side carries. A fixed random rotation is applied first, because that
is the part of RaBitQ that does the work: it spreads the energy evenly over the
coordinates so a per-vector scale fits all of them instead of being set by one
outlier. The rotation is free at serve time (one d x d multiply of the query,
0.03 ms for a batch of 2000 at d=1024) and it halves the error at every width.

Measured, MS MARCO 1M, V=32 L=16 B=16 eps=0.11, P=65536, 2000 val queries.
fp16 reaches 0.9947. "raw" is the quantised search's own answer; rrN rescores
the best N of its queue against the fp32 vectors:

    bits   B/vec     raw      rr16      rr64     rr512
       8    1024  0.9901    0.9952    0.9952    0.9952
       6     768  0.9754    0.9948    0.9948    0.9948
       5     640  0.9550    0.9946    0.9948    0.9948
       4     512  0.9168    0.9913    0.9949    0.9949
       3     384  0.8249    0.9489    0.9943    0.9946

So the quantiser's ceiling is a **ranking** problem and not a reaching one, all
the way down to 3 bits: the walk arrives at the right nodes and only orders them
wrong, and a rescore of 16 candidates fixes 5 bits and up, 64 fixes 3 and 4.
Without the rescore the same widths lose 0.005 (8 bits) to 0.17 (3 bits), which
is what makes the measurement worth having -- the raw column is what a reader
would assume the trade is.

Two things this does not measure. Speed: a dequantised vector moves as many
bytes as fp32, so no timing here means anything, and no int8 or int4 kernel
exists yet. And the width that is worth implementing is not the narrowest that
works -- 4 bits is a nibble, two per byte and 32 per 16-byte load, where 3 bits
needs a shift-and-mask chain for 25% fewer bytes, and 7 bits needs one for 12.5%
fewer than a byte-aligned int8.

Original question this was written for:

A quantiser's ceiling has two possible causes and they need different fixes. If
the walk reaches the right nodes and only orders them wrong, rescoring the final
queue against the real vectors restores everything and costs one pass over a few
hundred candidates. If the error steers the walk away from nodes it never
evaluates, no rescoring can help, because the answer was never in the queue.

So: run the search on the quantised corpus, then rescore every queue entry --
not just the reported top-k -- against fp32 and take the best ten. The gap that
survives is the reaching part.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
sys.path.insert(0, "/tmp/claude-0/-root/a2eabd91-bea7-4fb1-82ca-594f77dd2072/scratchpad")
from gpuann import engine, select_a
from quantbits import quantise, recall_at_k

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--graph", default="G_ptf32.npy")
ap.add_argument("--L", type=int, default=16)
ap.add_argument("--V", type=int, default=32)
ap.add_argument("--B", type=int, default=16)
ap.add_argument("--epss", type=float, nargs="+", default=[0.04, 0.11, 0.24])
ap.add_argument("--nq", type=int, default=2000)
ap.add_argument("--bits", type=int, nargs="+", default=[8])
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Q = np.ascontiguousarray(np.load(a.data / "Q_val.npy")[: a.nq])
gt = np.load(a.data / "gt_val_ids.npy")[: a.nq]
Qg = torch.from_numpy(Q).cuda()
Gg = torch.from_numpy(np.ascontiguousarray(np.load(a.data / a.graph))).cuda()
dg = engine.graph_degrees(Gg)
pool = np.load(a.data / "pools" / "pool_auto.npy").astype(np.int32)
pool_t = torch.from_numpy(pool).cuda()
catX = select_a.pool_vectors(Xg, pool_t)
near = select_a.select_nearest(Qg, catX, a.V, metric=metric, precision="tf32")
seeds = pool_t[near].unsqueeze(-1).contiguous()
N, d = Xg.shape
torch.manual_seed(0)
rot = torch.linalg.qr(torch.randn(d, d, device=Xg.device))[0]
Xh = Xg.half().contiguous()


@torch.no_grad()
def rerank(st, k=10, chunk=256, depth=None):
    """Best k of every id in the queue, scored against the fp32 corpus."""
    ids = st.ids.reshape(st.ids.shape[0], -1)                 # [Q, J*L]
    out = torch.full((ids.shape[0], k), -1, dtype=torch.int32, device=ids.device)
    for s in range(0, ids.shape[0], chunk):
        e = min(s + chunk, ids.shape[0])
        I = ids[s:e].long()
        ok = I >= 0
        V = Xg[I.clamp(min=0)]                                 # [b, m, d]
        dd = 1.0 - (V * Qg[s:e].unsqueeze(1)).sum(-1) if metric == engine.METRIC_IP \
            else ((V - Qg[s:e].unsqueeze(1)) ** 2).sum(-1)
        dd = torch.where(ok, dd, torch.full_like(dd, float("inf")))
        # one id can sit in several branches; dedup by keeping its first slot
        srt, order = torch.sort(I, dim=1)
        dup = torch.zeros_like(srt, dtype=torch.bool)
        dup[:, 1:] = srt[:, 1:] == srt[:, :-1]
        bad = torch.zeros_like(dup)
        bad.scatter_(1, order, dup)
        dd = torch.where(bad, torch.full_like(dd, float("inf")), dd)
        if depth is not None and depth < dd.shape[1]:
            # keep only the `depth` best by the *quantised* score, which is what
            # a real implementation has before it reads any fp32 vector
            qd = st.dist.reshape(st.dist.shape[0], -1)[s:e]
            qd = torch.where(ok & ~bad, qd, torch.full_like(qd, float("inf")))
            keep = torch.topk(qd, depth, dim=1, largest=False).indices
            mask = torch.ones_like(dd, dtype=torch.bool)
            mask.scatter_(1, keep, False)
            dd = torch.where(mask, torch.full_like(dd, float("inf")), dd)
        sel = torch.topk(dd, k, dim=1, largest=False).indices
        out[s:e] = torch.gather(I, 1, sel).to(torch.int32)
        del V, dd
    return out


print(f"{man['corpus']} d={d} P={len(pool):,} V={a.V} L={a.L} B={a.B} q={len(Q)}  "
      f"queue holds {a.V * a.L} ids per query\n")
DEPTHS = [16, 64, 512]
print(f"{'eps':>5s} {'bits':>5s} {'B/vec':>6s} {'fp16':>8s} {'raw':>8s} | " +
      "  ".join(f"{'rr'+str(x):>8s}" for x in DEPTHS))
for eps in a.epss:
  for bits in a.bits:
    X8 = quantise(Xg, bits, rot)
    r = {}
    for tag, X in (("fp16", Xh), ("q", X8)):
        ids, dist, st = engine.search(X, Gg, Qg, seeds, k=10, L=a.L, B=a.B,
                                      metric=metric, policy=engine.POLICY_BEST,
                                      stop_eps=eps, deg=dg, shared_cache=True,
                                      exclusive=True, shared_bound=True)
        r[tag] = recall_at_k(ids.cpu().numpy(), gt)
        if tag == "q":
            for x in DEPTHS:
                r[x] = recall_at_k(rerank(st, depth=x).cpu().numpy(), gt)
        del ids, dist, st
    print(f"{eps:5.2f} {bits:5d} {d*bits/8:6.0f} {r['fp16']:8.4f} {r['q']:8.4f} | " +
          "  ".join(f"{r[x]:8.4f}" for x in DEPTHS))
    del X8; torch.cuda.empty_cache()
