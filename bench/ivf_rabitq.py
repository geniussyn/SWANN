"""IVF + RaBitQ on the GPU, end to end: probe, one-bit scan, exact rerank.

The knob is the candidate budget -- how many vectors the one-bit scan looks at,
filled from the nearest lists -- crossed with the rerank ratio, how many of the
scan's survivors get an exact fp32 comparison. That pair is what a quantised
pipeline actually trades, and reporting `n_probes` instead would hide the fact
that lists are not the same size.

The scan reads the packed codes directly (128 bytes per vector at d=1024,
against 4096 for fp32), which is the whole point: cheap enough per candidate
that looking at many is affordable.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch


def ext():
    from torch.utils.cpp_extension import load
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "12.0")
    src = Path(__file__).resolve().parent.parent / "gpuann" / "csrc" / "rabitq.cu"
    return load(name="gpuann_rabitq", sources=[str(src)],
                extra_cuda_cflags=["-O3"], extra_cflags=["-O3"], verbose=False)


def per_query_recall(got, gt):
    k = gt.shape[1]
    return np.array([len(set(g.tolist()) & set(t.tolist())) / k for g, t in zip(got, gt)])


@torch.no_grad()
def gather_candidates(lists, order, starts, counts, budget):
    """[Q, budget] of vector ids, filled from the nearest probed lists, -1 padded."""
    Q, P = lists.shape
    dev = lists.device
    cnt = counts[lists].to(torch.int64)                       # [Q, P]
    pos = torch.cumsum(cnt, dim=1) - cnt                      # start slot per list
    cand = torch.full((Q, budget), -1, dtype=torch.int64, device=dev)
    lid = torch.full((Q, budget), 0, dtype=torch.int64, device=dev)
    slot = torch.arange(budget, device=dev).view(1, -1)
    for p in range(P):
        take = (slot >= pos[:, p:p + 1]) & (slot < (pos[:, p:p + 1] + cnt[:, p:p + 1]))
        rank = (slot - pos[:, p:p + 1]).clamp(min=0)
        src = (starts[lists[:, p]].to(torch.int64).view(-1, 1) + rank).clamp(
            max=order.numel() - 1)
        cand = torch.where(take, order[src].to(torch.int64), cand)
        lid = torch.where(take, lists[:, p:p + 1].expand(-1, budget), lid)
    return cand, lid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--index", default="rabitq")
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--budgets", type=int, nargs="+", default=[1000, 2000, 5000, 10000, 20000])
    ap.add_argument("--ratios", type=int, nargs="+", default=[2, 5, 10])
    ap.add_argument("--probes", type=int, default=64)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ix = a.data / a.index
    im = json.loads((ix / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    Q = np.load(a.data / f"Q_{a.split}.npy")
    gt = np.load(a.data / f"gt_{a.split}_ids.npy")
    nq = Q.shape[0]

    Xg = torch.from_numpy(X).cuda()
    Qg = torch.from_numpy(Q).cuda()
    C = torch.from_numpy(np.load(ix / "centroids.npy")).cuda()
    R = torch.from_numpy(np.load(ix / "rotation.npy")).cuda()
    codes = torch.from_numpy(np.load(ix / "codes.npy")).cuda()
    rnorm = torch.from_numpy(np.load(ix / "rnorm.npy")).cuda()
    err = torch.from_numpy(np.load(ix / "err.npy")).cuda()
    order = torch.from_numpy(np.load(ix / "order.npy")).cuda().to(torch.int64)
    starts = torch.from_numpy(np.load(ix / "starts.npy")).cuda()
    counts = torch.from_numpy(np.load(ix / "counts.npy")).cuda()
    m = ext()

    print(f"{man['corpus']} n={man['n']} d={man['d']}  {a.split} q={nq} k={a.k}\n"
          f"IVF-RaBitQ: {im['n_lists']} lists, {im['bytes_per_vector']['code']}B code + "
          f"{im['bytes_per_vector']['scalars']}B scalars per vector "
          f"({im['bytes_per_vector']['fp32'] / (im['bytes_per_vector']['code'] + 8):.0f}x "
          f"smaller than fp32), build {im['build_seconds']}s\n")
    print(f"{'budget':>7s} {'rerank':>7s} {'recall@10':>10s} {'ms(batch)':>10s} {'QPS':>9s}")

    rows = []
    for budget in a.budgets:
        for ratio in a.ratios:
            kk = a.k * ratio
            best, ids = None, None
            for r in range(a.repeats + 1):
                torch.cuda.synchronize(); t = time.perf_counter()
                qrot = (Qg @ R.T).contiguous()
                cdot_all = Qg @ C.T
                lists = torch.topk(cdot_all, a.probes, dim=1).indices
                cand, lid = gather_candidates(lists, order, starts, counts, budget)
                cdot = torch.gather(cdot_all, 1, lid)
                est = m.rabitq_scan(codes, rnorm, err, qrot,
                                    cand.to(torch.int32).contiguous(), cdot.contiguous())
                top = torch.topk(est, min(kk, budget), dim=1).indices
                shortlist = torch.gather(cand, 1, top)                 # [Q, kk]
                exact = torch.einsum("qd,qcd->qc", Qg, Xg[shortlist])  # fp32 rerank
                fin = torch.gather(shortlist, 1, torch.topk(exact, a.k, dim=1).indices)
                torch.cuda.synchronize()
                ms = (time.perf_counter() - t) * 1e3
                if r:
                    best = ms if best is None else min(best, ms)
                ids = fin
            rec = float(per_query_recall(ids.cpu().numpy(), gt).mean())
            rows.append(dict(algo="ivf_rabitq", itopk=budget, width=ratio, recall=rec,
                             recall_hard=None, batch_ms=best, qps=nq / (best / 1e3)))
            print(f"{budget:7d} {ratio:7d} {rec:10.4f} {best:10.2f} {nq/(best/1e3):9.0f}")

    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, index=im, split=a.split, k=a.k,
                                         nq=int(nq), probes=a.probes, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
