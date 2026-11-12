"""The dense-matmul brute force on this device, as a floor under the batch regime.

select_a.select_nearest already scores a catalogue slice by slice and folds each
slice into a running best, which is exact. Handed the whole corpus as the
catalogue it is exact brute force, and it is the thing a graph search has to beat
at the batch sizes where it collects its throughput.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, sys
sys.path.insert(0, "/root/gpuann")
from gpuann import engine, select_a

ap = argparse.ArgumentParser()
ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
ap.add_argument("--nq", type=int, nargs="+", default=[250, 1000, 2000])
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--slice", type=int, default=65536)
ap.add_argument("--precisions", nargs="+", default=["tf32", "fp32", "fp16"])
ap.add_argument("--repeats", type=int, default=3)
a = ap.parse_args()

man = json.loads((a.data / "manifest.json").read_text())
metric = engine.METRIC_IP if man["metric"].startswith("ip") else engine.METRIC_L2
Xg = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
Qall = np.load(a.data / "Q_val.npy")
gt = np.load(a.data / "gt_val_ids.npy")
N, d = Xg.shape
print(f"{man['corpus']} n={N} d={d}  exact dense brute force, k={a.k}, "
      f"slice={a.slice}\n")
print(f"{'prec':>5s} {'Q':>5s} {'ms':>8s} {'ms/query':>9s} {'QPS':>9s} "
      f"{'TFLOP/s':>8s} {'recall@10':>10s}")
for prec in a.precisions:
    X = Xg.half().contiguous() if prec == "fp16" else Xg
    for nq in a.nq:
        Qg = torch.from_numpy(np.ascontiguousarray(Qall[:nq])).cuda()
        Qs = Qg.half() if prec == "fp16" else Qg
        best = None
        for r in range(a.repeats + 1):
            torch.cuda.synchronize(); t = time.perf_counter()
            idx = select_a.select_nearest(Qs, X, a.k, metric=metric,
                                          precision=("tf32" if prec == "tf32" else
                                                     ("fp16" if prec == "fp16" else "fp32")),
                                          slice_P=a.slice)
            torch.cuda.synchronize(); ms = (time.perf_counter() - t) * 1e3
            if r and (best is None or ms < best[0]):
                best = (ms, idx)
        ms, idx = best
        got = idx.cpu().numpy()
        rc = float(np.mean([len(set(g.tolist()) & set(t.tolist())) / a.k
                            for g, t in zip(got, gt[:nq])]))
        tfl = 2.0 * nq * N * d / (ms * 1e-3) / 1e12
        print(f"{prec:>5s} {nq:5d} {ms:8.1f} {ms/nq:9.4f} {1e3*nq/ms:9.0f} "
              f"{tfl:8.1f} {rc:10.4f}")
        del Qg, Qs
        torch.cuda.empty_cache()
