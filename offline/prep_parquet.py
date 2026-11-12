"""Build a corpus from a HuggingFace parquet dataset, one file at a time.

The embedding sets this is for are tens of gigabytes as parquet and tens more as
float32, which does not fit beside everything else. So each shard is fetched,
its vector column copied into a memory-mapped output, and the shard deleted
before the next is fetched: the peak is one shard rather than the whole
download. Several columns can be taken in one pass, which is what makes the
Qdrant set cheap -- it carries the 1536 and the 3072 embedding of the same
documents, so both corpora come out of a single traversal.

Queries are taken from the end of the corpus and removed from it, because these
sets ship no query split. Exact fp32 ground truth is computed here, as
everywhere else in this repository.
"""
import argparse, json, time
from pathlib import Path

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--column", action="append", required=True,
                    help="name=dim=outdir, repeatable")
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--nq", type=int, default=5000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--cache", type=Path, default=Path("/dev/shm/hfcache"))
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    import pyarrow.parquet as pq

    cols = []
    for spec in a.column:
        name, dim, out = spec.split("=")
        cols.append((name, int(dim), Path(out)))
    api = HfApi()
    files = sorted(f.rfilename for f in api.dataset_info(a.repo).siblings
                   if f.rfilename.endswith(".parquet"))
    print(f"{a.repo}: {len(files)} shards, taking {a.n:,} rows")

    mm = {}
    for name, dim, out in cols:
        out.mkdir(parents=True, exist_ok=True)
        pad = (-dim) % 8
        mm[name] = (np.lib.format.open_memmap(out / "X.npy", mode="w+",
                                              dtype=np.float32,
                                              shape=(a.n, dim + pad)), dim, pad, out)
    done = 0
    t = time.perf_counter()
    for i, fn in enumerate(files):
        if done >= a.n:
            break
        p = Path(hf_hub_download(a.repo, fn, repo_type="dataset",
                                 cache_dir=str(a.cache)))
        tb = pq.read_table(p, columns=[c[0] for c in cols])
        m = min(tb.num_rows, a.n - done)
        for name, dim, pad, _ in ((n, d, pd, o) for n, (_, d, pd, o) in mm.items()):
            arr = np.asarray(tb[name].to_pylist()[:m], dtype=np.float32)
            assert arr.shape[1] == dim, f"{name}: {arr.shape[1]} != {dim}"
            mm[name][0][done:done + m, :dim] = arr
        done += m
        # Delete the blob, not the link into it: hf_hub_download returns a path
        # under snapshots/ that points at blobs/, and removing only the link
        # leaves the bytes behind, so the cache grows by the whole dataset even
        # though one shard at a time is all that is needed. Reaping orphans from
        # outside is not an alternative -- such a sweep eventually deletes the
        # .incomplete file the downloader is still writing, which is how this
        # conversion died once already.
        blob = p.resolve()
        p.unlink(missing_ok=True)
        if blob != p:
            blob.unlink(missing_ok=True)
        if i % 5 == 0 or done >= a.n:
            print(f"  {done:,}/{a.n:,} rows  ({time.perf_counter()-t:.0f}s)")
    print(f"read {done:,} rows in {time.perf_counter()-t:.0f}s")

    for name, (arr, dim, pad, out) in mm.items():
        arr.flush()
        X = np.load(out / "X.npy", mmap_mode="r")
        n = done - a.nq
        Xg = torch.from_numpy(np.ascontiguousarray(X[:n])).cuda()
        Qg = torch.from_numpy(np.ascontiguousarray(X[n:done])).cuda()
        Xg /= Xg.norm(dim=1, keepdim=True).clamp(min=1e-12)
        Qg /= Qg.norm(dim=1, keepdim=True).clamp(min=1e-12)
        np.save(out / "X.npy", Xg.cpu().numpy())
        np.save(out / "Q_test.npy", Qg.cpu().numpy())
        ids = np.empty((a.nq, a.k), np.int32); sim = np.empty((a.nq, a.k), np.float32)
        for s in range(0, a.nq, 256):
            e = min(s + 256, a.nq)
            v, idx = torch.topk(Qg[s:e] @ Xg.T, a.k, dim=1, largest=True)
            ids[s:e], sim[s:e] = idx.cpu().numpy(), v.cpu().numpy()
        np.save(out / "gt_test_ids.npy", ids)
        np.save(out / "gt_test_sim.npy", sim)
        (out / "manifest.json").write_text(json.dumps(dict(
            corpus=out.name, n=int(n), d=int(dim + pad), d_original=int(dim),
            k=a.k, metric="ip_on_l2_normalised", source=f"{a.repo}:{name}"), indent=2))
        print(f"  {out.name}: n={n:,} d={dim + pad} queries={a.nq}")
        del Xg, Qg
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
