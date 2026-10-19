"""Turn the Qwen MRL encodes into corpora, one per dimension.

The documents and queries are identical across the three widths -- they are
prefixes of one encode -- so anything that differs between the resulting
corpora is dimension and nothing else. Ground truth is recomputed per width,
because the exact top-10 of a 1024-dim prefix is not the exact top-10 of the
4096-dim vector and scoring one against the other would measure truncation
rather than search.
"""
import argparse, json
from pathlib import Path

import numpy as np

from prep_data import exact_topk, knn_graph        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=Path("/workspace/gpuann-data/qwen"))
    ap.add_argument("--dims", type=int, nargs="+", default=[4096, 2560, 1024])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("/dev/shm/gpuann"))
    a = ap.parse_args()

    src_man = json.loads((a.src / "manifest.json").read_text())
    for d in a.dims:
        X = np.ascontiguousarray(np.load(a.src / f"d{d}" / "docs.npy"), dtype=np.float32)
        Qall = np.ascontiguousarray(np.load(a.src / f"d{d}" / "queries.npy"), dtype=np.float32)
        out = a.out / f"qwen-{d}"
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(a.seed)
        perm = rng.permutation(Qall.shape[0])
        half = Qall.shape[0] // 2
        val, test = np.sort(perm[:half]), np.sort(perm[half:])
        print(f"qwen-{d}: X {X.shape}, queries val/test {val.size}/{test.size}", flush=True)
        np.save(out / "X.npy", X)
        np.save(out / "Q_val.npy", Qall[val]); np.save(out / "Q_test.npy", Qall[test])
        for split, Q in (("val", Qall[val]), ("test", Qall[test])):
            gi, gv = exact_topk(np.ascontiguousarray(Q), X, a.k)
            np.save(out / f"gt_{split}_ids.npy", gi); np.save(out / f"gt_{split}_sim.npy", gv)
        G = knn_graph(X, a.deg)
        np.save(out / "G.npy", G); np.save(out / "G_knn.npy", G)
        pools = out / "pools"; pools.mkdir(exist_ok=True)
        for P, nm in ((4096, "random4k"), (16384, "random16k")):
            r = np.random.default_rng(0)
            np.save(pools / f"pool_{nm}.npy",
                    np.sort(r.choice(X.shape[0], P, replace=False)).astype(np.int32))
        (out / "manifest.json").write_text(json.dumps({
            "corpus": f"qwen-{d}", "n": int(X.shape[0]), "d": int(d), "deg": int(a.deg),
            "k": int(a.k), "seed": int(a.seed), "metric": "ip_on_l2_normalised",
            "graph": "exact_knn", "model": src_man["model"],
            "note": "MRL prefix of one Qwen3-Embedding-8B encode, renormalised",
        }, indent=2))
        print(f"  wrote {out}", flush=True)


if __name__ == "__main__":
    main()
