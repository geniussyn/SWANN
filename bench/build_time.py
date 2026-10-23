"""Index build time for every method on one corpus, on whatever GPU is present.

Standalone on purpose: it takes a prepared corpus directory and writes one JSON,
so the measurement can be repeated on a different machine without the rest of
this repository's state. The device name goes in the output because build time
is the number in this paper that depends most on which GPU ran it, and a figure
drawn from several machines must be able to say so.

    python bench/build_time.py --data /dev/shm/swann/gist-960 \
        --corpus GIST --out results/buildtime/gist.json

Methods it can time, and what it needs for each:
    SWANN       offline/build_graph.py in this repository
    CAGRA, IVF-FLAT, IVF-PQ   the cuVS python package (PYTHONPATH=/opt/cuvsenv)
    IVF-RaBitQ  the torch extension in swann/csrc
Anything unavailable is skipped and recorded as such rather than omitted
silently, so a gap in the figure is a gap in the measurement and not a guess.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

import numpy as np


def device_name():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=30)
        return out.stdout.strip().splitlines()[0]
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--corpus", required=True, help="label for the figure")
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--degree", type=int, default=96)
    ap.add_argument("--rabitq-bits", type=int, default=8)
    ap.add_argument("--methods", nargs="+",
                    default=["SWANN", "CAGRA", "IVF-FLAT", "IVF-PQ",
                             "IVF-RaBitQ"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = np.load(a.data / "X.npy")
    n, d = X.shape
    rows = []

    def record(name, seconds, note=""):
        rows.append(dict(method=name, build_s=seconds, note=note))
        print(f"  {name:14} {seconds:8.2f}s  {note}")

    def skip(name, why):
        rows.append(dict(method=name, build_s=None, note=f"unavailable: {why}"))
        print(f"  {name:14} {'--':>8}   unavailable: {why}")

    print(f"{a.corpus}: n={n:,} d={d} on {device_name()}")

    if "SWANN" in a.methods:
        script = Path(__file__).resolve().parent.parent / "offline" / "build_graph.py"
        if script.exists():
            best = None
            for _ in range(a.repeats):
                t = time.perf_counter()
                r = subprocess.run([sys.executable, str(script), "--data", str(a.data),
                                    "--D", str(a.degree), "--cand", "128",
                                    "--alpha", "1.2", "--out", "G_buildtime.npy"],
                                   capture_output=True, text=True)
                el = time.perf_counter() - t
                if r.returncode == 0:
                    best = el if best is None else min(best, el)
            (a.data / "G_buildtime.npy").unlink(missing_ok=True)
            if best is None:
                skip("SWANN", "build_graph.py failed")
            else:
                record("SWANN", best, f"degree {a.degree}")
        else:
            skip("SWANN", "build_graph.py not found")

    cuvs_methods = [m for m in a.methods if m in ("CAGRA", "IVF-FLAT", "IVF-PQ")]
    if cuvs_methods:
        try:
            from pylibraft.common import device_ndarray
            from cuvs.neighbors import ivf_flat, ivf_pq, cagra
            metric = "inner_product" if ip else "sqeuclidean"
            Xd = device_ndarray(X)
            for m in cuvs_methods:
                best = None
                for _ in range(a.repeats):
                    t = time.perf_counter()
                    if m == "CAGRA":
                        idx = cagra.build(cagra.IndexParams(
                            metric=metric, graph_degree=man.get("deg", 64)), Xd)
                    elif m == "IVF-FLAT":
                        idx = ivf_flat.build(ivf_flat.IndexParams(
                            n_lists=a.lists, metric=metric), Xd)
                    else:
                        idx = ivf_pq.build(ivf_pq.IndexParams(
                            n_lists=a.lists, metric=metric, pq_bits=8), Xd)
                    el = time.perf_counter() - t
                    best = el if best is None else min(best, el)
                    del idx
                record(m, best, f"lists {a.lists}" if m != "CAGRA"
                       else f"degree {man.get('deg', 64)}")
        except Exception as e:
            for m in cuvs_methods:
                skip(m, f"{type(e).__name__}")

    if "IVF-RaBitQ" in a.methods:
        try:
            import torch
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            from cuvs_rabitq import ext
            m = ext()
            Xt = torch.from_numpy(X).cuda().contiguous()
            best = None
            for _ in range(a.repeats):
                t = time.perf_counter()
                idx = m.build(Xt, a.lists, a.rabitq_bits, 25, ip)
                el = time.perf_counter() - t
                best = el if best is None else min(best, el)
                del idx
                torch.cuda.empty_cache()
            record("IVF-RaBitQ", best, f"{a.rabitq_bits}-bit, lists {a.lists}")
        except Exception as e:
            skip("IVF-RaBitQ", f"{type(e).__name__}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(corpus=a.corpus, n=int(n), d=int(d),
                                     device=device_name(), lists=a.lists,
                                     rows=rows), indent=2))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
