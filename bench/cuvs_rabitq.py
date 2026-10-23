"""cuVS's own GPU IVF-RaBitQ, as a baseline on our corpora and our ground truth.

RaBitQ belongs on the axes for the same reason PQ does -- it changes what a
candidate costs, not which candidates a traversal visits -- so it is a different
way of spending the same memory bandwidth our expansion is bound by. cuVS 26.10
ships the authors' extended multi-bit variant as C++ only: not in its C API and
not in its Python package, so comparing against our own one-bit scan and calling
it RaBitQ would be comparing against our reading of the paper. This links the
library.

The build is awkward only in its flags: headers and the shared library come from
an unpacked wheel tree rather than a system install, so both are passed
explicitly and the library directory goes on the runtime path as well.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

CUVS = Path(os.environ.get("CUVS_HOME", "/opt/cuvsenv"))


def ext():
    # The wheel tree keeps each project's headers under its own directory, and
    # cuvs pulls in raft, rmm, kvikio and the rapids logger transitively. Each
    # wheel also bundles the CCCL headers it was built against, under
    # include/rapids -- those are used rather than any system CCCL, because a
    # raft built against one version of <cuda/std/...> does not compile against
    # another.
    inc = [str(CUVS / d / "include") for d in
           ("libcuvs", "libraft", "librmm", "libkvikio", "rapids_logger")]
    inc += [str(CUVS / d / "include" / "rapids") for d in ("libraft", "librmm")]
    missing = [p for p in inc if not Path(p).exists()]
    if missing:
        raise SystemExit(f"cuVS headers not found: {missing}. Set CUVS_HOME.")
    # libcuvs pulls in raft, rmm and kvikio at load time, and each lives in its
    # own wheel directory, so every one of them goes on the runtime path -- not
    # just the directory that holds the library being linked.
    libs = [CUVS / d / "lib64" for d in ("libcuvs", "libraft", "libkvikio")]
    libs += [CUVS / d / "lib64" for d in ("rapids_logger", "librmm")]
    libs += [CUVS / d for d in ("libcuvs_cu12.libs", "libraft_cu12.libs",
                                "libkvikio_cu12.libs")]
    libs = [d for d in libs if d.exists()]
    # An rpath on our own module does not help libcuvs find its dependencies:
    # the linker emits DT_RUNPATH, which the loader does not inherit for
    # transitive loads, so libcuvs.so still fails on libkvikio.so. Opening the
    # dependencies first, globally and in order, is the reliable way.
    # In dependency order, leaves first: each of these is a separate wheel with
    # its own lib64, and the ones deeper in the chain are not on anybody's
    # runtime path.
    #
    # The CUDA maths libraries come first and come from this wheel tree, not
    # from torch's. cuvs 26.10 ships for CUDA 12.9 and this box builds with
    # 12.8: the wheel's libcusolver.so.11 wants cublasSetEnvironmentMode, which
    # torch's bundled libcublas.so.12 does not export, so whichever cublas is
    # resolved first decides whether the import works. Opening the 12.9 one
    # globally settles it, and 12.9 serves torch's 12.8 calls.
    import ctypes
    for dep in ("nvidia/nvjitlink/lib/libnvJitLink.so.12",
                "nvidia/cublas/lib/libcublasLt.so.12",
                "nvidia/cublas/lib/libcublas.so.12",
                "nvidia/cusparse/lib/libcusparse.so.12",
                "nvidia/curand/lib/libcurand.so.10",
                "nvidia/cusolver/lib/libcusolver.so.11",
                "rapids_logger/lib64/librapids_logger.so",
                "librmm/lib64/librmm.so",
                "libkvikio/lib64/libkvikio.so",
                "libraft/lib64/libraft.so"):
        f = CUVS / dep
        if f.exists():
            ctypes.CDLL(str(f), mode=ctypes.RTLD_GLOBAL)

    src = Path(__file__).resolve().parent.parent / "swann" / "csrc" / "cuvs_rabitq.cu"
    ld = [f"-L{libs[0]}", "-lcuvs"] + [f"-Wl,-rpath,{d}" for d in libs]
    return load(name="swann_cuvs_rabitq", sources=[str(src)],
                extra_include_paths=inc,
                extra_cflags=["-O3", "-std=c++20"],
                extra_cuda_cflags=["-O3", "-std=c++20",
                                   "--expt-relaxed-constexpr",
                                   "--expt-extended-lambda"],
                extra_ldflags=ld,
                verbose=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/swann/msmarco-1m"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--bits", type=int, default=3,
                    help="bits per dimension; cuVS defaults to 3")
    ap.add_argument("--kmeans-iters", type=int, default=20)
    ap.add_argument("--probes", type=int, nargs="+",
                    default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--mode", type=int, nargs="+", default=[0],
                    help="cuVS search_mode values to sweep. The library offers "
                         "two ways to score a probed list -- a lookup table and "
                         "a bitwise path -- and they trade accuracy against "
                         "speed differently, so a single mode is half the curve")
    ap.add_argument("--repeats", type=int, default=3)
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

    t = time.perf_counter()
    idx = m.build(X, a.lists, a.bits, a.kmeans_iters, ip)
    torch.cuda.synchronize()
    build_s = time.perf_counter() - t
    print(f"{man['corpus']} n={man['n']:,} d={X.shape[1]} metric={man['metric']}\n"
          f"cuVS ivf_rabitq: n_lists={a.lists} bits_per_dim={a.bits} "
          f"mode={a.mode} -> built in {build_s:.1f}s\n")
    print(f"{'mode':>5} {'n_probes':>9} {'recall@' + str(a.k):>10} {'ms':>9} "
          f"{'QPS':>10}")
    rows = []
    MODE = {0: "LUT", 1: "Bitwise"}
    for md in a.mode:
      for pr in a.probes:
        best = None
        for r in range(a.repeats + 1):
            torch.cuda.synchronize(); s = time.perf_counter()
            nb, _ = m.search(idx, Q, a.k, pr, md)
            torch.cuda.synchronize(); ms = (time.perf_counter() - s) * 1e3
            if r:
                best = ms if best is None else min(best, ms)
        got = nb.cpu().numpy()
        rec = np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0}) / a.k
                       for q in range(NQ)])
        name = MODE.get(md, str(md))
        print(f"{name:>5} {pr:9} {rec:10.4f} {best:9.2f} {NQ/(best/1e3):10,.0f}")
        rows.append(dict(algo=f"IVF-RaBitQ ({name})", n_probes=pr, bits=a.bits,
                         mode=md, lists=a.lists, recall=float(rec),
                         batch_ms=best, qps=NQ / (best / 1e3)))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(manifest=man, split=a.split, k=a.k,
                                         nq=int(NQ), build_seconds=round(build_s, 1),
                                         lists=a.lists, bits=a.bits, modes=a.mode,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
