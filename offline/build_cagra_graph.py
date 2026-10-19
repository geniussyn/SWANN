"""Build the common search graph with cuVS CAGRA and write it as the lab's G.

Every design in the document compares entry selection *on the same graph*, so
the graph should be the one a real system would serve: CAGRA's own, not the
exact-kNN stand-in prep_data.py writes. Measured on 100K MS MARCO, the stand-in
costs about 8 recall points at equal search time, so keeping it would have made
every later comparison a comparison against a handicapped baseline.

Runs in the cuVS process (PYTHONPATH=/opt/cuvsenv), writes plain .npy, and
leaves the old graph beside it as G_knn.npy. Nothing else in the corpus changes.

    PYTHONPATH=/opt/cuvsenv python offline/build_cagra_graph.py
"""
import argparse, json, shutil, time
from pathlib import Path

import numpy as np
from cuvs.neighbors import cagra

try:
    from cuvs.common import Resources
except ImportError:
    from pylibraft.common import DeviceResources as Resources
from pylibraft.common import device_ndarray


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m-100k"))
    ap.add_argument("--deg", type=int, default=64)
    ap.add_argument("--intermediate", type=int, default=128)
    ap.add_argument("--metric", default="inner_product")
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    X = np.load(a.data / "X.npy")
    res = Resources()
    Xd = device_ndarray(X)

    t = time.perf_counter()
    idx = cagra.build(cagra.IndexParams(metric=a.metric, graph_degree=a.deg,
                                        intermediate_graph_degree=a.intermediate),
                      Xd, resources=res)
    res.sync()
    build_s = time.perf_counter() - t
    G = np.ascontiguousarray(idx.graph.copy_to_host().astype(np.int32))
    assert G.shape == (X.shape[0], a.deg), G.shape
    assert G.min() >= 0 and G.max() < X.shape[0], "graph holds an out-of-range id"

    old = a.data / "G.npy"
    if old.exists() and man.get("graph") == "exact_knn" and not (a.data / "G_knn.npy").exists():
        shutil.copy(old, a.data / "G_knn.npy")
    np.save(old, G)

    man.update(graph="cagra_build", deg=int(a.deg), graph_build_seconds=round(build_s, 2),
               graph_params=dict(graph_degree=a.deg, intermediate_graph_degree=a.intermediate,
                                 metric=a.metric, cuvs="26.10"))
    man.pop("graph_note", None)
    (a.data / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"built CAGRA graph {G.shape} in {build_s:.2f}s -> {old}  (kNN kept as G_knn.npy)")


if __name__ == "__main__":
    main()
