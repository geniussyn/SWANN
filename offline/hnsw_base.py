"""Pull HNSW's base layer out of a saved hnswlib index.

The point is to run our search on somebody else's graph. hnswlib exposes no
adjacency accessor, but its saved file is a flat dump: a header of sizes, then
cur_element_count fixed-size blocks, each holding the node's level-0 link list,
its vector and its label. Level 0 is the one that matters -- it holds every
node, and HNSW's upper levels only pick the entry point, which we supply
ourselves from the pool.

Layout of one level-0 block (offsetLevel0_ is 0):
    [uint32 linkcount][uint32 neighbour x maxM0_][float32 vector x dim][uint64 label]
"""
import argparse, struct
from pathlib import Path

import numpy as np


def read_header(f):
    g = lambda: struct.unpack("<Q", f.read(8))[0]
    off_level0 = g(); max_elems = g(); cur_count = g(); size_per = g()
    label_off = g(); data_off = g()
    maxlevel = struct.unpack("<i", f.read(4))[0]
    enterpoint = struct.unpack("<I", f.read(4))[0]
    maxM = g(); maxM0 = g(); M = g()
    mult = struct.unpack("<d", f.read(8))[0]
    efc = g()
    return dict(off_level0=off_level0, max_elems=max_elems, cur=cur_count,
                size_per=size_per, label_off=label_off, data_off=data_off,
                maxlevel=maxlevel, enterpoint=enterpoint, maxM=maxM, maxM0=maxM0,
                M=M, mult=mult, efc=efc, header_bytes=f.tell())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="G_hnsw.npy to write")
    ap.add_argument("--chunk", type=int, default=100_000)
    a = ap.parse_args()

    with open(a.index, "rb") as f:
        h = read_header(f)
        n, mm0, sz = h["cur"], h["maxM0"], h["size_per"]
        print(f"hnswlib index: n={n:,} maxM0={mm0} M={h['M']} efc={h['efc']} "
              f"maxlevel={h['maxlevel']} block={sz}B")
        G = np.full((n, mm0), -1, np.int32)
        labels = np.empty(n, np.int64)
        base = h["header_bytes"]
        for s in range(0, n, a.chunk):
            e = min(s + a.chunk, n)
            f.seek(base + s * sz)
            buf = np.frombuffer(f.read((e - s) * sz), dtype=np.uint8).reshape(e - s, sz)
            cnt = buf[:, :4].copy().view(np.uint32).reshape(-1)
            nb = buf[:, 4:4 + 4 * mm0].copy().view(np.uint32).reshape(e - s, mm0)
            keep = np.arange(mm0)[None, :] < np.minimum(cnt, mm0)[:, None]
            G[s:e] = np.where(keep, nb.astype(np.int32), -1)
            labels[s:e] = buf[:, h["label_off"]:h["label_off"] + 8].copy().view(np.int64).reshape(-1)
            del buf
    deg = (G >= 0).sum(1)
    print(f"out-degree: min/med/max {deg.min()}/{int(np.median(deg))}/{deg.max()}  "
          f"edges {int(deg.sum()):,}")
    # hnswlib's internal ids are the insertion order; we added items with
    # labels 0..n-1 in order, so the mapping is the identity. Check it rather
    # than assume it -- a permuted graph would be silently wrong, not broken.
    if not np.array_equal(labels, np.arange(n)):
        print("labels are not identity: remapping")
        inv = np.empty(n, np.int32); inv[labels] = np.arange(n, dtype=np.int32)
        m = G >= 0
        G[m] = inv[G[m]]
        G = G[labels.argsort()]
    np.save(a.out, G)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
