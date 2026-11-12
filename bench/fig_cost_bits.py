"""What the stored width buys on the axes throughput hides: bytes and memory.

A recall/QPS plot answers "how fast" and says nothing about "how much", and for
a quantiser those are different questions. Two bits buys no time over four on
this device -- the unpacking and the per-candidate bookkeeping are what is left
once the vector is this narrow -- and it still halves the traffic and the
resident index. Whether that matters is a deployment question, and it cannot be
read off the frontier.

Three panels, all at matched recall:

  time        batch ms, the frontier axis
  traffic     bytes a query must touch. Counted for SWANN -- the kernel tallies
              its own distance evaluations and each reads one whole stored
              vector, plus the adjacency it walked and the rescore it did.
              Derived from the structure for IVF-RaBitQ, which has no counter:
              probes * (n / lists) * d * bits / 8.
  footprint   what has to be resident to serve, which is where the rescore shows
              up: our narrow arm needs the fp16 corpus beside the packed one, and
              IVF-RaBitQ by design needs no raw vectors at all.

CAGRA is absent from the traffic panel on purpose: it chooses its iteration
count internally and it cannot be counted from outside, so a number there would
be a guess.

    python bench/fig_cost_bits.py --data results/degree/all_arms_work.json
"""
import argparse, glob, json, sys
from pathlib import Path


def frontier(pts):
    out, best = [], float("inf")
    for p in sorted(set(pts), key=lambda x: (-x[0], x[1])):
        if p[1] < best:
            out.append(p); best = p[1]
    return sorted(out)


def at(fr, rc):
    if not fr or rc < fr[0][0] or rc > fr[-1][0]:
        return None
    for (r0, v0), (r1, v1) in zip(fr, fr[1:]):
        if r0 <= rc <= r1:
            return v0 if r1 == r0 else v0 + (v1 - v0) * (rc - r0) / (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("results/degree/all_arms_work.json"))
    ap.add_argument("--rabitq", default="results/degree/rabitq_[0-9]bit*.json")
    ap.add_argument("--cagra", default="results/degree/cagra_1m_fp32*.json")
    ap.add_argument("--lists", type=int, default=4096)
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--width", type=float, default=2.3)
    ap.add_argument("--out", default="/root/swann-paper/figures/experiments/cost_bits")
    a = ap.parse_args()

    j = json.loads(a.data.read_text())
    n, d = j["manifest"]["n"], j["manifest"]["d"]
    # The *resident* graph, not the traversed one. Rows are front-compacted but
    # stored dense and padded with -1, so a pruned graph whose mean out-degree is
    # 51.6 still occupies its full 64-wide int32 row -- 0.238 GiB, the same as
    # CAGRA's fixed-degree 64. Charging the mean here instead understated our own
    # footprint by 24%; what a row costs to keep is the row.
    gwidth = 64

    # --- SWANN arms, counted ------------------------------------------------
    LAB = {4096: "SWANN fp32", 2048: "SWANN fp16",
           512: "SWANN 4-bit", 256: "SWANN 2-bit"}
    arms, foot = {}, {}
    for b, lab in LAB.items():
        rs = [r for r in j["swann_rows"] if r["bytes_per_vec"] == b]
        if not rs:
            continue
        # One frontier, not two. Taking a time frontier and a bytes frontier
        # independently lets the same recall be read off two *different*
        # configurations -- at 0.988 the 2-bit bytes frontier sat on
        # V=48 L=20 B=20 while its time frontier sat on V=16 L=16 B=16 -- and
        # the figure then draws them as one operating point. The point is
        # chosen by time, which is the axis an operator tunes, and the bytes
        # reported are the ones that configuration actually moved.
        front = frontier([(r["recall"], r["ms"]) for r in rs])
        pick = {}
        for rc, ms in front:
            for r in rs:
                if r["recall"] == rc and r["ms"] == ms:
                    pick[rc] = r["bytes_q"]; break
        arms[lab] = dict(ms=front, by=sorted(pick.items()))
        # resident: the stored corpus, the graph, and -- for a narrow arm -- the
        # fp16 copy the rescore reads
        g = n * gwidth * 4
        foot[lab] = n * b + g + (n * d * 2 if b <= 512 else 0)

    # --- IVF-RaBitQ, derived -------------------------------------------------
    rb = []
    for f in sorted(glob.glob(a.rabitq)):
        for r in json.loads(Path(f).read_text())["rows"]:
            B = r["bits"]
            by = r["n_probes"] * (n / a.lists) * d * B / 8
            rb.append((r["recall"], r["batch_ms"], by, B))
    if rb:
        arms["IVF-RaBitQ"] = dict(ms=frontier([(x[0], x[1]) for x in rb]),
                                  by=frontier([(x[0], x[2]) for x in rb]))
        # the width its own frontier uses at the top, which is what has to be stored
        top = max(rb, key=lambda x: x[0])
        foot["IVF-RaBitQ"] = n * d * top[3] / 8 + a.lists * d * 4

    cg = []
    for f in sorted(glob.glob(a.cagra)):
        cg += [(r["recall"], r["batch_ms"]) for r in json.loads(Path(f).read_text())["rows"]]
    if cg:
        arms["CAGRA"] = dict(ms=frontier(cg), by=[])
        foot["CAGRA"] = n * d * 4 + n * 64 * 4

    ORDER = ["SWANN 2-bit", "SWANN 4-bit", "SWANN fp16", "SWANN fp32",
             "CAGRA", "IVF-RaBitQ"]
    COL = {"SWANN 2-bit": "#B48DDD", "SWANN 4-bit": "#6A3D9A",
           "SWANN fp16": "#3F2566", "SWANN fp32": "#241338",
           "CAGRA": "#3B7DD8", "IVF-RaBitQ": "#E8A33D"}
    names = [x for x in ORDER if x in arms]

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    import numpy as np
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.85", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })
    fig, ax = plt.subplots(1, 3, figsize=(3 * a.width + 0.6,
                                          H.panel_h(a.width) + 0.7))
    x = np.arange(len(names))
    w = 0.8 / len(a.targets)
    HATCH = ["", "//"]
    for ti, t in enumerate(a.targets):
        for col, key, panel, lo in ((0, "ms", ax[0], 1.0), (1, "by", ax[1], 1e6)):
            vals = [at(arms[nm].get(key, []), t) for nm in names]
            pos = x - 0.4 + w * (ti + 0.5)
            panel.bar(pos, [0 if v is None else v / lo for v in vals], w * 0.9,
                      color=[COL[nm] for nm in names], hatch=HATCH[ti % 2],
                      edgecolor="white", linewidth=0.6,
                      label=f"recall {t:g}" if col == 0 else None)
            for p, v, nm in zip(pos, vals, names):
                if v is None:
                    panel.text(p, panel.get_ylim()[1] * 0.02, "n/a", ha="center",
                               fontsize=6, rotation=90, color="0.4")
    ax[2].bar(x, [foot[nm] / 2**30 for nm in names], 0.62,
              color=[COL[nm] for nm in names], edgecolor="white", linewidth=0.6)
    for i, nm in enumerate(names):
        ax[2].text(i, foot[nm] / 2**30, f"{foot[nm]/2**30:.2f}",
                   ha="center", va="bottom", fontsize=6.5, color="0.2")
    for k, (panel, ttl, yl) in enumerate(
            ((ax[0], "time", "batch ms"),
             (ax[1], "traffic", "MB a query"),
             (ax[2], "resident index", "GiB"))):
        panel.set_xticks(x)
        panel.set_xticklabels(names, rotation=38, ha="right", fontsize=6.8)
        panel.set_ylabel(yl, fontsize=8)
        panel.set_title(ttl, fontsize=8.5)
        panel.grid(True, axis="y")
        panel.set_axisbelow(True)
        if k < 2:
            panel.set_yscale("log")
    ax[0].legend(frameon=False, fontsize=6.8, loc="upper left")
    ax[1].text(0.5, 0.95, "CAGRA: not countable", transform=ax[1].transAxes,
               ha="center", va="top", fontsize=6.3, color="0.45")
    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=240, bbox_inches="tight")
    print(f"wrote {a.out}.pdf / .png\n")
    print(f"{'arm':14s} {'footprint':>10s}  " +
          "  ".join(f"{'ms@'+str(t):>9s} {'MB@'+str(t):>9s}" for t in a.targets))
    for nm in names:
        cells = []
        for t in a.targets:
            m, b = at(arms[nm].get("ms", []), t), at(arms[nm].get("by", []), t)
            cells.append(f"{'--':>9s}" if m is None else f"{m:9.2f}")
            cells.append(f"{'--':>9s}" if b is None else f"{b/1e6:9.2f}")
        print(f"{nm:14s} {foot[nm]/2**30:8.2f} GiB  " + "  ".join(cells))


if __name__ == "__main__":
    main()
