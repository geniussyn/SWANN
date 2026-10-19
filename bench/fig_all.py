"""Every system on one panel: ours, CAGRA, GGNN, IVF-PQ on the GPU, HNSW on CPU.

Each is drawn on its own recall-QPS frontier over whatever it exposes -- for us
the graph degree crossed with fan-out and retirement slack, for CAGRA itopk
crossed with its own graph degree, for GGNN k_build, for IVF-PQ probes crossed
with the refinement ratio, for hnswlib efSearch. Nobody is held to anybody
else's index size, which is the only way "its best against our best" means
anything.

The CPU line is a reference, not a like-for-like: one GPU against 128 cores. It
is on the axes to show what the GPU is buying at the recall that is wanted.
"""
import argparse, glob, json, sys
from pathlib import Path


def pts(spec):
    """glob of result files -> [(qps, recall)]. Handles both bench formats."""
    out = []
    for f in sorted(glob.glob(spec)):
        j = json.loads(Path(f).read_text())
        for r in j["rows"]:
            if r.get("recall") is None:
                continue
            out.append((r["qps"], r["recall"]))
    return out


def frontier(p):
    out, best = [], -1.0
    for q, r in sorted(p, key=lambda x: -x[0]):
        if r > best:
            out.append((q, r)); best = r
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="label:glob:colour:marker  (colour is a housestyle name)")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=5.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--ymin", type=float, default=None)
    ap.add_argument("--out", default="results/final/fig_all")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(13)

    fig, ax = plt.subplots(figsize=(a.width, H.panel_h(a.width) + 0.3))
    for spec in a.series:
        label, g, colour, mk = spec.split(":")
        f = frontier(pts(g))
        if not f:
            print(f"  (no points for {label})")
            continue
        ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk,
                color=getattr(H, colour), lw=H.LW, ms=H.ms(H.MS[mk]),
                mew=H.mew(mk), label=label, zorder=3)
    ax.set_xscale("log")
    ax.set_xlabel("QPS (whole query batch)")
    ax.set_ylabel("recall@10")
    if a.ymin is not None:
        ax.set_ylim(bottom=a.ymin)
    if a.title:
        ax.set_title(a.title, fontsize=11)
    ticks = [t for t in (1_000, 3_000, 10_000, 30_000, 100_000, 300_000)
             if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks); ax.set_xticks([], minor=True)
    ax.set_xticklabels([f"{t//1000}k" if t >= 1000 else str(t) for t in ticks])
    ax.legend(loc="lower left", frameon=True, facecolor=H.PANEL, fontsize=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); print(f"wrote {p}")


if __name__ == "__main__":
    main()
