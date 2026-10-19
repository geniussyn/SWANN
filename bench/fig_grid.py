"""One sheet, one panel per corpus, every system on each.

Panels are given as `name:layout_key`, and the files for each system are found
by convention under results/, so adding a corpus means adding a panel rather
than editing the plotting. The legend sits above the grid, once: at this many
panels a per-axes legend either covers a curve or is repeated seven times.

Each system is drawn on its own frontier over its own knobs -- for SWANN the
graph degree crossed with V and eps, for CAGRA itopk crossed with its graph
degree, for GGNN k_build, for IVF-RaBitQ the candidate budget crossed with the
rerank ratio. Nobody is held to anybody else's index.
"""
import argparse, glob, json, sys
from pathlib import Path


def pts(spec, algo=None):
    out = []
    for f in sorted(glob.glob(spec)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            if algo and r.get("algo") != algo:
                continue
            if r.get("recall") is None or r.get("qps") is None:
                continue
            out.append((r["recall"], r["qps"]))
    return out


def ggnn_pts(spec):
    """One index per k_build; its best point."""
    out = []
    for f in sorted(glob.glob(spec)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        rows = j.get("rows", [])
        if rows:
            b = max(rows, key=lambda r: (round(r["recall"], 4), r["qps"]))
            out.append((b["recall"], b["qps"]))
    return out


def frontier(p):
    out, best = [], -1.0
    for r, q in sorted(p, key=lambda x: -x[1]):
        if r > best:
            out.append((r, q)); best = r
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="title:swann_glob:cagra_glob:ggnn_glob:rabitq_glob "
                         "(empty fields are skipped)")
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=3.1)
    ap.add_argument("--ymin", type=float, default=0.90)
    ap.add_argument("--out", default="results/final/fig_grid")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(12)

    panels = [p.split(":") for p in a.panel]
    rows = (len(panels) + a.cols - 1) // a.cols
    fig, axes = plt.subplots(rows, a.cols, squeeze=False,
                             figsize=(a.cols * a.width + 0.4,
                                      rows * H.panel_h(a.width) + 0.9))
    handles, labels = None, None
    for i, spec in enumerate(panels):
        title = spec[0]
        srcs = spec[1:]
        ax = axes[i // a.cols][i % a.cols]
        series = []
        if len(srcs) > 0 and srcs[0]:
            series.append((frontier(pts(srcs[0])), "SWANN", H.BLUE, "o"))
        if len(srcs) > 1 and srcs[1]:
            series.append((frontier(pts(srcs[1], algo="single_cta")), "CAGRA", H.RED, "s"))
        if len(srcs) > 2 and srcs[2]:
            series.append((frontier(ggnn_pts(srcs[2])), "GGNN", H.GREEN, "^"))
        if len(srcs) > 3 and srcs[3]:
            series.append((frontier(pts(srcs[3])), "IVF-RaBitQ", H.ORANGE, "D"))
        for f, label, colour, mk in series:
            if not f:
                continue
            ax.plot([p[1] for p in f], [p[0] for p in f], marker=mk, color=colour,
                    lw=H.LW, ms=H.ms(H.MS[mk]), mew=H.mew(mk), label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_title(title, fontsize=10)
        # Per-panel floor: a corpus where everything sits above 0.97 should not
        # spend three quarters of its axis on empty space, and one that runs down
        # to 0.9 should not be clipped.
        lo = min((p[0] for f, *_ in series for p in f), default=a.ymin)
        ax.set_ylim(bottom=max(a.ymin, min(0.99, lo - 0.004)), top=1.002)
        if i % a.cols == 0:
            ax.set_ylabel("recall@10")
        if i // a.cols == rows - 1 or i >= len(panels) - a.cols:
            ax.set_xlabel("QPS")
        ticks = [t for t in (3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000)
                 if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
        ax.set_xticks(ticks); ax.set_xticks([], minor=True)
        ax.set_xticklabels([f"{t//1000}k" if t < 1e6 else f"{t/1e6:g}M" for t in ticks])
        if handles is None and series:
            handles, labels = ax.get_legend_handles_labels()
    for j in range(len(panels), rows * a.cols):
        axes[j // a.cols][j % a.cols].axis("off")
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(labels),
                   frameon=True, facecolor=H.PANEL, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); print(f"wrote {p}")


if __name__ == "__main__":
    main()
