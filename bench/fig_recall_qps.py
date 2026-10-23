"""Recall against throughput, one panel per corpus, drawn at the size it prints.

The convention here is the one the ANN benchmarks use -- recall on x, queries per
second on y, log -- which is the transpose of `fig_grid.py`. Both say the same
thing; this orientation reads better when the interesting part is a corpus where
everything is above 0.97 and the question is what the last point of recall
costs.

Every system gets its own frontier over its own knobs: for SWANN the stopping
rule, for IVF-RaBitQ the probe count crossed with bits per dimension. Nobody is
held to anybody else's index. A frontier here is the upper-left staircase --
among points at least as accurate, the fastest -- so a setting that is beaten on
both axes does not appear.
"""
import argparse, glob, json, sys
from pathlib import Path


def pts(spec, algo=None):
    """Points from every file matching the spec, which may be several globs
    separated by commas -- a system's knob settings do not always share one
    filename pattern."""
    out = []
    files = []
    for part in spec.split(","):
        part = part.strip()
        drop = False
        if part.startswith("!"):              # exclude what this pattern matches
            part, drop = part[1:], True
        hits = glob.glob(part)
        if drop:
            files = [f for f in files if f not in set(hits)]
        else:
            files += hits
    for f in sorted(set(files)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            name = r.get("algo")
            if algo:
                if algo == "__refine__":
                    # every refinement multiple belongs to one curve
                    if not (name or "").startswith("ivf_pq (w/ref"):
                        continue
                elif name != algo:
                    continue
            if r.get("recall") is None or r.get("qps") is None:
                continue
            out.append((float(r["recall"]), float(r["qps"])))
    return out


def frontier(p):
    """Keep a point only when nothing is both faster and at least as accurate."""
    out, best = [], -1.0
    for r, q in sorted(p, key=lambda x: -x[1]):
        if r > best:
            out.append((r, q)); best = r
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="title:swann_glob:swann_fp16_glob:rabitq_glob:cagra_glob")
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--k", type=int, default=10, help="only for the titles")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=2.3,
                    help="panel width in inches, as it will be placed")
    ap.add_argument("--xmin", type=float, default=None)
    ap.add_argument("--xlo", type=float, default=0.80,
                    help="left edge; the convention is to start where the "
                         "systems begin to differ rather than at zero")
    ap.add_argument("--xscale", default="linear", choices=["logit", "linear"],
                    help="logit spreads the last decimal places of recall, which "
                         "is where the systems differ; linear shows the whole "
                         "range including where the quantised baselines live")
    ap.add_argument("--marker-scale", type=float, default=0.34,
                    help="marker size against the house default")
    ap.add_argument("--line-scale", type=float, default=0.72)
    ap.add_argument("--palette", default="categorical",
                    help="'categorical' for the distinct-hue set these plots "
                         "normally use, or the name of a matplotlib colormap "
                         "to sample instead")
    ap.add_argument("--out", default="results/final/fig_recall_qps")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)

    # plasma, sampled rather than taken whole. Its top end is a pale yellow that
    # disappears against the light panel this style draws on, so the stops stay
    # inside 0.08-0.82; the systems are also distinguished by marker, so the
    # ramp only has to keep them apart, not encode an order.
    # Ten series are categories, not a quantity, and a sequential ramp cannot
    # separate ten of them. These are distinct hues at similar weight, in the
    # palette the ANN benchmark plots use, and each series also carries its own
    # marker so the figure survives being printed in grey.
    #
    # Our two arms are the two purples: they are one system measured twice, and
    # giving them neighbouring shades of one hue says that, where two unrelated
    # colours would have made them look like two systems.
    CAT = {
        "SWANN":                "#6A3D9A",   # deep purple
        "SWANN fp16":           "#A97FD9",   # light purple
        "CAGRA":                "#3B7DD8",   # blue
        "GGNN":                 "#4FA64F",   # green
        "HNSW (CPU)":           "#7F7F7F",   # grey
        "IVF-RaBitQ (LUT)":     "#E8B018",   # gold
        "IVF-RaBitQ (Bitwise)": "#D4682A",   # orange
        "IVF-PQ (w/ref)":       "#B4232B",   # dark red
        "IVF-PQ (wo/ref)":      "#EE8C84",   # salmon
        "IVF-FLAT":             "#17A2A2",   # teal
    }
    # The frame the ANN benchmarks draw: white panel, dashed grey grid, a box
    # on all four sides. The house style's filled grey panel is for sheets with
    # one or two curves; here a reader is tracing five lines through each other
    # and the lighter ground keeps them apart.
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "ytick.minor.size": 1.8,
    })

    panels = [p.split(":") for p in a.panel]
    rows = (len(panels) + a.cols - 1) // a.cols
    fig, axes = plt.subplots(rows, a.cols, squeeze=False,
                             figsize=(a.cols * a.width + 0.5,
                                      rows * H.panel_h(a.width) + 0.8))
    seen = {}
    for i, spec in enumerate(panels):
        title, srcs = spec[0], spec[1:]
        ax = axes[i // a.cols][i % a.cols]
        series = []
        # Order matters twice: it fixes the colour each system gets across every
        # panel, and it is the order the legend is written in.
        # The order here is the order of the panel specs and of the legend.
        want = (("SWANN", "o", None),
                ("SWANN fp16", "v", None),
                ("CAGRA", "s", None),
                ("IVF-RaBitQ (LUT)", "D", "IVF-RaBitQ (LUT)"),
                ("IVF-RaBitQ (Bitwise)", "d", "IVF-RaBitQ (Bitwise)"),
                ("IVF-PQ (w/ref)", "^", "__refine__"),
                ("IVF-PQ (wo/ref)", "P", "ivf_pq (wo/ref)"),
                ("IVF-FLAT", "X", "ivf_flat"),
                ("GGNN", "<", None),
                ("HNSW (CPU)", ">", "hnsw_cpu"))
        for (label, mk, algo), src in zip(want, srcs):
            colour = CAT[label]
            if not src:
                continue
            f = frontier(pts(src, algo=algo))
            if f:
                series.append((f, label, colour, mk))
        for f, label, colour, mk in series:
            # The house style sizes the markers it knows about; this figure
            # needs ten distinct ones, so anything outside that table falls
            # back to the size of a filled circle.
            base = H.MS.get(mk, H.MS.get("o", 5.0))
            try:
                mw = H.mew(mk)
            except Exception:
                mw = H.MEW_FILLED
            # Small markers on a dense frontier: the curve is the message and
            # the markers only have to say which curve it is.
            ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk, color=colour,
                    lw=H.LW * a.line_scale, ms=H.ms(base) * a.marker_scale,
                    mew=mw * a.marker_scale, label=label, zorder=3)
        ax.set_yscale("log")
        # Decades only on y. Matplotlib's log axis also labels 2x, 3x, 6x when a
        # panel spans less than one decade, which at this size collides with the
        # neighbouring panel; the minor ticks stay as unlabelled marks so the
        # scale is still readable between the decades.
        from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
        ax.yaxis.set_minor_formatter(NullFormatter())
        # Recall is read in its last two decimal places here -- every system is
        # above 0.9 and the question is what 0.99 against 0.999 costs -- and a
        # linear axis gives that range a few pixels. A logit axis spreads it,
        # which is the convention the ANN benchmarks use for the same reason.
        ax.set_xscale(a.xscale)
        ax.set_title(title, fontsize=8)
        # A floor per panel: a corpus whose systems all sit above 0.95 should not
        # spend most of its axis on recall nobody reaches.
        lo = min((p[0] for f, *_ in series for p in f), default=0.8)
        hi = max((p[0] for f, *_ in series for p in f), default=0.999)
        if a.xscale == "linear":
            x0 = a.xmin if a.xmin is not None else a.xlo
            x1 = 1.002
        else:
            # a logit axis carries neither 0 nor 1
            x0 = a.xmin if a.xmin is not None else max(0.5, lo - 0.01)
            x1 = min(0.9995, hi + (1 - hi) * 0.6)
        ax.set_xlim(x0, x1)
        # A logit axis labels itself as 1 - 10^-k and adds minor ticks, which at
        # this panel width overlap into illegibility. The ticks that mean
        # something to a reader are round recalls, so they are named directly.
        if a.xscale == "linear":
            step = 0.05 if (x1 - x0) <= 0.25 else 0.1
            t = x0
            keep = []
            while t <= 1.0001:
                keep.append(round(t, 3)); t += step
            ax.set_xticks(keep)
            ax.set_xticklabels([f"{t:.2f}" for t in keep])
        else:
            cand = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
            keep = [t for t in cand if x0 <= t <= x1]
            ax.set_xticks(keep)
            ax.set_xticklabels([("%g" % t).lstrip("0") for t in keep])
        ax.set_xticks([], minor=True)
        ax.tick_params(axis="x", labelsize=7)
        # The y range is set by what is visible, not by every point in the
        # file. Zooming x to the high-recall end leaves a baseline's fastest,
        # least accurate settings off the panel, and letting them keep setting
        # the top of the axis spends most of the height on throughput nothing
        # in view reaches.
        vis = [p[1] for f, *_ in series for p in f if x0 <= p[0] <= x1]
        if vis:
            ax.set_ylim(min(vis) * 0.82, max(vis) * 1.25)
        ax.grid(True, which="major", axis="both")
        for sp in ax.spines.values():
            sp.set_visible(True)
        if i % a.cols == 0:
            ax.set_ylabel("queries / s")
        if i // a.cols == rows - 1:
            ax.set_xlabel(f"Recall@{a.k}")
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            if ll not in seen:
                seen[ll] = hh
    for j in range(len(panels), rows * a.cols):
        axes[j // a.cols][j % a.cols].axis("off")
    if seen:
        order = [k for k in ("SWANN", "SWANN fp16", "CAGRA", "GGNN", "HNSW (CPU)",
                             "IVF-RaBitQ (LUT)", "IVF-RaBitQ (Bitwise)",
                             "IVF-PQ (w/ref)", "IVF-PQ (wo/ref)", "IVF-FLAT")
                 if k in seen]
        fig.legend([seen[k] for k in order], order, loc="upper center",
                   ncol=min(len(order), 5), frameon=True, fontsize=7,
                   bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png "
          f"({a.cols * a.width + 0.5:.1f} x "
          f"{rows * H.panel_h(a.width) + 0.8:.1f} in)")


if __name__ == "__main__":
    main()
