"""The main recall/throughput figure with the narrow stored formats.

Three families, each swept over its own knob and drawn as its own frontier:
SWANN at the packed widths, native cuVS CAGRA, and cuVS IVF-RaBitQ at two bit
widths. The orientation is the ANN benchmark one -- recall on x, QPS on y, log --
because above 0.97 the question is what the last point of recall costs.

The 1-bit RaBitQ line is short on purpose: it is drawn to its own ceiling, which
is where it stops regardless of how many lists are probed.

    python bench/fig_main_narrow.py --data results/degree/all_arms_msmarco.json
"""
import argparse, json, sys
from pathlib import Path


def frontier(pts):
    out, best = [], 0.0
    for rc, q in sorted(set(pts), key=lambda p: (-p[0], -p[1])):
        if q > best:
            out.append((rc, q)); best = q
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, nargs="+",
                    default=[Path("results/degree/all_arms_msmarco.json")])
    ap.add_argument("--titles", nargs="*", default=None)
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--xmin", type=float, default=0.60,
                    help="left edge of the full-range panel")
    ap.add_argument("--zoom", type=float, default=0.975,
                    help="left edge of the second panel, where the decision is")
    ap.add_argument("--width", type=float, default=3.1)
    ap.add_argument("--out", default="/root/swann-paper/figures/experiments/main_narrow",
                    help="written as .pdf for LaTeX and .png for looking at; "
                         "the default lands in the paper's experiments folder")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })
    # SWANN's widths are shades of one hue -- one system measured twice. The
    # baselines are the hues the other figures already give them.
    # Every quantised arm is drawn with the rescore, because the comparison is
    # otherwise one-sided: our narrow arm retrieves on a packed corpus and then
    # rescores against the full-precision one, and a RaBitQ line without that
    # stage is a different experiment. It is not a small difference -- taken to
    # all 4096 lists RaBitQ scans everything, so its 1-bit recall of 0.80 is the
    # code's ranking error and nothing else, and 64 rescored candidates take it
    # to 0.998. The un-rescored lines are kept as the dotted pair so the size of
    # that stage is visible rather than hidden.
    # IVF-RaBitQ is drawn as its own GPU paper reports it, with no rescore:
    # the selling point there is "avoiding accessing the raw vectors for
    # reranking", and bit width is what replaces it -- "for B=5 and B=7,
    # IVF-RaBitQ is sufficient to produce >0.95 and 0.99 recall" (Shi et al.,
    # VLDB 2026). Its frontier is therefore swept over bits *and* probes, which
    # is what makes it one line rather than one per width.
    #
    # Giving it our rescore instead takes it to recall 1.0 (bench/rabitq_rerank.py
    # measures that) but reports a configuration its authors argue against, so
    # that belongs in the text as an ablation and not on this axis.
    STYLE = [
        ("SWANN 4-bit",  "#6A3D9A", "^", "-"),
        ("SWANN 2-bit",  "#B48DDD", "D", "-"),
        ("CAGRA",        "#3B7DD8", "o", "-"),
        ("IVF-RaBitQ",   "#E8A33D", "s", "-"),
    ]

    # One corpus gets two panels: the whole range, where a baseline's ceiling is
    # visible, and the zoom where the choice is actually made. A single panel
    # cannot do both -- at the zoom an arm that stops early is off-scale, and a
    # legend entry with no curve reads as a missing measurement.
    single = len(a.data) == 1
    cols = [(p, lo) for p in a.data
            for lo in ((a.xmin, a.zoom) if single else (a.xmin,))]
    n = len(cols)
    fig, axes = plt.subplots(1, n, squeeze=False,
                             figsize=(n * a.width + 0.4, H.panel_h(a.width) + 0.6))
    seen = {}
    for pi, (path, xlo) in enumerate(cols):
        j = json.loads(Path(path).read_text())
        ax = axes[0][pi]
        nq = j["nq"]
        series = {}
        for r in j["swann_rows"]:
            b = r["bytes_per_vec"]
            key = {512: "SWANN 4-bit", 256: "SWANN 2-bit"}.get(b)
            if key:
                series.setdefault(key, []).append((r["recall"], 1e3 * nq / r["ms"]))
        for name, front in j["frontiers"].items():
            key = ("CAGRA" if "CAGRA" in name else
                   name if name.startswith("IVF-RaBitQ") else None)
            if key:
                series[key] = [(rc, 1e3 * nq / ms) for rc, ms in front]
        off = []
        for name, col, mk, ls in STYLE:
            if name not in series:
                continue
            f = frontier(series[name])
            vis = [p for p in f if p[0] >= xlo]
            if vis:
                # 45-75 points a line: markers every few points keep the series
                # identifiable without drawing a solid band of glyphs
                every = max(1, len(vis) // 9)
                h, = ax.plot([p[0] for p in vis], [p[1] for p in vis], color=col,
                             marker=mk, ms=3.4, lw=1.5, ls=ls,
                             markevery=every, markeredgewidth=0)
                seen.setdefault(name, h)
            else:
                # the arm never reaches this panel's range. Say so in the panel
                # rather than leaving a legend entry with no curve.
                off.append((name, f[-1][0], col))
        ax.set_yscale("log")
        ax.set_xlim(xlo, 1.0)
        ax.grid(True, which="both")
        ax.set_xlabel("recall@10")
        if pi == 0:
            ax.set_ylabel("queries / second")
        for i, (name, ceil, col) in enumerate(off):
            ax.text(0.02, 0.06 + 0.07 * i, f"{name}: ceiling {ceil:.3f}",
                    transform=ax.transAxes, fontsize=6.5, color=col, va="bottom")
        base = (a.titles[pi] if a.titles and pi < len(a.titles)
                else f"{j['manifest']['corpus']}  d={j['manifest']['d']}")
        ax.set_title(base + ("" if xlo <= a.xmin else "  (zoom)"), fontsize=8)
    order = [n for n, _, _, _ in STYLE if n in seen]
    fig.legend([seen[n] for n in order], order, frameon=False, fontsize=7.5,
               ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.03),
               columnspacing=1.4, handlelength=2.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=240, bbox_inches="tight")
    print(f"wrote {a.out}.pdf / .png")


if __name__ == "__main__":
    main()
