"""Index build time, one panel per corpus, one bar per method.

Build time spans three orders of magnitude between a quantised index and a
proximity graph, so each panel carries its own linear scale: the comparison a
reader makes is between methods on one corpus, and a shared log axis would
flatten exactly that. The cost of the choice is that panel heights are not
comparable, which the axis labels make plain.

A method that could not be measured on a corpus is left out of that panel
rather than drawn at zero, since zero is a claim and an absence is not.
"""
import argparse, glob, json
from pathlib import Path

ORDER = ["IVF-RaBitQ", "IVF-FLAT", "IVF-PQ", "CAGRA", "SWANN"]
COLOUR = {"IVF-RaBitQ": "#E8B018", "IVF-FLAT": "#17A2A2", "IVF-PQ": "#B4232B",
          "CAGRA": "#3B7DD8", "SWANN": "#6A3D9A"}
HATCH = {"IVF-RaBitQ": "", "IVF-FLAT": "..", "IVF-PQ": "xx", "CAGRA": "//",
         "SWANN": ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="results/buildtime/*.json")
    ap.add_argument("--order", nargs="+", default=None,
                    help="corpus labels, left to right; default is by dimension")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=1.05)
    ap.add_argument("--out", default="results/final/fig_build_time")
    a = ap.parse_args()

    files = sorted(glob.glob(a.glob))
    data = []
    for f in files:
        j = json.loads(Path(f).read_text())
        data.append(j)
    if not data:
        raise SystemExit(f"no files matched {a.glob}")
    if a.order:
        pos = {c: i for i, c in enumerate(a.order)}
        data.sort(key=lambda j: pos.get(j["corpus"], 1e9))
    else:
        data.sort(key=lambda j: (j.get("d", 0), j.get("n", 0)))

    import sys
    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })

    fig, axes = plt.subplots(1, len(data), squeeze=False,
                             figsize=(len(data) * a.width + 0.6, 1.9))
    seen = {}
    for i, j in enumerate(data):
        ax = axes[0][i]
        got = {r["method"]: r["build_s"] for r in j["rows"]
               if r.get("build_s") is not None}
        methods = [m for m in ORDER if m in got]
        for x, m in enumerate(methods):
            b = ax.bar(x, got[m], width=0.72, color=COLOUR[m],
                       edgecolor="0.15", linewidth=0.6, hatch=HATCH[m])
            seen.setdefault(m, b)
        ax.set_xticks([])
        ax.set_xlabel("Method", fontsize=7)
        ax.set_title(j["corpus"], fontsize=7.5)
        ax.grid(True, axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.5)
        if i == 0:
            ax.set_ylabel("Build time (s)", fontsize=7.5)
    order = [m for m in ORDER if m in seen]
    fig.legend([seen[m] for m in order], order, loc="upper center",
               ncol=len(order), frameon=True, fontsize=7,
               bbox_to_anchor=(0.5, 1.02))
    devices = sorted({j.get("device", "unknown") for j in data})
    if len(devices) == 1:
        fig.text(0.5, -0.04, f"measured on {devices[0]}", ha="center",
                 fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png ({len(data)} corpora, "
          f"devices: {', '.join(devices)})")


if __name__ == "__main__":
    main()
