"""Build time and storage, which run the opposite way to the search results.

Two panels because they are two different costs and a reader trades them
separately: the time before the first query can be answered, and the bytes the
method occupies for as long as it serves. Storage counts the vectors when the
method still needs them, which is what separates IVF-PQ with refinement from
IVF-PQ without it.
"""
import argparse, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/buildstore")
    ap.add_argument("--corpus", default="msmarco-1m")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--out", default="results/final/fig_cost")
    a = ap.parse_args()

    rows = {}
    for suf in ("cuvs", "rabitq", "swann"):
        p = Path(a.root) / f"{a.corpus}_{suf}.json"
        if not p.exists():
            continue
        j = json.loads(p.read_text())
        for r in j.get("rows", []):
            name = r.get("system") or r.get("method")
            rows.setdefault(name, {}).update(r)

    ORDER = [("IVF-RaBitQ (3-bit)", "#E8B018"), ("IVF-RaBitQ (8-bit)", "#D4682A"),
             ("IVF-PQ (wo/ref)", "#EE8C84"), ("IVF-PQ (w/ref)", "#B4232B"),
             ("IVF-FLAT", "#17A2A2"), ("CAGRA", "#3B7DD8"), ("SWANN", "#6A3D9A")]

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator, NullFormatter
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))
    names, build, store, cols = [], [], [], []
    for name, colour in ORDER:
        r = rows.get(name)
        if not r:
            continue
        names.append(name.replace("IVF-RaBitQ ", "RaBitQ ")
                         .replace("IVF-PQ ", "PQ ")
                         .replace("(wo/ref)", "no ref")
                         .replace("(w/ref)", "+ref")
                         .replace("(3-bit)", "3b")
                         .replace("(8-bit)", "8b"))
        build.append(r.get("build_s"))
        store.append(r.get("total_bytes", 0) / 1e9)
        cols.append(colour)
    xs = range(len(names))
    axes[0].bar(xs, [b if b else 0 for b in build], color=cols,
                edgecolor="0.15", linewidth=0.5, zorder=3)
    axes[1].bar(xs, store, color=cols, edgecolor="0.15", linewidth=0.5, zorder=3)
    raw = next((r["total_bytes"] / 1e9 for n, r in rows.items()
                if n.startswith("Raw vectors (fp32)")), None)
    if raw:
        axes[1].axhline(raw, color="0.25", lw=0.9, ls=(0, (4, 2)), zorder=4)
        axes[1].text(0.1, raw * 1.04, "the vectors alone",
                     fontsize=6.2, color="0.3", ha="left")
    for ax, ylab in ((axes[0], "build time (s)"), (axes[1], "storage (GB)")):
        ax.set_xticks(list(xs))
        ax.set_xticklabels(names, fontsize=6.4, rotation=20, ha="right")
        ax.set_ylabel(ylab, fontsize=7.8)
        ax.grid(True, axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.8)
    axes[0].set_yscale("log")
    axes[0].yaxis.set_minor_locator(NullLocator())
    axes[0].yaxis.set_minor_formatter(NullFormatter())
    axes[0].set_yticks([1, 10, 100])
    axes[0].set_yticklabels(["1", "10", "100"])
    fig.tight_layout()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
