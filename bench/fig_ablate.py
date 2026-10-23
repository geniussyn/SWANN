"""What each mechanism moves, drawn in the plane the paper is judged in.

A table of recall and throughput makes the reader do the subtraction; the same
numbers as a path through (recall, QPS) show it directly. Each panel walks one
corpus from nothing shared to all three, so the direction of every step is the
claim: the table is free speed, claiming buys recall and spends speed, and the
bound gives the speed back.
"""
import argparse, glob, json, sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/ablate")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--out", default="results/final/fig_ablate")
    a = ap.parse_args()

    rows = {}
    for f in glob.glob(f"{a.root}/*.json"):
        stem = Path(f).stem
        ds, rest = stem.split("_V", 1)
        V, rest = rest.split("_e", 1)
        eps, combo = rest.split("_", 1)
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            rows[(ds, int(V), combo)] = (r["recall"], r["qps"])

    CORPORA = [("MS MARCO 1M", "msmarco-1m"), ("FEVER 1M", "fever-1m"),
               ("GIST 1M", "gist-960"), ("GloVe 1.18M", "glove-100"),
               ("Qwen 1024", "qwen-1024")]
    STEPS = [("none", "none"), ("c", "+table"), ("cx", "+claims"),
             ("cbx", "+bound")]
    COL = {32: "#8E5FD3", 128: "#3B7DD8"}

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })

    # Each mechanism's own effect, rather than a path through the plane: the
    # path drew "none" and "+table" on top of each other, because the table
    # changes throughput at identical recall, which is exactly the finding it
    # then failed to show.
    MECH = [("+table", "c", "none", "#17A2A2"),
            ("+claims", "cx", "c", "#8E5FD3"),
            ("+bound", "cbx", "cx", "#E8B018")]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.2))
    labels = [t for t, _ in CORPORA]
    nm = len(MECH)
    width = 0.8 / nm
    seen = {}
    for si, (name, to, frm, colour) in enumerate(MECH):
        dr, dq, xs = [], [], []
        for ci, (title, ds) in enumerate(CORPORA):
            A, B = rows.get((ds, 128, frm)), rows.get((ds, 128, to))
            if not (A and B):
                continue
            xs.append(ci + (si - (nm - 1) / 2) * width)
            dr.append(B[0] - A[0])
            dq.append(B[1] / A[1])
        b1 = axes[0].bar(xs, dr, width=width * 0.9, color=colour,
                         edgecolor="0.15", linewidth=0.5, zorder=3)
        axes[1].bar(xs, dq, width=width * 0.9, color=colour,
                    edgecolor="0.15", linewidth=0.5, zorder=3)
        seen[name] = b1
    for ax, ylab, ref in ((axes[0], "change in recall@10", 0.0),
                          (axes[1], "throughput ratio", 1.0)):
        ax.axhline(ref, color="0.25", lw=0.9, zorder=4)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6.6, rotation=18, ha="right")
        ax.set_ylabel(ylab, fontsize=7.8)
        ax.grid(True, axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=6.8)
    from matplotlib.ticker import NullLocator, NullFormatter
    axes[1].set_yscale("log")
    axes[1].yaxis.set_minor_locator(NullLocator())
    axes[1].yaxis.set_minor_formatter(NullFormatter())
    axes[1].set_yticks([0.4, 0.5, 1, 2, 3])
    axes[1].set_yticklabels(["0.4", "0.5", "1", "2", "3"])
    order = list(seen)
    fig.legend([seen[k] for k in order], order, loc="upper center",
               ncol=len(order), frameon=True, fontsize=7,
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
