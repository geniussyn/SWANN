"""One query in SWANN, drawn as a dataflow.

Kept deliberately plain: a single column of stages, one enclosing box for the
part that repeats, and the shared structures to its right. Everything that
could be said in the caption is said there instead, because a diagram carrying
sentences stops being a diagram.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--out", default="results/final/fig_arch")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    H.setup(9)
    plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white"})

    PEACH, BLUE, VIOLET, GREEN = "#FBE0C8", "#CFE2F3", "#E4D5F5", "#D7EBD5"
    EDGE, GREY, INK, SHARE = "#5F5F5F", "#9A9A9A", "#1A1A1A", "#8E5FD3"

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_xlim(-18, 100); ax.set_ylim(0, 112); ax.axis("off")

    def box(x, y, w, h, text, fc, fs=8.2, ec=EDGE, lw=0.9):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.5,rounding_size=1.5",
                                    facecolor=fc, edgecolor=ec, linewidth=lw,
                                    zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=4)

    def arr(x1, y1, x2, y2, colour=INK, lw=0.9, style="-|>", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                     mutation_scale=8, color=colour, linewidth=lw,
                                     zorder=2, shrinkA=0.5, shrinkB=0.5,
                                     connectionstyle=f"arc3,rad={rad}"))

    CX, W = 8, 46
    SX, SW = 66, 30

    # the repeating part, one plain box with its multiplier outside
    ax.add_patch(FancyBboxPatch((CX - 3.5, 26), W + 7, 58,
                                boxstyle="round,pad=0.8,rounding_size=2",
                                facecolor="#F6F6F6", edgecolor=GREY,
                                linewidth=0.9, zorder=1))
    ax.text(CX - 14.5, 55, r"$V\times$", ha="center", va="center", fontsize=10,
            color=GREY, rotation=90)

    box(CX + 7, 2, W - 14, 8.5, "Query", "white")
    arr(CX + W / 2, 10.5, CX + W / 2, 15)
    box(CX, 15, W, 8.5, "Seed Selection", GREEN)
    arr(CX + W / 2, 23.5, CX + W / 2, 29)

    for y, text, fc in ((29, "Pop Nearest", BLUE),
                        (42, "Claim Neighbours", VIOLET),
                        (55, "Distance", PEACH),
                        (68, "Merge into Queue", BLUE)):
        box(CX, y, W, 9.5, text, fc)
    for y in (38.5, 51.5, 64.5):
        arr(CX + W / 2, y, CX + W / 2, y + 3.5)

    # The iteration runs outside the enclosing box. Inside, it crossed the
    # stages it is supposed to be wrapping around and read as another edge of
    # the dataflow rather than as the loop.
    arr(CX - 5.5, 72.5, CX - 5.5, 33.5, colour=GREY, lw=0.8, rad=0.30)

    arr(CX + W / 2, 77.5, CX + W / 2, 86)
    box(CX, 86, W, 8.5, "Top-$k$ Merge", "white")
    arr(CX + W / 2, 94.5, CX + W / 2, 101)
    box(CX + 7, 101, W - 14, 8.5, "Output", "white")

    # the three structures, reached by one line rather than a bundle
    for y, text in ((29, "Shared Bound"), (46, "Claim Map"),
                    (63, "Distance Table")):
        box(SX, y, SW, 9.5, text, "white", ec=SHARE, lw=1.1)
        arr(SX - 6, y + 4.75, SX - 0.6, y + 4.75, colour=SHARE, lw=0.9,
            style="<|-|>")
    ax.plot([SX - 6, SX - 6], [33.75, 67.75], color=SHARE, lw=1.1, zorder=1,
            solid_capstyle="round")
    ax.plot([CX + W + 3.5, SX - 6], [50.75, 50.75], color=SHARE,
            lw=1.1, zorder=2, solid_capstyle="round")
    ax.text(SX + SW / 2, 77, "Shared by all $V$ branches", ha="center",
            fontsize=7.2, color=SHARE)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
