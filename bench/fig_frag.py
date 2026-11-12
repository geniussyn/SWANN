"""Why a wider queue does not reach a scattered answer, and what does.

A schematic, not a measurement. The layer is drawn as a slab seen at an angle
so that the seeds can come down onto it from the query, which is the one thing
the flat version could not show: in (a) a single descent, in (b) $V$ of them
landing apart. Kept sparse on purpose -- five answer nodes and a thin corpus,
because the figure has to be read at a column width.
"""
import argparse
from pathlib import Path

import numpy as np

SKEW, FLAT = 0.45, 0.40      # plane (u,v) -> screen (u + SKEW v, FLAT v)
DROP = 0.42                  # how far above the layer a seed starts
SIDE = 1.12                  # half-width of the drawn slab


def shade(hex_colour, k):
    """The same hue, k of the way to black."""
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % (int(r * k), int(g * k), int(b * k))


def proj(P):
    P = np.atleast_2d(P)
    return np.c_[P[:, 0] + SKEW * P[:, 1], FLAT * P[:, 1]]


RING, CLEAR, RIM = 0.52, 0.61, 1.00   # answers, then the nearest corpus, then the rim


def layout(seed=11, n=170, nans=5):
    """The answer nodes, then the corpus.

    The answers are the exact top-$k$, so nothing in the corpus may be nearer
    the query than they are: they sit on a ring at RING and every other node
    is beyond CLEAR. The ring leaves the query's neighbourhood empty, which is
    what being the nearest neighbours looks like.
    """
    rng = np.random.default_rng(seed)
    ang = np.linspace(0, 2 * np.pi, nans, endpoint=False) + 1.25
    rad = RING + 0.025 * rng.standard_normal(nans)       # near-equidistant
    ans = np.c_[rad * np.cos(ang), rad * np.sin(ang)]
    base = RIM * rng.uniform(-1.0, 1.0, (n, 2))
    d = np.linalg.norm(base, axis=1)
    keep = (d < RIM) & (d > CLEAR)
    keep &= np.min(np.linalg.norm(base[:, None] - ans[None], axis=2), axis=1) > 0.09
    assert d[keep].min() > np.linalg.norm(ans, axis=1).max()
    return base[keep], ans


def spread(T, r=0.15, n=72):
    """Slide each landing spot around its own piece.

    The pieces lie on a ring, and a sheared ring projects into a narrow band
    of screen x, so drops aimed at the centres collide with each other and
    with the query. Each spot is free to sit anywhere on its piece, so take
    the one that is furthest, in projection, from everything placed already.
    """
    out = T.copy()
    taken = [0.0]                                   # the query, at the origin
    for i in np.argsort(-np.abs(proj(T)[:, 0])):    # outermost piece first
        best, spot = -1.0, None
        for t in np.linspace(0, 2 * np.pi, n, endpoint=False):
            c = T[i] + r * np.array([np.cos(t), np.sin(t)])
            d = min(abs(proj(c)[0, 0] - u) for u in taken)
            if d > best:
                best, spot = d, c
        out[i] = spot
        taken.append(proj(spot)[0, 0])
    return out


def knn_edges(P, k=3, nans=0):
    """Neighbour edges, minus any edge joining two answers.

    That omission is the whole premise: the answers are mutually close, so a
    raw $k$-NN graph would link them and there would be no pieces to speak of.
    A pruned serving graph does not, and the walk can only travel on edges.
    """
    d = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    lo = len(P) - nans
    return {(min(i, j), max(i, j)) for i, row in enumerate(nn) for j in row
            if not (i >= lo and j >= lo)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--green", default="#4F8A00")
    ap.add_argument("--violet", default="#9080C6")
    ap.add_argument("--out", default="/root/swann-paper/figures/frag")
    a = ap.parse_args()
    import sys
    sys.path.insert(0, a.style)
    import housestyle as H                                       # noqa: F401
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Polygon
    from matplotlib.transforms import Affine2D
    H.setup(9)
    plt.rcParams.update({"axes.facecolor": "white", "figure.facecolor": "white"})

    GREEN = a.green       # the answer's pieces
    GREY = "#C0C0C0"      # everything else in the corpus
    EDGE, SLAB = "#DEDEDE", "#BFBFBF"
    RED = "#CC3344"       # the query
    # The dashed lobes are large and can afford a pale violet; the arrows and
    # the labels are hairlines and small type, so they take a deeper one.
    VIOLET = a.violet     # the ground a branch covers
    INK = "#1A1A1A"
    PEN = shade(VIOLET, 0.68)   # seeds, and anything set in type

    base, ans = layout()
    pts = np.vstack([base, ans])
    E = knn_edges(pts, nans=len(ans))
    S = proj(pts)
    # circles live on the layer, so they are drawn through the same shear
    TR = Affine2D(np.array([[1.0, SKEW, 0.0], [0.0, FLAT, 0.0], [0, 0, 1.0]]))

    fig, axes = plt.subplots(1, 2, figsize=(3.33, 1.30))
    for ax in axes:
        ax.set_xlim(-1.78, 1.78); ax.set_ylim(-0.58, 0.72)
        ax.set_aspect("equal"); ax.axis("off")
        ax.add_patch(Polygon(proj(np.array([[-SIDE, -SIDE], [SIDE, -SIDE],
                                            [SIDE, SIDE], [-SIDE, SIDE]])),
                             closed=True, fc="#F5F5F5", ec=SLAB, lw=0.7,
                             zorder=0))
        for i, j in E:
            ax.plot(*zip(S[i], S[j]), color=EDGE, lw=0.28, zorder=1)
        ax.scatter(S[:len(base), 0], S[:len(base), 1], s=3.0, c=GREY, lw=0,
                   zorder=2)
        ax.scatter(S[len(base):, 0], S[len(base):, 1], s=9, c=GREEN, lw=0,
                   zorder=4)
        ax.scatter(0, 0, marker="*", s=62, c=RED, lw=0, zorder=6)

    def drop(ax, to, lw=0.75):
        """A seed coming straight down onto its place on the layer."""
        q = proj(to)[0]
        tail = (q[0], q[1] + DROP)
        ax.annotate("", xy=q, xytext=tail, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=PEN, lw=lw,
                                    mutation_scale=6.5, shrinkA=0, shrinkB=1.5))
        return tail

    # (a) one seed, and the region it explores. Drawn as a lobe along the
    # branch it descended rather than a ball around the query: a walk travels
    # on edges, so it reaches outward from where it entered. Both lobes hold
    # the query and one answer node; the next answer is 0.47 away, across the
    # grain, and no setting of the budget turns the lobe sideways.
    ax = axes[0]
    axis = ans[0] / np.linalg.norm(ans[0])
    mid = ans[0] * 0.60                          # between query and answer
    drop(ax, ans[0] * 0.26, lw=0.9)
    for ha, hb in ((0.32, 0.17), (0.48, 0.26)):
        ax.add_patch(Ellipse(mid, 2 * ha, 2 * hb, fill=False, lw=0.75,
                             angle=np.degrees(np.arctan2(*axis[::-1])),
                             ls=(0, (2.5, 1.8)), ec=VIOLET, zorder=3,
                             transform=TR + ax.transData))
    p = proj(mid + 0.66 * axis)[0]
    ax.text(p[0], p[1], "search\nbudget", ha="left", va="center",
            fontsize=5.4, color=PEN, zorder=5, linespacing=1.25)
    # the knob turned up. Drawn long with no shrink: the default 2pt shrink
    # at each end leaves a short arrow as a bare arrowhead.
    ax.annotate("", xy=(p[0] + 0.60, p[1] + 0.21),
                xytext=(p[0] + 0.60, p[1] - 0.19), zorder=5,
                arrowprops=dict(arrowstyle="-|>", color=PEN, lw=0.7,
                                mutation_scale=4.5, shrinkA=0, shrinkB=0))
    ax.text(*proj(np.array([-SIDE + 0.04, -SIDE - 0.10]))[0], "base layer",
            ha="left", va="top", fontsize=5.4, color="#8A8A8A", style="italic")
    ax.set_title("(a) one descent, any budget", fontsize=6.6, color=INK, pad=2)

    # (b) one seed per piece, held apart
    ax = axes[1]
    tails = []
    for c, spot in zip(ans, spread(ans * 1.12, r=0.15)):
        tails.append(drop(ax, spot))
        ax.add_patch(Circle(c, 0.19, fill=False, ls=(0, (2.5, 1.8)), lw=0.75,
                            ec=VIOLET, zorder=3, transform=TR + ax.transData))
    tails = np.array(tails)
    left = tails[np.argmin(tails[:, 0])]           # beside the first drop
    ax.text(left[0] - 0.09, left[1], "$V$", ha="right", va="center",
            fontsize=6.2, color=PEN, zorder=6)
    ax.set_title("(b) $V$ branches, held apart", fontsize=6.6, color=INK, pad=2)

    h = [plt.Line2D([], [], marker="*", ls="", ms=6.5, c=RED),
         plt.Line2D([], [], marker="o", ls="", ms=3.0, c=GREEN),
         plt.Line2D([], [], marker="o", ls="", ms=2.2, c=GREY)]
    fig.legend(h, ["query", "answer", "corpus"], loc="lower center", ncol=3,
               frameon=False, fontsize=6.2, handletextpad=0.5,
               columnspacing=1.6, bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0, 0.05, 1, 1), w_pad=0.3)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
