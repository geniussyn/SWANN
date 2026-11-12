"""The recall/QPS figure style, in one file with no dependencies but matplotlib.

This is what `bench/fig_main_narrow.py`, `fig_bits.py` and `fig_cost_bits.py`
draw with, flattened so it can be copied to another machine on its own. It is
two layers that used to live apart: a house `setup()` (fonts, sizes, Type-42)
and the white-frame override each recall/QPS figure applied on top of it.

    import figstyle as F
    F.setup(9)                       # 9 pt base: a two-column paper panel
    fig, ax = plt.subplots(figsize=(3.1, F.panel_h(3.1) + 0.6))
    F.curve(ax, xs, ys, *F.SERIES[0])
    F.frontier_axes(ax, xlim=(0.60, 1.0))
    F.legend(fig, ax)

Three things are worth knowing before you change anything.

**Fonts.** The chain names Helvetica first and falls through to Nimbus Sans,
URW's metric-compatible clone and exactly what LaTeX renders for
`\\usepackage{helvet}`. A machine with a licensed Helvetica picks it up with no
edit and the metrics do not move. `font.family` is the explicit chain and not
the generic `sans-serif` alias, because matplotlib falls back *per glyph* only
through a list of real family names -- under the alias a missing glyph becomes
a last-resort box with nothing but a line on stderr to say so. If none of the
chain is installed every figure silently renders in DejaVu and the metrics
shift; `check_fonts()` below says so out loud.

**Type 42.** ACM camera-ready rejects Type 3 fonts, which is what matplotlib
emits into a PDF by default.

**The frame.** `setup()` starts from seaborn's darkgrid and then this module
turns the panel white with a grey dashed grid, because a recall/QPS panel is
mostly empty space above the curves and a grey field makes a log axis look
heavier than it is. If you want the darkgrid look back, drop `_white_frame()`.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

# --- colour ----------------------------------------------------------------
# The recall/QPS figures carry four families. SWANN's stored widths are shades
# of one hue, because they are one system measured twice and a reader should
# see them as a pair before they read the labels; each baseline gets a hue of
# its own. All four hold their own against each other in greyscale, which is
# the test that killed a sage green and a light amber earlier.
PURPLE = "#6A3D9A"      # the system under test
LILAC  = "#B48DDD"      # the same system, second variant
BLUE   = "#3B7DD8"      # first baseline
AMBER  = "#E8A33D"      # second baseline
INK    = "#262626"      # text, and a neutral fifth series
PANEL  = "#EAEAF2"      # seaborn's axes grey, for legend frames and dead cells

# (label, colour, marker, linestyle) in the order the legend should read.
SERIES = [
    ("series A", PURPLE, "^", "-"),
    ("series B", LILAC,  "D", "-"),
    ("series C", BLUE,   "o", "-"),
    ("series D", AMBER,  "s", "-"),
    ("series E", INK,    "v", "-"),
]
PALETTE = [PURPLE, BLUE, AMBER, INK, LILAC]

# A sequential ramp in the same hues, for panels carrying a magnitude rather
# than a set of series. It runs from the panel grey through amber to red, so
# "hot" reads as "worse" the way the categorical red already does.
RAMP_STOPS = ["#EAEAF2", "#F3D3A6", "#D97706", "#EA3323", "#7A1510"]


def ramp(name="house_warm"):
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, RAMP_STOPS)


# --- shape -----------------------------------------------------------------
# One panel shape for every sheet. Each figure keeps its own WIDTH, tuned to
# how much axis its data spans, and takes its height from here -- otherwise the
# same curve looks steeper in one figure than in the next.
PANEL_RATIO = 5.4 / 4.3          # 1.256 : 1


def panel_h(width):
    """The panel height that puts a panel of this width at the house shape."""
    return round(width / PANEL_RATIO, 3)


# Curve weights. At column width the markers separate the series -- a reader
# picks a triangle out from a square long before a solid line from a dashed one
# -- so the line is kept light enough not to smear where curves cross.
LW = 1.5
MS = 3.4
MARKEVERY_TARGET = 9             # markers per curve, whatever its point count


# --- frame -----------------------------------------------------------------
def _white_frame():
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })


FONT_CHAIN = ["Helvetica", "Helvetica Neue", "Nimbus Sans",
              "TeX Gyre Heros", "Arial", "DejaVu Sans"]


def setup(size=9, white=True):
    """Set the house frame at a given base font size.

    `size` is the body size of the figure, not of the document: 9 for a panel
    in a two-column paper, 13 for a full-width sheet or a slide.
    """
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,          # ACM rejects Type 3
        "figure.dpi": 130, "savefig.dpi": 200, "savefig.bbox": "tight",
        "font.family": FONT_CHAIN,
        "font.sans-serif": FONT_CHAIN,
        # Maths in the same face as the labels. The generic aliases are the
        # point: naming a specific clone here would pin the maths to it on a
        # machine whose text resolved to a real Helvetica, and the two would
        # sit side by side inside one label. cal/sf/tt are set because a custom
        # fontset otherwise leaves them at matplotlib's defaults, and the
        # default for cal is a cursive family almost nobody has.
        "mathtext.fontset": "custom",
        "mathtext.rm": "sans", "mathtext.it": "sans:italic",
        "mathtext.bf": "sans:bold", "mathtext.cal": "sans:italic",
        "mathtext.sf": "sans", "mathtext.tt": "monospace",
        "font.size": size,
        "axes.titlesize": size + 2, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1,
        "legend.fontsize": size - 2,
        "legend.frameon": True, "legend.framealpha": 0.92,
        "legend.edgecolor": "0.8",
        "lines.linewidth": LW, "lines.markersize": MS,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    if white:
        _white_frame()


def check_fonts(verbose=True):
    """Which of the font chain this machine actually has.

    Worth calling once in any script that will produce a final figure: a
    missing chain is silent apart from a `findfont` line on stderr, and the
    figure that comes out has different metrics from the one on the machine
    the paper was built on.
    """
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    found = [n for n in FONT_CHAIN if n in have]
    if verbose:
        missing = [n for n in FONT_CHAIN if n not in have]
        print(f"fonts present: {found or 'NONE OF THE CHAIN'}")
        if missing:
            print(f"fonts missing: {missing}")
        if found[:1] == ["DejaVu Sans"] or not found:
            print("  -> rendering in a fallback; re-render on a machine with "
                  "Helvetica or Nimbus Sans before submitting")
    return found


# --- drawing ---------------------------------------------------------------
def frontier(pts, maximise_y=True):
    """The Pareto frontier of (x, y), sorted by x.

    For recall against throughput: keep a point only if nothing at a higher
    recall is also faster. Pass `maximise_y=False` for (recall, milliseconds).
    """
    out, best = [], (0.0 if maximise_y else float("inf"))
    for x, y in sorted(set(map(tuple, pts)), key=lambda p: (-p[0], -p[1] if maximise_y else p[1])):
        if (y > best) if maximise_y else (y < best):
            out.append((x, y)); best = y
    return sorted(out)


def curve(ax, xs, ys, label=None, color=PURPLE, marker="o", ls="-", **kw):
    """One series, with markers thinned to about `MARKEVERY_TARGET` of them."""
    every = max(1, len(xs) // MARKEVERY_TARGET)
    h, = ax.plot(xs, ys, color=color, marker=marker, ls=ls, ms=MS, lw=LW,
                 markevery=every, markeredgewidth=0, label=label, **kw)
    return h


def frontier_axes(ax, xlim=(0.60, 1.0), xlabel="recall@10",
                  ylabel="queries / second", log=True, title=None):
    """The axes treatment every recall/QPS panel shares."""
    if log:
        ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.grid(True, which="both")
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=8)


def note_offscale(ax, entries):
    """Say in the panel that an arm never reaches its range.

    A legend entry whose curve is off-scale reads as a missing measurement, so
    the arms that stop early are named where they stopped instead.
    `entries` is (label, ceiling, colour).
    """
    for i, (label, ceiling, col) in enumerate(entries):
        ax.text(0.02, 0.06 + 0.07 * i, f"{label}: ceiling {ceiling:.3f}",
                transform=ax.transAxes, fontsize=6.5, color=col, va="bottom")


def legend(fig, handles, labels, ncol=None):
    """The house legend: one unframed strip under the panels."""
    fig.legend(handles, labels, frameon=False, fontsize=7.5,
               ncol=ncol or len(labels), loc="lower center",
               bbox_to_anchor=(0.5, -0.03), columnspacing=1.4, handlelength=2.2)
    return (0, 0.07, 1, 1)          # tight_layout rect that reserves the strip


def save(fig, stem, exts=("pdf", "png"), dpi=240):
    """Write the figure beside itself as PDF and PNG.

    The PDF is what LaTeX takes; the PNG is for looking at. Writing both from
    one call is the only way they stay the same picture.
    """
    from pathlib import Path
    Path(stem).parent.mkdir(parents=True, exist_ok=True)
    for ext in exts:
        fig.savefig(f"{stem}.{ext}", dpi=dpi, bbox_inches="tight")
    return [f"{stem}.{e}" for e in exts]


if __name__ == "__main__":
    # A worked example that is also the smoke test: two panels, four series,
    # one arm deliberately stopping early so the off-scale note is exercised.
    import numpy as np
    setup(9)
    check_fonts()
    rng = np.random.default_rng(0)
    data = {}
    for i, (name, col, mk, ls) in enumerate(SERIES[:4]):
        rc = np.linspace(0.70, 0.999 - 0.02 * i, 40)
        qps = 4e5 * (1 - rc) ** 1.6 * (1.0 - 0.18 * i) + 2e3
        data[name] = (rc, qps, col, mk, ls)
    fig, axes = plt.subplots(1, 2, figsize=(2 * 3.1 + 0.4, panel_h(3.1) + 0.6))
    handles, labels = [], []
    for pi, (lo, tag) in enumerate(((0.70, ""), (0.975, "  (zoom)"))):
        ax = axes[pi]
        off = []
        for name, (rc, qps, col, mk, ls) in data.items():
            vis = [(x, y) for x, y in zip(rc, qps) if x >= lo]
            if vis:
                h = curve(ax, [p[0] for p in vis], [p[1] for p in vis],
                          color=col, marker=mk, ls=ls)
                if pi == 0:
                    handles.append(h); labels.append(name)
            else:
                off.append((name, rc[-1], col))
        frontier_axes(ax, xlim=(lo, 1.0), ylabel="queries / second" if pi == 0 else None,
                      title="example corpus  d=1024" + tag)
        note_offscale(ax, off)
    rect = legend(fig, handles, labels)
    fig.tight_layout(rect=rect)
    # beside this file, not in /tmp: the demo is the first thing someone runs
    # after copying the module to a new machine, and it should leave its output
    # where they are standing.
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    print("wrote", save(fig, os.path.join(here, "demo")))
