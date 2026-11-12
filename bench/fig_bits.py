"""What a narrower stored vector buys, and where it stops buying.

Two panels off one measurement. Left: the recall/time frontier of each stored
width, which is the question a reader has -- does 4 bits cost accuracy. Right:
bytes a vector against the time at a fixed recall, which is the question the
design has -- where do fewer bytes stop being fewer milliseconds.

The point of the right panel is the floor. Below 4 bits the bytes keep halving
and the time does not, because what is left is the unpacking and the
per-candidate bookkeeping, neither of which scales with the width.

    python bench/fig_bits.py --data results/degree/all_arms_msmarco.json
"""
import argparse, json, sys
from pathlib import Path


def frontier(pts):
    out, best = [], float("inf")
    for rc, t in sorted(set(pts), key=lambda p: (-p[0], p[1])):
        if t < best:
            out.append((rc, t)); best = t
    return sorted(out)


def time_at(front, rc):
    if not front or rc < front[0][0] or rc > front[-1][0]:
        return None
    for (r0, t0), (r1, t1) in zip(front, front[1:]):
        if r0 <= rc <= r1:
            return t0 if r1 == r0 else t0 + (t1 - t0) * (rc - r0) / (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("results/degree/all_arms_msmarco.json"))
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--target", type=float, default=0.99,
                    help="recall the right panel is read at")
    ap.add_argument("--width", type=float, default=3.1)
    ap.add_argument("--out", default="/root/swann-paper/figures/experiments/bits",
                    help="written as .pdf for LaTeX and .png for looking at; "
                         "the default lands in the paper's experiments folder")
    a = ap.parse_args()

    j = json.loads(a.data.read_text())
    rows = j["swann_rows"]
    arms = {}
    for r in rows:
        arms.setdefault(r["arm"], []).append(r)

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })
    # one hue, darkening as the vector widens: these are one system measured at
    # four precisions, not four systems
    SHADE = {4096: "#2A1B4A", 2048: "#6A3D9A", 512: "#9B6BC9", 256: "#C08FE3"}
    LABEL = {4096: "fp32", 2048: "fp16", 512: "4-bit", 256: "2-bit"}
    MARK = {4096: "o", 2048: "s", 512: "^", 256: "D"}

    fig, ax = plt.subplots(1, 2, figsize=(2 * a.width + 0.4,
                                          H.panel_h(a.width) + 0.5))
    byb = {}
    for name, rs in arms.items():
        b = rs[0]["bytes_per_vec"]
        f = frontier([(r["recall"], r["ms"]) for r in rs])
        byb[b] = f
        ax[0].plot([p[0] for p in f], [p[1] for p in f],
                   color=SHADE.get(b, "0.4"), marker=MARK.get(b, "o"),
                   ms=3.4, lw=1.35, label=f"{LABEL.get(b, name)}  ({b} B)")
    ax[0].set_xlabel("recall@10")
    ax[0].set_ylabel("batch ms")
    ax[0].set_yscale("log")
    ax[0].grid(True, which="both")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0].margins(x=0.04)
    ax[0].set_title("frontier by stored width", fontsize=8)

    xs, ys, lab = [], [], []
    for b in sorted(byb, reverse=True):
        t = time_at(byb[b], a.target)
        if t is not None:
            xs.append(b); ys.append(t); lab.append(LABEL.get(b, str(b)))
    ax[1].plot(xs, ys, color="#6A3D9A", marker="o", ms=4.2, lw=1.5, zorder=3)
    # the line fewer bytes would follow if bytes were all there was
    if xs:
        ideal = [ys[0] * x / xs[0] for x in xs]
        ax[1].plot(xs, ideal, color="0.55", ls=":", lw=1.1,
                   label="proportional to bytes")
    # Place each label by where its point sits, not by its index: the series is
    # ordered widest-first, so the *last* entry is the leftmost on the axis and
    # a label hung to its left is the one that gets clipped.
    lo, hi = min(xs), max(xs)
    for x, y, t in zip(xs, ys, lab):
        at_right = x == hi
        ax[1].annotate(t, (x, y), textcoords="offset points",
                       xytext=(-6 if at_right else 6, 7),
                       ha="right" if at_right else "left",
                       fontsize=7, color="#2A1B4A")
    ax[1].margins(x=0.12, y=0.15)
    ax[1].set_xscale("log", base=2)
    ax[1].set_xlabel("bytes per vector")
    ax[1].set_ylabel(f"batch ms at recall {a.target:g}")
    ax[1].grid(True, which="both")
    ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    ax[1].set_title("where bytes stop being milliseconds", fontsize=8)

    fig.tight_layout()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", dpi=240, bbox_inches="tight")
    print(f"wrote {a.out}.pdf / .png")
    for x, y, t in zip(xs, ys, lab):
        print(f"  {t:>6s} {x:5d} B  {y:7.2f} ms at recall {a.target:g}")


if __name__ == "__main__":
    main()
