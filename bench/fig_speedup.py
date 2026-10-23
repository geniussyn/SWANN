"""Speed-up at a fixed target recall, on four corpora chosen to span the range.

Four rather than ten, and drawn rather than tabulated, because the finding is a
shape and not a set of numbers: the advantage grows with dimension, and two
corpora run against it. The four are the ones that carry that --- the low
dimension where we lose, the corpus where the quantised baseline wins, a
mid-dimensional text corpus, and the widest.

Arms are matched in precision. A bar below the line at $1$ is a loss, and a
missing bar is a baseline that does not reach the target at all, which is a
different statement from a slow one and is drawn as a gap.
"""
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speedup_table import pts, best_at


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean")
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.99])
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--out", default="results/final/fig_speedup")
    a = ap.parse_args()
    r = a.root

    CORPORA = [("SIFT 1M", "sift-128", 128),
               ("GIST 1M", "gist-960", 960),
               ("MS MARCO 1M", "msmarco-1m", 1024),
               ("OpenAI-3072 1M", "openai-3072", 3072)]

    def o32(d): return pts(f"{r}/{d}_swann_e*.json,{r}/{d}_swann_B16_e*.json,"
                           f"!{r}/{d}_swann_e*_h.json,!{r}/{d}_swann_B16_e*_h.json")
    def o16(d): return pts(f"{r}/{d}_swann_e*_h.json,{r}/{d}_swann_B16_e*_h.json")
    def c32(d): return pts(f"{r}/{d}_cagra.json")
    def c16(d): return pts(f"{r}/{d}_cagra_h.json")
    def rab(d): return pts(f"{r}/{d}_rabitq_b*.json")

    SERIES = [("vs CAGRA, fp32", o32, c32, "#3B7DD8", ""),
              ("vs CAGRA, fp16", o16, c16, "#3B7DD8", "//"),
              ("vs IVF-RaBitQ, fp32", o32, rab, "#E8B018", ""),
              ("vs IVF-RaBitQ, fp16", o16, rab, "#E8B018", "//")]

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })

    fig, axes = plt.subplots(1, len(CORPORA), squeeze=False,
                             figsize=(len(CORPORA) * 1.65 + 0.5, 2.1),
                             sharey=True)
    seen = {}
    for ci, (title, ds, dim) in enumerate(CORPORA):
        ax = axes[0][ci]
        nser = len(SERIES)
        width = 0.8 / nser
        for si, (label, ours, base, colour, hatch) in enumerate(SERIES):
            xs, hs = [], []
            for ti, t in enumerate(a.targets):
                x = ti + (si - (nser - 1) / 2) * width
                A, B = best_at(ours(ds), t), best_at(base(ds), t)
                if A and B:
                    xs.append(x); hs.append(A / B)
            # A baseline that never reaches the target has no ratio. Leaving a
            # silent gap would read as a missing measurement, so the gap is
            # marked for what it is.
            for ti, t in enumerate(a.targets):
                x = ti + (si - (nser - 1) / 2) * width
                if best_at(ours(ds), t) and not best_at(base(ds), t):
                    ax.text(x, 1.06, "\u00d7", ha="center", va="bottom",
                            fontsize=7, color=colour, zorder=5)
            if not xs:
                continue
            b = ax.bar(xs, hs, width=width * 0.92, color=colour, hatch=hatch,
                       edgecolor="0.15", linewidth=0.5, zorder=3)
            seen.setdefault(label, b)
        ax.axhline(1.0, color="0.25", lw=0.9, zorder=4)
        ax.set_yscale("log")
        ax.set_xticks(range(len(a.targets)))
        ax.set_xticklabels([f"{t:g}" for t in a.targets], fontsize=7.5)
        ax.set_xlabel("Recall@10", fontsize=7.5)
        ax.set_title(f"{title}  ($D={dim}$)", fontsize=7.8)
        ax.grid(True, axis="y", alpha=0.9)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7)
        if ci == 0:
            ax.set_ylabel("speed-up", fontsize=7.8)
    axes[0][0].set_yticks([0.5, 1, 2, 4, 8])
    axes[0][0].set_yticklabels(["0.5", "1", "2", "4", "8"])
    order = [k for k, _, _, _, _ in SERIES if k in seen]
    fig.legend([seen[k] for k in order], order, loc="upper center",
               ncol=len(order), frameon=True, fontsize=7,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
