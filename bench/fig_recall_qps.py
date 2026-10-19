"""The recall-QPS sheet, in the lab's house style.

Two panels off one set of runs: the same batch, the same timings, read once over
all queries and once over the hardest decile. Drawing them side by side is the
point -- a method that only looks good on the average is visible immediately.

Style comes from the lab's `housestyle`, so this sheet sits with the others
without a reader having to relearn the colours: BLUE is the arm under test,
RED the baseline, and the frame, marker sizes and panel shape are the module's.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np


def load_ours(pattern):
    import glob
    rows = []
    for f in sorted(glob.glob(pattern)):
        j = json.loads(Path(f).read_text())
        for r in j["rows"]:
            if r.get("recall_hard") is None:
                continue
            rows.append((r["recall"], r["recall_hard"], r["qps"], r["m"], r["E"]))
    return sorted(rows, key=lambda x: x[2])


def load_native(path, algo="single_cta"):
    j = json.loads(Path(path).read_text())
    rows = [(r["recall"], r["recall_hard"], r["qps"], r["itopk"], r["width"])
            for r in j["rows"] if r["algo"] == algo and r.get("recall_hard") is not None]
    return sorted(rows, key=lambda x: x[2])


def frontier(rows, yi):
    """Keep only points no other point beats on both axes (higher QPS, higher recall)."""
    out, best = [], -1.0
    for r in sorted(rows, key=lambda x: -x[2]):          # QPS descending
        if r[yi] > best:
            out.append(r); best = r[yi]
    return sorted(out, key=lambda x: x[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="results/final/ours_hardsplit_*.json")
    ap.add_argument("--native", default="results/final/cagra_hardsplit.json")
    ap.add_argument("--style", default="/workspace/SYN/code",
                    help="directory holding the lab's housestyle.py")
    ap.add_argument("--width", type=float, default=3.4, help="panel width in inches")
    ap.add_argument("--out", default="results/final/fig_recall_qps")
    ap.add_argument("--title", default="MS MARCO 1M (100K subset), d=1024, fp32, k=10")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt

    H.setup(13)
    ours, nat = load_ours(a.ours), load_native(a.native)
    if not ours or not nat:
        raise SystemExit("no rows: run the benches with --hard first")

    fig, axes = plt.subplots(1, 2, figsize=(2 * a.width + 0.6, H.panel_h(a.width) + 0.35))
    allq = [r[2] for r in ours + nat]
    lo_x, hi_x = min(allq), max(allq)
    panels = [(0, "all queries", 0), (1, "hardest 10%", 1)]
    for ax_i, label, yi in panels:
        ax = axes[ax_i]
        for rows, name, colour, mk in ((nat, "CAGRA", H.RED, "s"),
                                       (ours, "ours", H.BLUE, "o")):
            f = frontier(rows, yi)
            ax.plot([r[2] for r in f], [r[yi] for r in f],
                    marker=mk, color=colour, lw=H.LW, ms=H.ms(H.MS[mk]),
                    mew=H.mew(mk), label=name, zorder=3)
        ax.set_xscale("log")
        # A log axis here fills itself with 3x10^4-style minor labels that
        # collide at panel width; QPS is read off a handful of round values, so
        # the ticks are set to those and nothing else.
        ticks = [t for t in (25_000, 50_000, 100_000, 200_000, 300_000)
                 if lo_x / 1.5 <= t <= hi_x * 1.5]
        ax.set_xticks(ticks, minor=False)
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([f"{t//1000}k" for t in ticks])
        ax.set_xlabel("QPS (batch of 3490 queries)")
        ax.set_ylabel("recall@10" if ax_i == 0 else "recall@10, hardest 10%")
        ax.set_title(label, fontsize=11)
    axes[0].legend(loc="lower left", frameon=True, facecolor=H.PANEL)
    fig.suptitle(a.title, fontsize=10, color=H.INK, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}")
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p)
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
