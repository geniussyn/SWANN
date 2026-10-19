"""ours against CAGRA and GGNN on one panel per corpus, all queries.

Each system is drawn on its own recall-QPS frontier, built from whatever knob
that system actually exposes: queue width for ours, itopk for CAGRA, and
k_build for GGNN -- whose tau_query turned out not to move its recall at all,
so sweeping it would have drawn a vertical line and called it a curve.
"""
import argparse, glob, json, sys
from pathlib import Path


def pts_ours(pattern):
    out = []
    for f in sorted(glob.glob(pattern)):
        r = json.loads(Path(f).read_text())["rows"][0]
        out.append((r["qps"], r["recall"]))
    return out


def pts_native(path, algo=None):
    j = json.loads(Path(path).read_text())
    return [(r["qps"], r["recall"]) for r in j["rows"]
            if algo is None or r["algo"] == algo]


def pts_ggnn(patterns):
    """One index per k_build; the fastest tau that reaches its ceiling."""
    out = []
    for f in sorted(glob.glob(patterns)):
        j = json.loads(Path(f).read_text())
        best = max(j["rows"], key=lambda r: (round(r["recall"], 4), r["qps"]))
        out.append((best["qps"], best["recall"]))
    return out


def frontier(pts):
    out, best = [], -1.0
    for q, r in sorted(pts, key=lambda p: -p[0]):
        if r > best:
            out.append((q, r)); best = r
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="name:ours_glob:cagra_json:ggnn_glob (ggnn may be empty)")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=3.6)
    ap.add_argument("--out", default="results/final/fig_three_way")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(13)

    panels = [p.split(":") for p in a.panel]
    fig, axes = plt.subplots(1, len(panels),
                             figsize=(len(panels) * a.width + 0.4,
                                      H.panel_h(a.width) + 0.5), squeeze=False)
    for i, (name, og, cj, gg) in enumerate(panels):
        ax = axes[0][i]
        series = [(pts_native(cj, "single_cta"), "CAGRA", H.RED, "s"),
                  (pts_ours(og), "ours", H.BLUE, "o")]
        if gg:
            series.insert(0, (pts_ggnn(gg), "GGNN", H.GREEN, "^"))
        for pts, label, colour, mk in series:
            f = frontier(pts)
            if not f:
                continue
            ax.plot([p[0] for p in f], [p[1] for p in f], marker=mk, color=colour,
                    lw=H.LW, ms=H.ms(H.MS[mk]), mew=H.mew(mk), label=label, zorder=3)
        ax.set_xscale("log")
        ax.set_xlabel("QPS")
        ax.set_ylabel("recall@10" if i == 0 else "")
        ax.set_title(name, fontsize=11)
        ax.set_xticks([t for t in (20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000)
                       if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
        ax.set_xticks([], minor=True)
        ax.set_xticklabels([f"{int(t/1000)}k" if t < 1e6 else f"{t/1e6:g}M"
                            for t in ax.get_xticks()])
    axes[0][0].legend(loc="lower left", frameon=True, facecolor=H.PANEL)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); print(f"wrote {p}")


if __name__ == "__main__":
    main()
