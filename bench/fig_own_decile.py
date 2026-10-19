"""recall-QPS on all queries and on each arm's OWN hardest decile.

The lab reports both, and they answer different questions. A fixed stratum asks
"on these queries, who is faster"; an own decile asks "where each system is
weakest, how weak is it". The second cannot be shared between arms by
construction -- it is defined by where *that* system does badly -- so each curve
here is scored against its own 10%, exactly as `build_final_folder.py` describes.

The stratum is built the way `run_cell.py:hard_decile` builds it: mean recall@10
over that system's own effort sweep, lowest 10%. The effort sweep is the same set
of runs the curve is drawn from, so no extra measurement is involved and no
index outside the comparison is consulted.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np


def own_decile(perq, frac=0.10):
    """Indices of the queries this system does worst on, averaged over its sweep."""
    mean_recall = perq.mean(axis=0)
    n = max(1, int(round(frac * mean_recall.size)))
    order = np.lexsort((np.arange(mean_recall.size), mean_recall))
    return np.sort(order[:n]), mean_recall


def load_ours(pattern, perq_tmpl):
    rows = []
    for f in sorted(glob.glob(pattern)):
        j = json.loads(Path(f).read_text()); r = j["rows"][0]
        z = np.load(perq_tmpl.format(m=r["m"]))
        rows.append(dict(label=f"m={r['m']}", recall=r["recall"], qps=r["qps"],
                         perq=z["perq"][0]))
    return sorted(rows, key=lambda d: d["recall"])


def load_native(path, perq_path, algo="single_cta"):
    j = json.loads(Path(path).read_text())
    z = np.load(perq_path)["perq"]
    rows = [r for r in j["rows"] if r["algo"] == algo]
    out = [dict(label=f"itopk={r['itopk']}", recall=r["recall"], qps=r["qps"], perq=z[i])
           for i, r in enumerate(rows)]
    return sorted(out, key=lambda d: d["recall"])


def frontier(pts):
    out, best = [], -1.0
    for q, r in sorted(pts, key=lambda p: -p[0]):
        if r > best:
            out.append((q, r)); best = r
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", action="append", required=True,
                    help="name:ours_glob:ours_perq_template:cagra_json:cagra_perq")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--out", default="results/final/fig_own_decile")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(13)

    panels = [p.split(":") for p in a.panel]
    fig, axes = plt.subplots(len(panels), 2,
                             figsize=(2 * a.width + 0.6,
                                      len(panels) * H.panel_h(a.width) + 0.6),
                             squeeze=False)
    for row, (name, og, ot, cj, cp) in enumerate(panels):
        ours, nat = load_ours(og, ot), load_native(cj, cp)
        for col in (0, 1):
            ax = axes[row][col]
            for rows, label, colour, mk in ((nat, "CAGRA", H.RED, "s"),
                                            (ours, "SWANN", H.BLUE, "o")):
                P = np.array([r["perq"] for r in rows])
                if col == 0:
                    ys = [r["recall"] for r in rows]
                else:
                    hard, _ = own_decile(P)
                    ys = [float(r["perq"][hard].mean()) for r in rows]
                pts = frontier(list(zip([r["qps"] for r in rows], ys)))
                ax.plot([p[0] for p in pts], [p[1] for p in pts], marker=mk,
                        color=colour, lw=H.LW, ms=H.ms(H.MS[mk]), mew=H.mew(mk),
                        label=label, zorder=3)
            ax.set_xscale("log")
            ax.set_xlabel("QPS" if row == len(panels) - 1 else "")
            ax.set_ylabel(f"{name}\nrecall@10" if col == 0 else "recall@10")
            if row == 0:
                ax.set_title("all queries" if col == 0 else "own hardest 10%",
                             fontsize=11)
            ax.set_xticks([t for t in (50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000)
                           if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]])
            ax.set_xticks([], minor=True)
            ax.set_xticklabels([f"{int(t/1000)}k" if t < 1e6 else f"{t/1e6:g}M"
                                for t in ax.get_xticks()])
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               frameon=True, facecolor=H.PANEL, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); print(f"wrote {p}")

    # the numbers behind the picture
    for name, og, ot, cj, cp in panels:
        print(f"\n=== {name} ===")
        for rows, label in ((load_native(cj, cp), "CAGRA"), (load_ours(og, ot), "ours")):
            P = np.array([r["perq"] for r in rows])
            hard, mr = own_decile(P)
            print(f"{label}: own decile = {hard.size} queries, sweep mean recall "
                  f"{mr[hard].mean():.4f} against {mr.mean():.4f} overall")
            for r in rows:
                print(f"   {r['label']:12s} all {r['recall']:.4f}  "
                      f"own-hard {float(r['perq'][hard].mean()):.4f}  "
                      f"{r['qps']:9.0f} QPS")


if __name__ == "__main__":
    main()
