"""Build cost, as a table you can read the colour of.

Cells are shaded with the lab's sequential ramp -- the one the fragmentation
heatmaps use, where hot means worse -- so the shape of the cost is visible
before any number is read. The scale is per column, because a build second and
a query microsecond do not belong on one scale.

SWANN's stages are broken out because the story is in them: the prune and the
reverse pass are seconds, and the exact candidate sweep is everything else.
"""
import argparse, glob, json, sys
from pathlib import Path

import numpy as np


def swann_build(data_dir):
    """Median over the degrees built for this corpus, plus the stage split."""
    rows = []
    for f in sorted(glob.glob(str(Path(data_dir) / "G_d*.json"))):
        j = json.loads(Path(f).read_text())
        st = j.get("stage_seconds", {})
        rows.append((j["seconds"], st.get("candidates", 0.0),
                     st.get("prune", 0.0), st.get("reverse", 0.0)))
    if not rows:
        return None
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--row", action="append", required=True,
                    help="label:data_dir  (CAGRA times come from cagra_build.json there)")
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--out", default="results/final/fig_build")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    H.setup(12)
    ramp = H.ramp()

    labels, cells = [], []
    for spec in a.row:
        label, d = spec.split(":", 1)
        sw = swann_build(d)
        cg = None
        p = Path(d) / "cagra_build.json"
        if p.exists():
            cg = float(np.median([v for v in json.loads(p.read_text()).values()]))
        if sw is None:
            continue
        labels.append(label)
        cells.append([sw[0], sw[1], sw[2] + sw[3], cg if cg else np.nan,
                      (sw[0] / cg) if cg else np.nan])

    cols = ["SWANN\ntotal (s)", "of which\ncandidates (s)", "prune +\nreverse (s)",
            "CAGRA\ntotal (s)", "SWANN / CAGRA"]
    M = np.array(cells, dtype=float)
    fig, ax = plt.subplots(figsize=(1.55 * len(cols) + 1.6, 0.52 * len(labels) + 1.5))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlim(-0.5, len(cols) - 0.5); ax.set_ylim(len(labels) - 0.5, -0.5)
    ax.grid(False)
    ax.set_facecolor("white")
    for j in range(len(cols)):
        col = M[:, j]
        fin = col[np.isfinite(col)]
        lo, hi = (fin.min(), fin.max()) if fin.size else (0, 1)
        for i in range(len(labels)):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "--", ha="center", va="center", fontsize=9, color=H.INK)
                continue
            t = 0.0 if hi == lo else (v - lo) / (hi - lo)
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=ramp(0.12 + 0.78 * t), edgecolor="white",
                                       lw=1.5, zorder=0))
            txt = f"{v:.1f}" if j < 4 else f"{v:.1f}x"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9.5,
                    color="white" if t > 0.55 else H.INK, zorder=2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Index build cost, one GPU", fontsize=11, pad=10)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        p = Path(f"{a.out}.{ext}"); p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p); print(f"wrote {p}")


if __name__ == "__main__":
    main()
