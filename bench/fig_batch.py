"""Batch-size scaling: throughput, latency, traffic and arithmetic on one sheet.

Columns are (corpus, target recall) -- every system in a column is solving a job
of the same difficulty, with the knobs taken from the measured frontier, so the
curves differ in how they scale and not in what they are answering.

Rows:
  QPS            measured
  latency        measured, the batch's own wall clock; what a caller waits
  GB/s           bytes moved per second
  bytes/query    the traffic one query costs, which does not depend on the batch
                 for any of these designs and is drawn to show exactly that
  FLOP/query     the arithmetic one query costs

The last three are counted for SWANN, whose kernel tallies its own distance
evaluations, and modelled from the index structure for the IVF family, which
exposes no counter. CAGRA chooses its iteration count internally and cannot be
counted from outside, so it appears in the measured rows only -- a gap rather
than a guess.
"""
import argparse, glob, json, sys
from pathlib import Path

CAT = {
    "SWANN fp32":           "#6A3D9A",
    "SWANN fp16":           "#A97FD9",
    "CAGRA":                "#3B7DD8",
    "IVF-RaBitQ (LUT)":     "#E8B018",
    "IVF-RaBitQ (Bitwise)": "#D4682A",
    "IVF-PQ (w/ref)":       "#B4232B",
    "IVF-PQ (wo/ref)":      "#EE8C84",
    "IVF-FLAT":             "#17A2A2",
}
MARK = {"SWANN fp32": "o", "SWANN fp16": "v", "CAGRA": "s",
        "IVF-RaBitQ (LUT)": "D", "IVF-RaBitQ (Bitwise)": "d",
        "IVF-PQ (w/ref)": "^", "IVF-PQ (wo/ref)": "P", "IVF-FLAT": "X"}
ORDER = list(CAT)

ROWS = [("qps", "QPS", True),
        ("latency_ms", "Latency (ms)", True),
        ("gb_per_s", "traffic (GB/s)", True),
        ("bytes_per_query", "bytes / query", True),
        ("flops_per_query", "FLOP / query", True),
        ("gflop_per_s", "compute (GFLOP/s)", True)]

# Measured on this device rather than taken from a specification: a streaming
# read sustains 1523 GB/s and a large fp32 matmul 80.3 TFLOP/s, both with the
# same clocks and thermals as the search runs. Traffic here is what the
# algorithm touches, not what crosses the bus, so a curve above the memory line
# is the cache serving reads the algorithm issued more than once.
PEAK_GB_S = 1523.0
PEAK_GFLOP_S = 80300.0


def derive(r):
    """Compute rate follows from the per-query arithmetic and the measured time;
    recording it separately would only be a second chance to get it wrong."""
    if r.get("flops_per_query") and r.get("latency_ms"):
        r["gflop_per_s"] = (r["flops_per_query"] * r["batch"] / 1e9
                            / (r["latency_ms"] / 1e3))
    return r


def load(root, corpora):
    out = {}
    for ds in corpora:
        rows = []
        for f in glob.glob(f"{root}/{ds}_*.json"):
            if f.endswith("settings.json"):
                continue
            try:
                j = json.loads(Path(f).read_text())
            except Exception:
                continue
            rows += [derive(x) for x in j.get("rows", [])]
        out[ds] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/batch")
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--titles", nargs="+", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--style", default="/workspace/SYN/code")
    ap.add_argument("--width", type=float, default=1.55)
    ap.add_argument("--rows", nargs="+", default=None,
                    help="subset of qps latency_ms gb_per_s bytes_per_query flops_per_query")
    ap.add_argument("--note", default=None,
                    help="caption line drawn under the sheet")
    ap.add_argument("--out", default="results/final/fig_batch")
    a = ap.parse_args()

    sys.path.insert(0, a.style)
    import housestyle as H
    import matplotlib.pyplot as plt
    from matplotlib.ticker import LogLocator, LogFormatterSciNotation, NullFormatter
    H.setup(9)
    plt.rcParams.update({
        "axes.facecolor": "white", "figure.facecolor": "white",
        "axes.edgecolor": "0.25", "axes.linewidth": 0.8,
        "grid.color": "0.80", "grid.linestyle": "--", "grid.linewidth": 0.6,
    })

    rows = [r for r in ROWS if a.rows is None or r[0] in a.rows]
    data = load(a.root, a.corpora)
    cols = [(ds, t, title) for ds, title in zip(a.corpora, a.titles)
            for t in a.targets]
    fig, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                             figsize=(len(cols) * a.width + 0.7,
                                      len(rows) * 1.25 + 0.9))
    seen = {}
    for ci, (ds, target, title) in enumerate(cols):
        for ri, (field, ylab, logy) in enumerate(rows):
            ax = axes[ri][ci]
            for name in ORDER:
                pts = sorted((r["batch"], r.get(field)) for r in data[ds]
                             if r["system"] == name
                             and abs(r["target"] - target) < 1e-9
                             and r.get(field) is not None)
                if not pts:
                    continue
                h, = ax.plot([p[0] for p in pts], [p[1] for p in pts],
                             marker=MARK[name], color=CAT[name], lw=1.1,
                             ms=3.2, mew=0.6, label=name, zorder=3)
                seen.setdefault(name, h)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
                ax.yaxis.set_major_locator(LogLocator(base=10.0))
                ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
                ax.yaxis.set_minor_locator(
                    LogLocator(base=10.0, subs=tuple(range(2, 10))))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if field == "gb_per_s":
                ax.axhline(PEAK_GB_S, color="0.35", lw=0.7, ls=(0, (4, 2)),
                           zorder=2)
            if field == "gflop_per_s":
                ax.axhline(PEAK_GFLOP_S, color="0.35", lw=0.7, ls=(0, (4, 2)),
                           zorder=2)
            ax.grid(True, which="major", alpha=0.9)
            ax.tick_params(labelsize=6.5)
            if ri == 0:
                ax.set_title(f"{title}\nRecall@10={target:g}", fontsize=7)
            if ri == len(rows) - 1:
                ax.set_xlabel("Batch size", fontsize=7.5)
            else:
                ax.set_xticklabels([])
            if ci == 0:
                ax.set_ylabel(ylab, fontsize=7.5)
    order = [k for k in ORDER if k in seen]
    fig.legend([seen[k] for k in order], order, loc="upper center",
               ncol=min(len(order), 8), frameon=True, fontsize=7,
               bbox_to_anchor=(0.5, 1.0))
    if a.note:
        fig.text(0.5, 0.005, a.note, ha="center", fontsize=6.5, color="0.3")
    fig.tight_layout(rect=(0, 0.02 if a.note else 0, 1,
                           0.94 if len(rows) > 2 else 0.88))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", dpi=300, bbox_inches="tight")
    print(f"wrote {out}.pdf and {out}.png")


if __name__ == "__main__":
    main()
