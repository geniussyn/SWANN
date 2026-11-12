"""Speed-up against each baseline separately, per corpus and per recall target.

Averaging over baselines hides which one is close, and averaging over corpora
hides that the factor is a function of dimension. So nothing is averaged: each
cell is the fastest setting of ours that reaches the target divided by the
fastest setting of that baseline that reaches it, and a baseline that never
reaches the target leaves the cell empty rather than filled with its best try.

Precision is paired where a baseline has a precision: CAGRA is compared fp32 to
fp32 and fp16 to fp16. The quantised families have their own bit widths instead,
and are swept over them.
"""
import argparse, glob, json
from pathlib import Path

CORPORA = [("SIFT 1M", "sift-128"), ("GloVe 1.18M", "glove-100"),
           ("NYTimes 290K", "nytimes-256"), ("GIST 1M", "gist-960"),
           ("MS MARCO 1M", "msmarco-1m"), ("FEVER 1M", "fever-1m"),
           ("OpenAI-1536 1M", "openai-1536"), ("Qwen-2560 100K", "qwen-2560"),
           ("OpenAI-3072 1M", "openai-3072"), ("Qwen-4096 100K", "qwen-4096")]


def rows(pats, half=None, algo=None):
    files = []
    for p in pats:
        if p.startswith("!"):
            drop = set(glob.glob(p[1:]))
            files = [f for f in files if f not in drop]
        else:
            files += glob.glob(p)
    if half is not None:
        files = [f for f in files if f.endswith("_h.json") == half]
    out = []
    for f in sorted(set(files)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            if r.get("recall") is None or not r.get("qps"):
                continue
            name = r.get("algo") or ""
            if algo and not name.startswith(algo):
                continue
            out.append((float(r["recall"]), float(r["qps"])))
    return out


def best(points, target):
    ok = [q for r, q in points if r >= target]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean", help="ours, CAGRA, GGNN, hnswlib")
    ap.add_argument("--baseline-root", default=None,
                    help="where the partitioned family comes from, if not --root")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.90, 0.95, 0.97, 0.99, 0.999])
    ap.add_argument("--ours", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()
    r, b = a.root, a.baseline_root or a.root

    def series(ds):
        return {
            "CAGRA": rows([f"{r}/{ds}_cagra_h.json"] if a.ours == "fp16"
                          else [f"{r}/{ds}_cagra.json"]),
            "IVF-RaBitQ": rows([f"{b}/{ds}_rabitq_b*.json"]),
            "IVF-PQ w/ref": rows([f"{b}/{ds}_ivf.json"], algo="ivf_pq (w/ref"),
            "IVF-PQ": rows([f"{b}/{ds}_ivf.json"], algo="ivf_pq (wo/ref"),
            "IVF-FLAT": rows([f"{b}/{ds}_ivf.json"], algo="ivf_flat"),
            "GGNN": rows([f"{r}/{ds}_ggnn_k*.json"]),
            "HNSW (CPU)": rows([f"{r}/{ds}_hnsw.json"]),
        }

    names = ["CAGRA", "IVF-RaBitQ", "IVF-PQ w/ref", "IVF-PQ", "IVF-FLAT",
             "GGNN", "HNSW (CPU)"]
    print(f"SWANN {a.ours}: speed-up at a fixed recall, per corpus and baseline")
    print(f"ours from {r}, partitioned family from {b}\n")
    head = f"{'corpus':15}{'target':>7}{'ours QPS':>11}" + "".join(f"{n:>14}" for n in names)
    print(head)
    print("-" * len(head))
    for title, ds in CORPORA:
        ours = rows([f"{r}/{ds}_swann_e*.json", f"{r}/{ds}_swann_B16_e*.json"],
                    half=(a.ours == "fp16"))
        if not ours:
            continue
        S = series(ds)
        for t in a.targets:
            qo = best(ours, t)
            if qo is None:
                print(f"{title:15}{t:>7}{'--':>11}" +
                      "".join(f"{'':>14}" for _ in names))
                continue
            line = f"{title:15}{t:>7}{qo:>11,.0f}"
            for n in names:
                qb = best(S[n], t)
                line += f"{qo/qb:>13.2f}x" if qb else f"{'--':>14}"
            print(line)
        print()


if __name__ == "__main__":
    main()
