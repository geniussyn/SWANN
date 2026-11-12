"""Every number the paper quotes from the frontier, recomputed from one root.

The paper states ratios per corpus, so a re-measurement has to be re-read the
same way rather than spot-checked. One rule throughout: a system is compared
only against a configuration of its own precision, and a system that never
reaches a target has no entry -- which is a statement, not a missing value.
"""
import argparse, glob, json
from pathlib import Path

CORPORA = [("SIFT 1M", "sift-128", 128), ("GloVe 1.18M", "glove-100", 100),
           ("NYTimes 290K", "nytimes-256", 256), ("GIST 1M", "gist-960", 960),
           ("MS MARCO 1M", "msmarco-1m", 1024), ("FEVER 1M", "fever-1m", 1024),
           ("OpenAI-1536 1M", "openai-1536", 1536), ("Qwen-2560 100K", "qwen-2560", 2560),
           ("OpenAI-3072 1M", "openai-3072", 3072), ("Qwen-4096 100K", "qwen-4096", 4096)]


def pts(root, pats, algo=None):
    files = []
    for p in pats:
        if p.startswith("!"):
            drop = set(glob.glob(f"{root}/{p[1:]}"))
            files = [f for f in files if f not in drop]
        else:
            files += glob.glob(f"{root}/{p}")
    out = []
    for f in sorted(set(files)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            if r.get("recall") is None or r.get("qps") is None:
                continue
            if algo and not str(r.get("algo", "")).startswith(algo):
                continue
            out.append((float(r["recall"]), float(r["qps"])))
    return out


def best(points, target):
    ok = [q for r, q in points if r >= target]
    return max(ok) if ok else None


def recall_at_cost(points, qps):
    ok = [r for r, q in points if q >= qps]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean")
    a = ap.parse_args()
    r = a.root
    S = {}
    for title, ds, d in CORPORA:
        S[ds] = dict(
            ours32=pts(r, [f"{ds}_swann_e*.json", f"{ds}_swann_B16_e*.json",
                           f"!{ds}_swann_e*_h.json", f"!{ds}_swann_B16_e*_h.json"]),
            ours16=pts(r, [f"{ds}_swann_e*_h.json", f"{ds}_swann_B16_e*_h.json"]),
            cagra32=pts(r, [f"{ds}_cagra.json"]),
            cagra16=pts(r, [f"{ds}_cagra_h.json"]),
            rabitq=pts(r, [f"{ds}_rabitq_b*.json"]),
            ivfflat=pts(r, [f"{ds}_ivf.json"], algo="ivf_flat"),
            ivfpqref=pts(r, [f"{ds}_ivf.json"], algo="ivf_pq (w/ref"),
        )

    print("== how many corpora each system reaches a target on (of 10)")
    for t in (0.99, 0.999):
        line = []
        for k in ("ours32", "ours16", "cagra32", "cagra16", "rabitq",
                  "ivfflat", "ivfpqref"):
            n = sum(1 for _, ds, _ in CORPORA if best(S[ds][k], t))
            line.append(f"{k} {n}")
        print(f"  >={t}: " + "  ".join(line))

    print("\n== ratio ours/baseline at matched precision")
    print(f"  {'corpus':16}{'d':>5}  {'fp32 vs CAGRA':>26}  {'fp16 vs CAGRA':>26}")
    print(f"  {'':16}{'':>5}  {'@.95':>12}{'@.99':>13}  {'@.95':>12}{'@.99':>13}")
    for title, ds, d in CORPORA:
        s = S[ds]
        cells = ""
        for o, c in (("ours32", "cagra32"), ("ours16", "cagra16")):
            for t in (0.95, 0.99):
                qo, qc = best(s[o], t), best(s[c], t)
                cells += f"{qo/qc:12.2f}x" if qo and qc else f"{'--':>13}"
        print(f"  {title:16}{d:>5}  {cells}")

    print("\n== max recall reached (where a target is missed, this says why)")
    KEYS = ("ours32","ours16","cagra32","cagra16","rabitq","ivfflat","ivfpqref")
    print(f"  {'corpus':16}" + "".join(f"{k:>10}" for k in KEYS))
    for title, ds, d in CORPORA:
        s = S[ds]
        print(f"  {title:16}" + "".join(
            f"{max((x for x, _ in s[k]), default=0):>10.4f}" for k in KEYS))

    print("\n== half precision: does it help each system? (fp16 QPS / fp32 QPS at a target)")
    for t in (0.95, 0.99):
        print(f"  target {t}")
        for title, ds, d in CORPORA:
            s = S[ds]
            o3, o1 = best(s["ours32"], t), best(s["ours16"], t)
            c3, c1 = best(s["cagra32"], t), best(s["cagra16"], t)
            f = lambda x, y: f"{y/x:5.2f}x" if x and y else "   -- "
            print(f"    {title:16} ours {f(o3,o1)}   cagra {f(c3,c1)}")

    print("\n== equal cost: held to the QPS IVF-RaBitQ needs for recall 0.90")
    for title, ds, d in CORPORA:
        s = S[ds]
        q = best(s["rabitq"], 0.90)
        if not q:
            print(f"  {title:16} RaBitQ never reaches 0.90")
            continue
        rb = max((x for x, y in s["rabitq"] if y >= q), default=None)
        print(f"  {title:16} at {q:9.0f} QPS  RaBitQ {rb:.4f}  "
              f"ours fp32 {recall_at_cost(s['ours32'], q) or float('nan'):.4f}  "
              f"ours fp16 {recall_at_cost(s['ours16'], q) or float('nan'):.4f}")

    print("\n== ratio ours/IVF-RaBitQ (quantised: no precision pairing)")
    for t in (0.95, 0.99):
        print(f"  target {t}")
        for title, ds, d in CORPORA:
            s = S[ds]
            qr = best(s["rabitq"], t)
            for k in ("ours32", "ours16"):
                qo = best(s[k], t)
                v = f"{qo/qr:6.2f}x" if qo and qr else "    --"
                print(f"    {title:16} {k} {v}", end="")
            print()


if __name__ == "__main__":
    main()
