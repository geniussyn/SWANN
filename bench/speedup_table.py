"""Speed-up at a target recall, which is the comparison a reader actually makes.

A recall/QPS curve answers "how fast at this accuracy", so the systems are
compared at fixed accuracy: for each target, take the fastest setting of each
system that still reaches it, and divide. A system whose curve never reaches the
target has no entry -- that is a stronger statement than a slow one, and it is
reported as such rather than filled in with its best effort.
"""
import argparse, glob, json
from pathlib import Path


def pts(spec, algo=None):
    out, files = [], []
    for part in spec.split(","):
        part = part.strip()
        if part.startswith("!"):
            drop = set(glob.glob(part[1:]))
            files = [f for f in files if f not in drop]
        else:
            files += glob.glob(part)
    for f in sorted(set(files)):
        try:
            j = json.loads(Path(f).read_text())
        except Exception:
            continue
        for r in j.get("rows", []):
            name = r.get("algo")
            if algo:
                if algo == "__refine__":
                    if not (name or "").startswith("ivf_pq (w/ref"):
                        continue
                elif name != algo:
                    continue
            if r.get("recall") is None or r.get("qps") is None:
                continue
            out.append((float(r["recall"]), float(r["qps"])))
    return out


def best_at(points, target):
    """Fastest setting that still reaches the target recall."""
    ok = [q for r, q in points if r >= target]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.90, 0.95, 0.99, 0.999])
    ap.add_argument("--ours", default="fp16", choices=["fp16", "fp32"])
    a = ap.parse_args()

    r = a.root
    CORPORA = [("SIFT 1M", "sift-128"), ("GloVe 1.18M", "glove-100"),
               ("NYTimes 290K", "nytimes-256"), ("GIST 1M", "gist-960"),
               ("MS MARCO 1M", "msmarco-1m"), ("FEVER 1M", "fever-1m"),
               ("OpenAI-1536 1M", "openai-1536"), ("Qwen-2560 100K", "qwen-2560"),
               ("OpenAI-3072 1M", "openai-3072"), ("Qwen-4096 100K", "qwen-4096")]

    def ours(ds):
        if a.ours == "fp16":
            return pts(f"{r}/{ds}_swann_e*_h.json,{r}/{ds}_swann_B16_e*_h.json")
        return pts(f"{r}/{ds}_swann_e*.json,{r}/{ds}_swann_B16_e*.json,"
                   f"!{r}/{ds}_swann_e*_h.json,!{r}/{ds}_swann_B16_e*_h.json")

    def rabitq(ds):
        return pts(f"{r}/{ds}_rabitq_b*.json")

    def cagra(ds):
        return pts(f"{r}/{ds}_cagra.json")

    print(f"SWANN {a.ours} speed-up at a target recall "
          f"(blank = that system never reaches the target)\n")
    head = "".join(f"{'  RaBitQ':>9}{'   CAGRA':>9}" for _ in a.targets)
    print(f"{'corpus':18} " + "".join(f"{t:^18}" for t in a.targets))
    print(f"{'':18} " + head)
    agg = {t: {"rab": [], "cag": []} for t in a.targets}
    for title, ds in CORPORA:
        o, rb, cg = ours(ds), rabitq(ds), cagra(ds)
        if not o:
            continue
        line = f"{title:18} "
        for t in a.targets:
            qo, qr, qc = best_at(o, t), best_at(rb, t), best_at(cg, t)
            for key, qb in (("rab", qr), ("cag", qc)):
                if qo and qb:
                    agg[t][key].append(qo / qb)
            line += (f"{qo/qr:8.2f}x" if qo and qr else f"{'--':>9}")
            line += (f"{qo/qc:8.2f}x" if qo and qc else f"{'--':>9}")
        print(line)
    print()
    for t in a.targets:
        for key, name in (("rab", "IVF-RaBitQ"), ("cag", "CAGRA")):
            v = agg[t][key]
            if v:
                v = sorted(v)
                med = v[len(v) // 2] if len(v) % 2 else (v[len(v)//2-1]+v[len(v)//2])/2
                print(f"recall>={t:<6} vs {name:11} median {med:5.2f}x  "
                      f"min {min(v):5.2f}x  max {max(v):6.2f}x  "
                      f"({len(v)} corpora)")


if __name__ == "__main__":
    main()
