"""What a kernel change did to the frontier, corpus by corpus.

Two result roots that differ only in the kernel: the same sweep, the same
baselines, the same device. Reported two ways, because they answer different
objections -- the same settings compared directly says whether anything got
slower, and the frontier at a fixed recall says whether it matters.
"""
import argparse, glob, json, statistics as st
from pathlib import Path

CORPORA = [("SIFT 1M", "sift-128"), ("GloVe 1.18M", "glove-100"),
           ("NYTimes 290K", "nytimes-256"), ("GIST 1M", "gist-960"),
           ("MS MARCO 1M", "msmarco-1m"), ("FEVER 1M", "fever-1m"),
           ("OpenAI-1536 1M", "openai-1536"), ("Qwen-2560 100K", "qwen-2560"),
           ("OpenAI-3072 1M", "openai-3072"), ("Qwen-4096 100K", "qwen-4096")]


def pts(root, ds, half, pats=("{ds}_swann_e*.json", "{ds}_swann_B16_e*.json")):
    files = []
    for p in pats:
        files += glob.glob(f"results/{root}/" + p.format(ds=ds))
    files = [f for f in files if f.endswith("_h.json") == half]
    out = []
    for f in sorted(set(files)):
        for r in json.loads(Path(f).read_text()).get("rows", []):
            if r.get("recall") is not None and r.get("qps"):
                out.append((r["recall"], r["qps"], r.get("E"), r.get("stop_eps")))
    return out


def best(P, t):
    ok = [q for r, q, *_ in P if r >= t]
    return max(ok) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", default="clean")
    ap.add_argument("--after", default="clean_ep")
    ap.add_argument("--baselines", default="clean",
                    help="root the baselines come from; they are unaffected")
    ap.add_argument("--half", action="store_true", default=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.95, 0.99, 0.995])
    a = ap.parse_args()

    print(f"{'corpus':15}{'settings':>9}{'median':>9}{'worst':>8}{'best':>8}"
          + "".join(f"{f'  vs CAGRA @{t}':>19}" for t in a.targets))
    for title, ds in CORPORA:
        o, n = pts(a.before, ds, True), pts(a.after, ds, True)
        if not n:
            print(f"{title:15}{'(pending)':>9}")
            continue
        ob = {(x[2], x[3]): x[1] for x in o}
        nb = {(x[2], x[3]): x[1] for x in n}
        both = sorted(set(ob) & set(nb))
        rat = sorted(nb[k] / ob[k] for k in both)
        cag = [(r, q) for r, q, *_ in
               [(x["recall"], x["qps"]) + () for x in
                json.loads(Path(f"results/{a.baselines}/{ds}_cagra_h.json").read_text())["rows"]
                if x.get("recall") is not None and x.get("qps")]]
        cells = ""
        for t in a.targets:
            qo, qn = best(o, t), best(n, t)
            qc = max([q for r, q in cag if r >= t], default=None)
            cells += (f"{f'{qo/qc:.2f}x -> {qn/qc:.2f}x':>19}" if (qo and qn and qc)
                      else f"{'--':>19}")
        print(f"{title:15}{len(both):>9}{st.median(rat):>8.3f}x{rat[0]:>7.3f}x"
              f"{rat[-1]:>7.3f}x{cells}")


if __name__ == "__main__":
    main()
