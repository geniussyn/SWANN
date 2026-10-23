"""For each system, the setting that reaches a target recall most cheaply.

A batch-size study has to hold accuracy fixed: the curves are only comparable
if every system is doing a job of the same difficulty. So the knobs are taken
from the measured frontier -- the fastest setting that still reached the target
when the whole grid was swept -- rather than guessed, and written out for the
runners to replay at each batch size.
"""
import argparse, glob, json
from pathlib import Path


def rows(spec, algo=None):
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
            if r.get("recall") is not None and r.get("qps") is not None:
                out.append(r)
    return out


def cheapest(rs, target, arm=None):
    ok = [r for r in rs if r["recall"] >= target
          and (arm is None or r.get("arm") == arm)]
    return max(ok, key=lambda r: r["qps"]) if ok else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/clean")
    ap.add_argument("--corpora", nargs="+", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    r = a.root
    out = {}
    for ds in a.corpora:
        out[ds] = {}
        src = {
            "SWANN fp16": (f"{r}/{ds}_swann_e*_h.json,{r}/{ds}_swann_B16_e*_h.json", None),
            "SWANN fp32": (f"{r}/{ds}_swann_e*.json,{r}/{ds}_swann_B16_e*.json,"
                           f"!{r}/{ds}_swann_e*_h.json,!{r}/{ds}_swann_B16_e*_h.json", None),
            # CAGRA's own graph. The from_graph arm answers a different
            # question (same graph, different engine) and its index cannot be
            # rebuilt in the batch runner, so the batch study uses the arm a
            # CAGRA user would actually deploy.
            "CAGRA": (f"{r}/{ds}_cagra.json", None),
            "IVF-RaBitQ (LUT)": (f"{r}/{ds}_rabitq_b*.json", "IVF-RaBitQ (LUT)"),
            "IVF-RaBitQ (Bitwise)": (f"{r}/{ds}_rabitq_b*.json", "IVF-RaBitQ (Bitwise)"),
            "IVF-FLAT": (f"{r}/{ds}_ivf.json", "ivf_flat"),
            "IVF-PQ (w/ref)": (f"{r}/{ds}_ivf.json", "__refine__"),
            "IVF-PQ (wo/ref)": (f"{r}/{ds}_ivf.json", "ivf_pq (wo/ref)"),
        }
        for sysname, (spec, algo) in src.items():
            rs = rows(spec, algo)
            for t in a.targets:
                c = cheapest(rs, t, arm="built" if sysname == "CAGRA" else None)
                key = f"{t:g}"
                out[ds].setdefault(key, {})
                if c is None:
                    out[ds][key][sysname] = None
                    continue
                keep = {k: c[k] for k in
                        ("V_per_branch", "B", "L", "stop_eps", "dist_per_q",
                         "n_probes", "bits", "lists", "mode", "itopk", "width",
                         "arm", "algo", "refine", "recall", "qps")
                        if k in c}
                out[ds][key][sysname] = keep
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2))
    n = sum(1 for ds in out for t in out[ds] for s in out[ds][t] if out[ds][t][s])
    print(f"wrote {a.out}: {n} settings "
          f"({len(a.corpora)} corpora x {len(a.targets)} targets x 8 systems)")


if __name__ == "__main__":
    main()
