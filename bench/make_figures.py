"""Regenerate every figure from whatever has been measured so far.

Runs at any point in a measurement campaign: a corpus with no results yet is
skipped rather than failing, so the same command gives a partial sheet early and
the full one at the end. Nothing here measures anything -- it only reads what
the benches wrote.
"""
import argparse, glob, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# corpus key -> (panel title, data dir, swann results glob, cagra tag)
CORPORA = [
    ("msmarco", "MS MARCO 1M (d=1024)", "/dev/shm/gpuann/msmarco-1m",
     "results/sweep1m/*.json", "m1m"),
    ("fever",   "FEVER 1M (d=1024)",    "/dev/shm/gpuann/fever-1m",
     "results/sweepfever/*.json", "fever1m"),
    ("glove",   "GloVe 1.18M (d=100)",  "/dev/shm/gpuann/glove-100",
     "results/sweepglove/*.json", "glove1m"),
    ("gist",    "GIST 1M (d=960)",      "/dev/shm/gpuann/gist-960",
     "results/sweepgist/*.json", "gist1m"),
    ("qwen4096", "Qwen 100K (d=4096)",  "/dev/shm/gpuann/qwen-4096",
     "results/sweepqwen4096/*.json", "qwen4096"),
    ("qwen2560", "Qwen 100K (d=2560)",  "/dev/shm/gpuann/qwen-2560",
     "results/sweepqwen2560/*.json", "qwen2560"),
    ("qwen1024", "Qwen 100K (d=1024)",  "/dev/shm/gpuann/qwen-1024",
     "results/sweepqwen1024/*.json", "qwen1024"),
]

# per corpus: ggnn glob, rabitq glob. Missing files are skipped, so this can
# list everything that is planned and fill in as it lands.
EXTRA = {
    "msmarco":  ("results/final/m1m_ggnn_k*.json",      "results/final/m1m_rabitq.json"),
    "fever":    ("results/final/fever1m_ggnn_k*.json",  "results/final/fever1m_rabitq.json"),
    "glove":    ("results/final/glove1m_ggnn_k*.json",  "results/final/glove1m_rabitq.json"),
    "gist":     ("results/final/gist1m_ggnn_k*.json",   ""),
    "qwen4096": ("results/final/qwen4096_ggnn_k*.json", "results/final/qwen4096_rabitq.json"),
    "qwen2560": ("results/final/qwen2560_ggnn_k*.json", "results/final/qwen2560_rabitq.json"),
    "qwen1024": ("results/final/qwen1024_ggnn_k*.json", "results/final/qwen1024_rabitq.json"),
}


def merge_cagra(tag):
    """One frontier per corpus out of its per-degree sweeps."""
    rows = []
    for f in sorted(glob.glob(str(ROOT / f"results/final/{tag}_cagra_d*.json"))):
        j = json.loads(Path(f).read_text())
        rows += [r for r in j["rows"] if r.get("algo") == "single_cta"]
    if not rows:
        return None
    base = json.loads(Path(sorted(glob.glob(str(
        ROOT / f"results/final/{tag}_cagra_d*.json")))[0]).read_text())
    base["rows"] = sorted(rows, key=lambda r: r["recall"])
    out = ROOT / f"results/final/{tag}_cagra_best.json"
    out.write_text(json.dumps(base, indent=2))
    return out


def have(spec):
    return bool(glob.glob(str(ROOT / spec))) if not spec.startswith("/") else bool(glob.glob(spec))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ymin", type=float, default=0.88)
    a = ap.parse_args()

    panels, rows_build, ready = [], [], []
    for key, title, data, swann, tag in CORPORA:
        if not have(swann):
            continue
        cag = merge_cagra(tag)
        ggnn, rabitq = EXTRA.get(key, ("", ""))
        panels.append(":".join([title, swann, str(cag.relative_to(ROOT)) if cag else "",
                                ggnn, rabitq]))
        ready.append(title)
        if glob.glob(f"{data}/G_d*.json"):
            rows_build.append(f"{title}:{data}")

    if not panels:
        print("nothing measured yet")
        return
    print("panels:", ", ".join(ready))

    cols = 4 if len(panels) > 4 else max(1, len(panels))
    cmd = [sys.executable, str(ROOT / "bench/fig_grid.py"), "--cols", str(cols),
           "--ymin", str(a.ymin), "--width", "3.1",
           "--out", "results/final/fig_grid"]
    for p in panels:
        cmd += ["--panel", p]
    subprocess.run(cmd, cwd=ROOT, check=True)

    if rows_build:
        cmd = [sys.executable, str(ROOT / "bench/fig_build.py"),
               "--out", "results/final/fig_build"]
        for r in rows_build:
            cmd += ["--row", r]
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
