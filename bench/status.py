"""Where the overnight run got to, in one screen.

Reads the logs and the result files rather than any bookkeeping of its own, so
it is honest about what actually exists on disk.
"""
import glob, json
from pathlib import Path

STAGES = [("grid sweep (contaminated, kept as a record)", "results/dense.log", "DENSE DONE"),
          ("CAGRA", "results/cagra.log", "CAGRA DONE"),
          ("IVF / RaBitQ modes / batch scaling", "results/rest.log", "REST DONE"),
          ("GGNN and HNSW", "results/ggnnhnsw.log", "GGNN HNSW DONE"),
          ("clean pass, repeats=5, idle device", "results/clean.log", "CLEAN DONE"),
          ("figures", "results/finish.log", "FIGURES DONE")]

print(f"{'stage':44} {'state':>10}")
for name, log, marker in STAGES:
    p = Path(log)
    if not p.exists():
        state = "not started"
    else:
        txt = p.read_text()
        state = "done" if marker in txt else ("running" if txt.strip() else "waiting")
    print(f"{name:44} {state:>10}")

print(f"\n{'corpus':14} {'swann':>7} {'rabitq':>7} {'ivf':>5} {'cagra':>6} "
      f"{'ggnn':>5} {'hnsw':>5}")
for ds in ("sift-128", "nytimes-256", "fever-1m", "arxiv-4096", "msmarco-1m",
           "gist-960", "glove-100", "qwen-1024", "qwen-2560", "qwen-4096",
           "openai-1536", "openai-3072"):
    def n(pat):
        t = 0
        for f in glob.glob(f"results/clean/{ds}_{pat}"):
            try:
                t += len(json.loads(Path(f).read_text()).get("rows", []))
            except Exception:
                pass
        return t
    row = (n("swann_*.json"), n("rabitq_*.json"), n("ivf.json"),
           n("cagra.json"), n("ggnn_*.json"), n("hnsw.json"))
    if any(row):
        print(f"{ds:14} " + " ".join(f"{v:>7}" if i == 0 else f"{v:>6}"
                                     for i, v in enumerate(row)))
