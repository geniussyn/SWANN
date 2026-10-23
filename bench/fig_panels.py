"""Panel specs for the recall/QPS sheet, built from what is on disk.

Each system's points live under several filename patterns -- SWANN's grid is
split across one file per (eps, precision) and a second pass at a smaller
budget, and IVF-RaBitQ's across one file per bits-per-dimension -- so the globs
are assembled here rather than typed out on the command line, where getting one
wrong silently drops half a frontier.
"""
import argparse
from pathlib import Path

CORPORA = [
    ("SIFT 1M",        "sift-128",    "dense"),
    ("GloVe 1.18M",    "glove-100",   "dense"),
    ("NYTimes 290K",   "nytimes-256", "dense"),
    ("GIST 1M",        "gist-960",    "dense"),
    ("MS MARCO 1M",    "msmarco-1m",  "dense"),
    ("FEVER 1M",       "fever-1m",    "dense"),
    ("OpenAI-1536 1M", "openai-1536", "dense"),
    ("Qwen-2560 100K", "qwen-2560",   "dense"),
    ("OpenAI-3072 1M", "openai-3072", "dense"),
    ("Qwen-4096 100K", "qwen-4096",   "dense"),
]


def spec(title, ds, root):
    r = f"results/{root}"
    # `*_e*.json` also matches `*_e0.05_h.json`, so the fp32 series would carry
    # the fp16 points as well and the two curves would be the same curve. A
    # leading ! removes what the pattern matches.
    swann = (f"{r}/{ds}_swann_e*.json,{r}/{ds}_swann_B16_e*.json,"
             f"!{r}/{ds}_swann_e*_h.json,!{r}/{ds}_swann_B16_e*_h.json")
    half = f"{r}/{ds}_swann_e*_h.json,{r}/{ds}_swann_B16_e*_h.json"
    cagra = f"{r}/{ds}_cagra.json"
    rab = f"{r}/{ds}_rabitq_b*.json,{r}/{ds}_rabitq_m_b*.json"
    ivf = f"{r}/{ds}_ivf.json"
    ggnn = f"{r}/{ds}_ggnn_k*.json"
    hnsw = f"{r}/{ds}_hnsw.json"
    # one spec per series, in the order the figure expects them
    return ":".join([title, swann, half, cagra, rab, rab, ivf, ivf, ivf,
                     ggnn, hnsw])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="clean")
    ap.add_argument("--only-ready", action="store_true",
                    help="skip a corpus with no SWANN file yet")
    a = ap.parse_args()
    for title, ds, root in CORPORA:
        root = a.root
        if a.only_ready and not list(Path(f"results/{root}").glob(f"{ds}_swann_e*.json")):
            continue
        print(f"--panel\n{spec(title, ds, root)}")


if __name__ == "__main__":
    main()
