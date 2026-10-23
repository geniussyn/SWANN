"""One encode at the model's full width, then MRL truncations of it.

The question is what dimension does to the search, so the model has to be held
fixed: encoding three different Qwen sizes would vary capacity and dimension
together and the answer would mean nothing. Qwen3-Embedding is trained with
Matryoshka representation learning, so a prefix of the 4096-dim vector is a
usable 2560- or 1024-dim embedding of the same model -- which is exactly the
axis wanted, at one encode instead of three.
"""
import argparse, json, os, time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def last_token_pool(h, mask):
    """Qwen3-Embedding pools the last non-padding token."""
    left = mask[:, -1].sum() == mask.shape[0]
    if left:
        return h[:, -1]
    idx = mask.sum(dim=1) - 1
    return h[torch.arange(h.shape[0], device=h.device), idx]


@torch.no_grad()
def encode(texts, tok, model, bs, maxlen, device="cuda"):
    out = None
    t0 = time.perf_counter()
    for s in range(0, len(texts), bs):
        batch = texts[s:s + bs]
        enc = tok(batch, padding=True, truncation=True, max_length=maxlen,
                  return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        v = last_token_pool(h, enc["attention_mask"]).float()
        v = torch.nn.functional.normalize(v, dim=1)
        if out is None:
            out = torch.empty((len(texts), v.shape[1]), dtype=torch.float32, device="cpu")
        out[s:s + len(batch)] = v.cpu()
        if (s // bs) % 50 == 0:
            done = s + len(batch)
            rate = done / max(time.perf_counter() - t0, 1e-9)
            print(f"  {done}/{len(texts)}  {rate:.0f}/s", flush=True)
    return out.numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--docs", default="/workspace/SYN/corpora/msmarco-1m_docs.txt")
    ap.add_argument("--queries", default="/workspace/SYN/corpora/msmarco-1m_queries.jsonl")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--maxlen", type=int, default=256)
    ap.add_argument("--dims", type=int, nargs="+", default=[4096, 2560, 1024])
    ap.add_argument("--out", type=Path, default=Path("/data/qwen"))
    a = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/hf")
    docs = []
    with open(a.docs) as f:
        for i, line in enumerate(f):
            if i >= a.n:
                break
            docs.append(line.rstrip("\n"))
    queries = []
    with open(a.queries) as f:
        for line in f:
            o = json.loads(line)
            queries.append(o.get("text") or o.get("query") or o.get("question"))
    print(f"{len(docs)} docs, {len(queries)} queries, model {a.model}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model, padding_side="left")
    model = AutoModel.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()

    D = encode(docs, tok, model, a.bs, a.maxlen)
    # Qwen3-Embedding's query side takes an instruction prefix; documents do not.
    instr = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: "
    Q = encode([instr + q for q in queries], tok, model, a.bs, a.maxlen)
    print(f"encoded: docs {D.shape}, queries {Q.shape}", flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    for d in a.dims:
        sub = a.out / f"d{d}"
        sub.mkdir(exist_ok=True)
        Dd = np.ascontiguousarray(D[:, :d])
        Qd = np.ascontiguousarray(Q[:, :d])
        Dd /= np.maximum(np.linalg.norm(Dd, axis=1, keepdims=True), 1e-12)
        Qd /= np.maximum(np.linalg.norm(Qd, axis=1, keepdims=True), 1e-12)
        np.save(sub / "docs.npy", Dd.astype(np.float32))
        np.save(sub / "queries.npy", Qd.astype(np.float32))
        print(f"  wrote d={d}: {Dd.shape}", flush=True)
    (a.out / "manifest.json").write_text(json.dumps(
        dict(model=a.model, n=len(docs), nq=len(queries), dims=a.dims,
             maxlen=a.maxlen, note="MRL prefixes of one encode, renormalised"), indent=2))


if __name__ == "__main__":
    main()
