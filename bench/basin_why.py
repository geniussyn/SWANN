"""Why is an entry's basin unpredictable from its position? Data, or graph?

The entry side's one fact (docs/SUBSPACE-ENTRIES.md 8-11): whether an entry's
greedy descent ends where an answer's descent ends is predicted by the entry's
own similarity to the query at AUC 0.54 and by an 8-step walk at 0.9995, and
~40% of the nearest V entries land in a basin holding no answer. This asks what
sets that number, by holding everything fixed but one axis at a time.

  graph       msmarco-1m under DiskANN prune, CAGRA, HNSW and NN-descent. Same
              vectors, same queries, same pool: a difference is the build rule.
  dimension   qwen-1024 / 2560 / 4096 are the SAME 100k documents at three
              encoder widths, so this is dimension with content fixed.
  size        ms-125k / 250k / 500k / msmarco-1m, same encoder and content.
  domain      learned text (msmarco, fever, nytimes), word vectors (glove),
              and hand-built descriptors under L2 (gist, sift).

`pool_random4k` exists for every corpus and is the common pool, so the cells are
comparable to each other even though the paper's arms use `nest4k`.

The number that should explain the rest is `att/entry`: distinct attractors among
the nearest M entries, divided by M. At 1.0 every entry has its own basin and no
function of position can predict which one; near 0 the entries share basins and
position is informative. If that ratio tracks the AUC across every axis, the
answer is the landscape's roughness at the scale the catalogue samples it, and
the AUC 0.54 is a consequence rather than a fact of its own.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


def auc(score, label):
    out = []
    for s, y in zip(score, label):
        p, n = y.sum(), (~y).sum()
        if p == 0 or n == 0:
            continue
        r = np.argsort(np.argsort(s)) + 1
        out.append((r[y].sum() - p * (p + 1) / 2) / (p * n))
    return float(np.mean(out)) if out else float("nan")


def cell(data, graph, pool, P, nq, M, k, Vs, device="cuda"):
    man = json.loads((data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(data / "X.npy")).to(device)
    G = torch.from_numpy(np.load(data / graph)).to(device)
    Q = torch.from_numpy(np.load(data / "Q_val.npy")[:nq]).to(device)
    gt = np.load(data / "gt_val_ids.npy")[:nq, :k]
    NQ = Q.shape[0]
    pl = torch.from_numpy(load_pool(data, pool, P).astype(np.int64)).to(device)
    Vp = select_a.pool_vectors(X, pl)
    metric = engine.METRIC_IP if ip else engine.METRIC_L2

    def score(nodes):                       # uniform "larger is better"
        v = X[nodes.reshape(-1)].view(nodes.shape[0], -1, X.shape[1])
        if ip:
            return (v * Q.unsqueeze(1)).sum(2)
        return -((v - Q.unsqueeze(1)) ** 2).sum(2)

    ent_idx = select_a.select_nearest(Q, Vp, M, metric)
    ent = pl[ent_idx]
    gtt = torch.from_numpy(gt.astype(np.int64)).to(device)
    att_e = climb(X, G, ent, Q, ip=ip)
    att_g = climb(X, G, gtt, Q, ip=ip)

    # how many answers are their own local maximum -- the pure-descent ceiling
    nb = G[gtt.reshape(-1)].long()
    sv = X[nb.clamp(min=0)]
    qv = Q.unsqueeze(1).expand(-1, k, -1).reshape(-1, X.shape[1]).unsqueeze(1)
    sn = (sv * qv).sum(2) if ip else -((sv - qv) ** 2).sum(2)
    sn = torch.where(nb >= 0, sn, torch.full((1,), -1e30, device=X.device))
    here = score(gtt).reshape(-1)
    locmax = float((sn.max(1).values <= here).float().mean())

    ae, ag = att_e.cpu().numpy(), att_g.cpu().numpy()
    gsets = [set(r.tolist()) for r in ag]
    label = np.zeros(ae.shape, bool)
    for q in range(NQ):
        label[q] = np.isin(ae[q], list(gsets[q]))
    att_per_entry = float(np.mean([len(set(ae[q].tolist())) for q in range(NQ)])) / M
    ans_basins = float(np.mean([len(gsets[q]) for q in range(NQ)]))
    s_ent = score(ent).cpu().numpy()
    s_c8 = score(climb(X, G, ent, Q, steps=8, ip=ip)).cpu().numpy()
    out = dict(data=data.name, graph=graph, d=man["d"], n=man["n"],
               metric=man["metric"], locmax=locmax, att_per_entry=att_per_entry,
               ans_basins=ans_basins, auc_prox=auc(s_ent, label),
               auc_climb8=auc(s_c8, label), pos_rate=float(label.mean()))
    for V in Vs:
        out[f"in_mass{V}"] = float(label[:, :V].mean())
    del X, G, Q, Vp, att_e, att_g, sv, sn
    torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/dev/shm/gpuann"))
    ap.add_argument("--nq", type=int, default=600)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 32])
    ap.add_argument("--pool", default="random4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--axis", nargs="+",
                    default=["graph", "dimension", "size", "domain"])
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    axes = {
        "graph": [("msmarco-1m", g) for g in
                  ("G_d64.npy", "G_cagra_d64.npy", "G_hnsw.npy", "G_nnd64.npy",
                   "G_nnd128.npy")],
        "dimension": [(c, "G_d64.npy") for c in
                      ("qwen-1024", "qwen-2560", "qwen-4096")],
        "size": [(c, "G_d64.npy") for c in
                 ("ms-125k", "ms-250k", "ms-500k", "msmarco-1m")],
        "domain": [(c, "G_d64.npy") for c in
                   ("msmarco-1m", "fever-1m", "nytimes-256", "glove-100",
                    "gist-960", "sift-128")],
    }
    hdr = (f"  {'corpus':14} {'graph':16} {'d':>5} {'n':>9} "
           f"{'locmax':>7} {'att/entry':>10} {'ansbasin':>9} "
           + "".join(f"{f'inmass{V}':>10}" for V in a.V)
           + f" {'AUC pos':>8} {'AUC c8':>7}")
    rows = []
    for ax in a.axis:
        print(f"\n=== {ax} ===")
        print(hdr)
        for corpus, graph in axes[ax]:
            d = a.root / corpus
            if not (d / graph).exists() or not (d / "X.npy").exists():
                print(f"  {corpus:14} {graph:16} -- missing, skipped")
                continue
            try:
                r = cell(d, graph, a.pool, a.P, a.nq, a.M, a.k, a.V)
            except Exception as e:                        # noqa: BLE001
                print(f"  {corpus:14} {graph:16} -- {type(e).__name__}: {e}")
                continue
            r["axis"] = ax
            rows.append(r)
            print(f"  {r['data']:14} {r['graph']:16} {r['d']:5d} {r['n']:9,d} "
                  f"{r['locmax']:7.3f} {r['att_per_entry']:10.3f} "
                  f"{r['ans_basins']:9.2f} "
                  + "".join(f"{r[f'in_mass{V}']:10.3f}" for V in a.V)
                  + f" {r['auc_prox']:8.4f} {r['auc_climb8']:7.4f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(nq=a.nq, M=a.M, k=a.k, pool=a.pool,
                                         P=a.P, rows=rows), indent=2))
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
