"""A one-step head that predicts the query's answer pieces, and what it buys.

The question and its scale are already measured (docs/SUBSPACE-ENTRIES.md 8, 12):
allocating V entries over the ANSWER-HOLDING basins takes in_mass from 0.560 to
0.950 and recall from 0.9627 to 0.9823 at V=8 (+0.0196, p=2e-47). An entry's own
similarity to the query predicts an answer-bearing basin at AUC 0.554, so the
information has to come from somewhere else; 3b and 11 close the geometry and the
graph, which leaves supervision.

This is the cheapest member of that family, and deliberately so. A diffusion
sampler distils to a feed-forward head for serving anyway, so the head answers
the question first: **is the signal learnable from a query log at all?** The dial
is already calibrated at both ends -- 0.560 for nearest-V, 0.950 for the oracle --
so the result is readable without a baseline sweep.

Design, with the reason for each choice from a measurement rather than taste:

  canonical order, no Hungarian   the pieces have a natural order, by how many of
                                  the k answers they hold -- 71.5% in the largest
                                  and 86.4% in the largest two (8.1). So output j
                                  targets the j-th largest piece and the matching
                                  is fixed. Distinct targets are also what stops
                                  the outputs collapsing onto each other, which is
                                  the failure the multi-interest literature calls
                                  interest collapse and which diffusion exists to
                                  avoid.
  residual, around q              out_j = normalize(q + delta_j). The arm that
                                  ever worked on the geometry side probed
                                  normalize(q +- lambda*axis) (3), so the head
                                  starts where that started and q is the identity
                                  initialisation.
  a presence logit per output     SetMIR's presence score (6). It decides how many
                                  outputs to spend entries on, which is what turns
                                  this into an entry-budget result rather than a
                                  recall result.
  no mass regression              equal and mass-proportional quotas tie at every
                                  V (8.2), so the weight is not the lever and is
                                  not modelled.

Trains on the 59,242 trainpool queries (bench/head_data.py), tunes on val,
reports on test. Collapse is reported whether or not it happens: mean pairwise
cosine between the N outputs, and how many distinct pieces their entries reach.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, select_a                                  # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402
from bench.basin_coverage import climb                               # noqa: E402


class Head(nn.Module):
    """q -> N residual directions and N presence logits."""

    def __init__(self, d, N, hidden):
        super().__init__()
        self.N, self.d = N, d
        self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, N * d)
        self.pres = nn.Linear(hidden, N)
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def forward(self, q):
        h = self.body(q)
        dl = self.delta(h).view(-1, self.N, self.d)
        out = torch.nn.functional.normalize(q.unsqueeze(1) + dl, dim=2)
        return out, self.pres(h)


def piece_targets(X, gt, att, N):
    """[NQ, N, d] centroids of the N largest pieces, and [NQ, N] validity.

    Pieces are ordered by how many answers they hold, ties by the piece's best
    answer rank, so the order is a function of the data and not of the loop.
    """
    NQ, k = gt.shape
    d = X.shape[1]
    tgt = torch.zeros(NQ, N, d, device=X.device)
    val = torch.zeros(NQ, N, dtype=torch.bool, device=X.device)
    av = X[gt.reshape(-1).to(torch.long)].view(NQ, k, d)
    an = att.cpu().numpy()
    for q in range(NQ):
        u, c = np.unique(an[q], return_counts=True)
        first = np.array([np.argmax(an[q] == p) for p in u])
        order = np.lexsort((first, -c))
        for j, pi in enumerate(order[:N]):
            m = torch.from_numpy(an[q] == u[pi]).to(X.device)
            tgt[q, j] = torch.nn.functional.normalize(av[q][m].mean(0), dim=0)
            val[q, j] = True
    return tgt, val


def allocate(out, pres, Vp, V, thresh):
    """V entries over the outputs whose presence clears `thresh`, round robin.

    Output 0 always counts, so the arm degrades to nearest-V rather than to
    nothing when the head says a query has one piece.
    """
    NQ, N, _ = out.shape
    rank = torch.topk(out.reshape(-1, out.shape[2]) @ Vp.T, V, dim=1).indices
    rank = rank.view(NQ, N, V)
    live = torch.sigmoid(pres) >= thresh
    live[:, 0] = True
    sel = torch.full((NQ, V), -1, dtype=torch.long, device=out.device)
    for q in range(NQ):
        js = [j for j in range(N) if live[q, j]]
        got, ptr = [], {j: 0 for j in js}
        while len(got) < V:
            moved = False
            for j in js:
                while ptr[j] < V:
                    c = int(rank[q, j, ptr[j]]); ptr[j] += 1
                    if c not in got:
                        got.append(c); moved = True
                        break
                if len(got) == V:
                    break
            if not moved:
                break
        while len(got) < V:
            got.append(int(rank[q, 0, 0]))
        sel[q] = torch.tensor(got[:V], device=out.device)
    return sel, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--cache", type=Path, default=Path("/dev/shm/gpuann/head"))
    ap.add_argument("--trainq", type=Path,
                    default=Path("/workspace/SYN/emb/"
                                 "msmarco-1m-trainpool__vanilla_queries.npy"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--V", type=int, nargs="+", default=[8])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--total-B", type=int, default=2048)
    ap.add_argument("--stop-eps", type=float, default=0.05)
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--team", type=int, default=16)
    ap.add_argument("--nq-eval", type=int, default=2000)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    G = torch.from_numpy(np.load(a.data / a.graph)).cuda()
    pl = torch.from_numpy(load_pool(a.data, a.pool, a.P).astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pl)
    d = X.shape[1]

    gt_tr = torch.from_numpy(np.load(a.cache / "gt_train.npy")).cuda()
    att_tr = torch.from_numpy(np.load(a.cache / "att_train.npy")).cuda()
    Qtr = torch.from_numpy(np.load(a.trainq).astype(np.float32)).cuda()
    Qtr = torch.nn.functional.normalize(Qtr, dim=1)[: gt_tr.shape[0]]
    print(f"train {Qtr.shape[0]:,} queries   N={a.N} outputs   hidden={a.hidden}")

    t0 = time.perf_counter()
    tgt_tr, val_tr = piece_targets(X, gt_tr, att_tr, a.N)
    print(f"piece targets built in {time.perf_counter()-t0:.1f}s   "
          f"live outputs per query {val_tr.float().sum(1).mean():.2f}\n")

    # the number that makes the training loss readable: what out_j = q scores.
    # The head is initialised there (delta is zero-init), so anything it learns
    # has to come off this, per output index. Output 0 targets the largest piece
    # and is near q by construction; the later ones are the question.
    with torch.no_grad():
        cq = (Qtr.unsqueeze(1) * tgt_tr).sum(2)
        base_all = (((1 - cq) * val_tr).sum() / val_tr.float().sum()).item()
        base_j = [float((((1 - cq[:, j]) * val_tr[:, j]).sum()
                         / val_tr[:, j].float().sum().clamp(min=1)))
                  for j in range(a.N)]
    print("identity baseline (out_j = q), the loss to beat:")
    print(f"  overall {base_all:.4f}   per output " +
          "  ".join(f"j={j}: {b:.4f}" for j, b in enumerate(base_j)))
    print(f"  live fraction per output " +
          "  ".join(f"j={j}: {float(val_tr[:, j].float().mean()):.3f}"
                    for j in range(a.N)) + "\n")

    head = Head(d, a.N, a.hidden).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=a.lr, total_steps=a.epochs * ((Qtr.shape[0] + a.bs - 1) // a.bs))
    bce = nn.BCEWithLogitsLoss()
    for ep in range(a.epochs):
        perm = torch.randperm(Qtr.shape[0], device=X.device)
        tot = n = 0.0
        for s in range(0, Qtr.shape[0], a.bs):
            idx = perm[s:s + a.bs]
            out, pres = head(Qtr[idx])
            v = val_tr[idx]
            cos = (out * tgt_tr[idx]).sum(2)
            lo = ((1 - cos) * v).sum() / v.float().sum().clamp(min=1)
            lp = bce(pres, v.float())
            loss = lo + 0.1 * lp
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.detach().item() * len(idx); n += len(idx)
        if ep % max(1, a.epochs // 5) == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                o, _ = head(Qtr[:4096])
                c = (o * tgt_tr[:4096]).sum(2); v = val_tr[:4096]
                pj = [float((((1 - c[:, j]) * v[:, j]).sum()
                             / v[:, j].float().sum().clamp(min=1)))
                      for j in range(a.N)]
            print(f"  epoch {ep:3d}  loss {tot/n:.4f}   per output " +
                  "  ".join(f"j={j}: {x:.4f}" for j, x in enumerate(pj)))
    print()

    rows = []
    for split in ("val", "test"):
        Q = torch.from_numpy(np.load(a.data / f"Q_{split}.npy")[: a.nq_eval]).cuda()
        gt = np.load(a.data / f"gt_{split}_ids.npy")[: a.nq_eval, : a.k]
        NQ = Q.shape[0]
        gts = [set(int(t) for t in r) for r in gt]
        att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q)
        gs = [set(r.tolist()) for r in att_gt.cpu()]
        head.eval()
        with torch.no_grad():
            out, pres = head(Q)
        pair = (out @ out.transpose(1, 2))
        pair = ((pair.sum((1, 2)) - a.N) / (a.N * (a.N - 1))).mean()
        print(f"=== {split}, {NQ} queries ===")
        print(f"  mean pairwise cosine between the {a.N} outputs: {pair:.4f}  "
              f"(1.0 would be total collapse)")
        print(f"  live outputs per query at thresh {a.thresh}: "
              f"{(torch.sigmoid(pres) >= a.thresh).float().sum(1).mean():.2f}")

        def report(name, sel, V, B):
            ids, _, st = engine.search(X, G, Q, pl[sel].to(torch.int32), k=a.k,
                                       L=a.L, B=B, metric=engine.METRIC_IP,
                                       team_size=a.team, stop_eps=a.stop_eps,
                                       shared_cache=True, shared_bound=True,
                                       exclusive=True, elide_visited=True)
            got = ids.cpu().numpy()
            rec = float(np.mean([len(gts[q] & {int(v) for v in got[q] if v >= 0})
                                 / a.k for q in range(NQ)]))
            at = climb(X, G, pl[sel], Q).cpu().numpy()
            ape = float(np.mean([len(set(at[q].tolist())) / V for q in range(NQ)]))
            imm = float(np.mean([np.mean([x in gs[q] for x in at[q]])
                                 for q in range(NQ)]))
            de = float(st.dist_evals.sum(1).float().mean())
            print(f"  {name:22} V={V:<3d} recall {rec:.4f}  att/entry {ape:.3f}  "
                  f"in_mass {imm:.3f}  dist/q {de:.0f}")
            rows.append(dict(split=split, arm=name, V=V, recall=rec,
                             att_per_entry=ape, in_mass=imm, dist=de,
                             pair_cos=float(pair)))

        # two controls that split "is the head's prediction good enough" from
        # "is aiming at a centroid and taking the nearest pool entry good enough".
        gtt = torch.from_numpy(gt.astype(np.int64)).cuda()
        tgt_ev, val_ev = piece_targets(X, gtt.to(torch.int32), att_gt, a.N)
        pres_ev = torch.where(val_ev, 10.0, -10.0)
        # and the arm 8.2 measured at +0.0196: quotas over the GT attractors,
        # filled with the nearest entries that actually REACH each one.
        nearM = select_a.select_nearest(Q, Vp, 256, engine.METRIC_IP)
        att_en = climb(X, G, pl[nearM], Q).cpu().numpy()
        massq = []
        for q in range(NQ):
            u, c = np.unique(att_gt[q].cpu().numpy(), return_counts=True)
            o = np.argsort(-c); massq.append(list(zip(u[o].tolist(), c[o].tolist())))

        def oracle_attr(V):
            o = np.empty((NQ, V), np.int64)
            for q in range(NQ):
                tot = sum(c for _, c in massq[q])
                raw = [(A, V * c / tot) for A, c in massq[q]]
                qa = {A: int(np.floor(r)) for A, r in raw}
                rem = V - sum(qa.values())
                for i, (A, r) in enumerate(sorted(raw,
                                                  key=lambda t: -(t[1] - np.floor(t[1])))):
                    if i < rem:
                        qa[A] += 1
                got = {A: 0 for A in qa}; pick = []
                for j in range(nearM.shape[1]):
                    A = att_en[q, j]
                    if A in qa and got[A] < qa[A]:
                        got[A] += 1; pick.append(int(nearM[q, j]))
                        if len(pick) == V:
                            break
                for j in range(nearM.shape[1]):
                    if len(pick) == V:
                        break
                    c = int(nearM[q, j])
                    if c not in pick:
                        pick.append(c)
                o[q] = pick[:V]
            return torch.from_numpy(o).cuda()

        for V in a.V:
            B = max(1, a.total_B // V)
            report("nearest-V", select_a.select_nearest(Q, Vp, V,
                                                        engine.METRIC_IP), V, B)
            with torch.no_grad():
                sel, live = allocate(out, pres, Vp, V, a.thresh)
            report("head", sel, V, B)
            with torch.no_grad():
                selo, _ = allocate(tgt_ev, pres_ev, Vp, V, a.thresh)
            report("TRUE centroids", selo, V, B)
            report("TRUE attractors (8.2)", oracle_attr(V), V, B)
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(N=a.N, hidden=a.hidden, epochs=a.epochs,
                                         lr=a.lr, bs=a.bs, thresh=a.thresh,
                                         pool=a.pool, P=a.P, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
