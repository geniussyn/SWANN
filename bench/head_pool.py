"""Score the POOL directly: learn which entries reach an answer-bearing basin.

Why this and not the vector-generating head of bench/head_train.py: that route's
two losses are measured (test, V=8, 2000 queries, nest4k):

    nearest-V              recall 0.9658   in_mass 0.655
    head (predicted)       recall 0.9620   in_mass 0.706
    TRUE piece centroids   recall 0.9725   in_mass 0.795
    TRUE attractors (8.2)  recall 0.9830   in_mass 0.986

Aiming at a centroid and taking the nearest pool entry gives up 58% of the prize
with exact centroids, and a cosine-0.876 prediction gives up the rest. The
oracle's target is a **set of entries**, so this predicts that.

Parameterisation, chosen so that the untrained model IS the baseline:

    score_p = <q, x_p>  +  alpha * max_j <out_j, x_p>,    out_j = norm(q + delta_j)

`alpha` is learned and initialised at 0, `delta` is zero-init, so at step 0 this
is exactly the catalogue scan and training can only depart from it deliberately.
The **max** over j is what keeps the proposal alive past Proposition 1 -- an
additive linear term in q is a pool re-placement and nothing more (section 0) --
and it is the same asymmetry ColBERT/MUVERA and RaBitQ use.

Loss is listwise over the M candidates the scan already ranks, toward the
positives of bench/head_pool_data.py: entries whose basin holds one of the
query's answers, which is the quantity the oracle maximises.

Trains on 59,242 trainpool queries, tunes on val, reports on test. The dial is
calibrated at both ends by the table above, so the result reads without a sweep.
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


class PoolHead(nn.Module):
    def __init__(self, d, N, hidden):
        super().__init__()
        self.N, self.d = N, d
        self.body = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden), nn.GELU())
        self.delta = nn.Linear(hidden, N * d)
        self.alpha = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.delta.weight); nn.init.zeros_(self.delta.bias)

    def dirs(self, q):
        h = self.body(q)
        dl = self.delta(h).view(-1, self.N, self.d)
        return torch.nn.functional.normalize(q.unsqueeze(1) + dl, dim=2)

    def score(self, q, cand_vecs, base):
        """base [B,M] = <q, x_p>; cand_vecs [B,M,d]. Returns [B,M]."""
        o = self.dirs(q)                                        # [B,N,d]
        extra = torch.einsum("bnd,bmd->bnm", o, cand_vecs).amax(1)
        return base + self.alpha * extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--cache", type=Path, default=Path("/dev/shm/gpuann/head"))
    ap.add_argument("--trainq", type=Path,
                    default=Path("/workspace/SYN/emb/"
                                 "msmarco-1m-trainpool__vanilla_queries.npy"))
    ap.add_argument("--graph", default="G_d64.npy")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--tau", type=float, default=0.05)
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

    nearM = torch.from_numpy(np.load(a.cache / f"nearM{a.M}_train.npy")).cuda().long()
    lab = torch.from_numpy(np.load(a.cache / f"lab{a.M}_train.npy")).cuda()
    Qtr = torch.from_numpy(np.load(a.trainq).astype(np.float32)).cuda()
    Qtr = torch.nn.functional.normalize(Qtr, dim=1)[: nearM.shape[0]]
    NT = Qtr.shape[0]
    print(f"train {NT:,} queries, M={a.M} candidates, N={a.N} directions, "
          f"hidden={a.hidden}")
    print(f"  positives {lab.float().mean():.1%} of candidates, "
          f"{lab[:, :8].float().mean():.1%} among the nearest 8")
    print(f"  queries with no positive: {(~lab.any(1)).float().mean():.2%}\n")

    head = PoolHead(d, a.N, a.hidden).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = a.epochs * ((NT + a.bs - 1) // a.bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)

    def listwise(sc, y):
        """-log of the positives' share of the softmax over the M candidates."""
        lse = torch.logsumexp(sc / a.tau, dim=1, keepdim=True)
        num = (((sc / a.tau) - lse) * y).sum(1)
        return -(num / y.sum(1).clamp(min=1)).mean()

    def prec_at(sc, y, v):
        top = torch.topk(sc, v, dim=1).indices
        return torch.gather(y.float(), 1, top).mean().item()

    print("  (precision@8 over the candidates == in_mass of the chosen entries)")
    for ep in range(a.epochs):
        perm = torch.randperm(NT, device=X.device)
        tot = n = 0.0
        for s in range(0, NT, a.bs):
            idx = perm[s:s + a.bs]
            cv = Vp[nearM[idx]]
            base = torch.einsum("bd,bmd->bm", Qtr[idx], cv)
            y = lab[idx]
            keep = y.any(1)
            if keep.sum() == 0:
                continue
            sc = head.score(Qtr[idx], cv, base)
            loss = listwise(sc[keep], y[keep].float())
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
            tot += loss.detach().item() * int(keep.sum()); n += int(keep.sum())
        if ep % max(1, a.epochs // 6) == 0 or ep == a.epochs - 1:
            with torch.no_grad():
                sl = slice(0, 4096)
                cv = Vp[nearM[sl]]
                base = torch.einsum("bd,bmd->bm", Qtr[sl], cv)
                sc = head.score(Qtr[sl], cv, base)
                p8, p8b = prec_at(sc, lab[sl], 8), prec_at(base, lab[sl], 8)
            print(f"  epoch {ep:3d}  loss {tot/n:.4f}  alpha "
                  f"{float(head.alpha):+.3f}  train prec@8 {p8:.3f} "
                  f"(scan {p8b:.3f})")
    print()

    rows = []
    for split in ("val", "test"):
        Q = torch.from_numpy(np.load(a.data / f"Q_{split}.npy")[: a.nq_eval]).cuda()
        gt = np.load(a.data / f"gt_{split}_ids.npy")[: a.nq_eval, : a.k]
        NQ = Q.shape[0]
        gts = [set(int(t) for t in r) for r in gt]
        att_gt = climb(X, G, torch.from_numpy(gt.astype(np.int64)).cuda(), Q)
        gs = [set(r.tolist()) for r in att_gt.cpu()]
        nm = select_a.select_nearest(Q, Vp, a.M, engine.METRIC_IP).long()
        head.eval()
        with torch.no_grad():
            cv = Vp[nm]
            base = torch.einsum("bd,bmd->bm", Q, cv)
            sc = head.score(Q, cv, base)
        print(f"=== {split}, {NQ} queries ===   alpha {float(head.alpha):+.3f}")

        pos = torch.empty(NQ, a.P, dtype=torch.long, device=X.device)
        pos.scatter_(1, torch.argsort(
            torch.einsum("bd,pd->bp", Q, Vp), dim=1, descending=True),
            torch.arange(a.P, device=X.device).expand(NQ, a.P))

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
            # 4.7: every entry diverted off the nearest ranks costs recall. This
            # is the column in_mass cannot see.
            rk = pos.gather(1, sel).float()
            print(f"  {name:20} V={V:<3d} recall {rec:.4f}  att/entry {ape:.3f}  "
                  f"in_mass {imm:.3f}  rank mean {rk.mean():6.1f} max "
                  f"{rk.max(1).values.mean():6.1f}  dist/q {de:.0f}")
            rows.append(dict(split=split, arm=name, V=V, recall=rec,
                             att_per_entry=ape, in_mass=imm, dist=de,
                             rank_mean=float(rk.mean()),
                             rank_max=float(rk.max(1).values.mean()),
                             alpha=float(head.alpha)))

        for V in a.V:
            B = max(1, a.total_B // V)
            report("nearest-V", nm[:, :V], V, B)
            report("pool head", torch.gather(nm, 1,
                                             torch.topk(sc, V, dim=1).indices), V, B)
            # the same score, but only allowed to rerank the nearest M' -- which
            # bounds how far off the nearest ranks it can divert an entry.
            for mp in (2 * V, 4 * V):
                if mp > a.M:
                    continue
                sc2 = sc.clone()
                sc2[:, mp:] = -1e30
                report(f"pool head, top-{mp} only",
                       torch.gather(nm, 1, torch.topk(sc2, V, dim=1).indices), V, B)
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(M=a.M, N=a.N, hidden=a.hidden,
                                         epochs=a.epochs, lr=a.lr, bs=a.bs,
                                         tau=a.tau, pool=a.pool, P=a.P, L=a.L,
                                         total_B=a.total_B, eps=a.stop_eps,
                                         rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
