"""Is the oracle's entry choice a LOW-RANK correction to the catalogue scan?

The entry scan picks top-V of S = Q Vp^T. Perfect entry choice at the same V is
worth 1.2-1.9x wall clock at matched recall (results/subspace/full_V*.json), and
no geometric selector over the pool has taken any of it: entry similarity
predicts fragment disjointness at AUC 0.606, and V=32 entries reach only 12.4
distinct attractors.

The proposal under test augments both sides with D*N extra coordinates --
x_p -> [x_p ; b_p], q -> [q ; a(q)] -- so the score becomes

    <q, x_p> + <a(q), b_p>      i.e.   S + A B^T,   rank(Delta) <= r = D*N.

b_p is offline (one row per pool node, P x r); a(q) must come from the query
alone. So the proposal is exactly the claim that the gap between the scan and
the oracle is a RANK-r ADDITIVE CORRECTION to the [Q,P] score matrix.

That splits into two independent questions, and this file answers only the
first, which is the cheap one:

  REALIZABILITY  does a single offline B (rank r) exist such that for every
                 query SOME a(q) puts the oracle entries in the top V?
                 B is fitted on the val query log -- legitimate, it is corpus
                 side, the same license NGFix/RoarGraph take -- and a(q) is
                 handed oracle freedom per test query. An upper bound.

  PREDICTABILITY is that a(q) a function of q?  (not here.)

If realizability fails at r = 1024 the idea is dead for any encoder, and we
learn that from one SVD and one Adam loop instead of from a trained model.

Target set: the pool node nearest each of the query's k true answers, deduped,
first V kept -- bitwise the `oracle-pieces` arm of results/subspace, whose
recall is already measured, so recovery@V converts to recall with no new run.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import select_a                                          # noqa: E402
from gpuann.pools import load_pool                                   # noqa: E402


def nearest_pool_to_answers(X, Vp, gt, ip, chunk=4096):
    """[NQ, k] -- the pool node nearest each true answer. No query involved."""
    out = []
    g = torch.from_numpy(gt.astype(np.int64)).cuda()
    for s in range(0, g.shape[0], chunk):
        tv = X[g[s:s + chunk].reshape(-1)]
        d = (tv @ Vp.T) if ip else -torch.cdist(tv, Vp)
        out.append(d.argmax(1).view(-1, g.shape[1]))
        del tv, d
    return torch.cat(out)


def target_sets(nearT, V):
    """Dedup each row keeping first appearance, keep the first V -- `fill`'s order.

    Returns a [NQ, V] int64 of pool indices, -1 padded, and the count per row.
    """
    NQ, k = nearT.shape
    keep = torch.ones_like(nearT, dtype=torch.bool)
    for j in range(1, k):
        keep[:, j] &= (nearT[:, j:j + 1] != nearT[:, :j]).all(1)
    rank = torch.where(keep, torch.arange(k, device=nearT.device).expand_as(nearT),
                       torch.full_like(nearT, 10 ** 6))
    order = torch.argsort(rank, 1)
    tgt = torch.gather(nearT, 1, order)
    nkeep = keep.sum(1).clamp(max=V)
    tgt = tgt[:, :V].clone()
    cols = torch.arange(V, device=nearT.device).expand(NQ, V)
    tgt[cols >= nkeep.unsqueeze(1)] = -1
    return tgt, nkeep


def recovery(scores, tgt, V):
    """Fraction of each row's targets that land in top-V of scores."""
    top = torch.topk(scores, V, dim=1).indices                       # [NQ, V]
    hit = (tgt.unsqueeze(2) == top.unsqueeze(1)) & (tgt.unsqueeze(2) >= 0)
    got = hit.any(2).sum(1).float()
    n = (tgt >= 0).sum(1).float()
    return (got / n).mean().item(), got.sum().item(), n.sum().item()


def fit_alpha(S, B, tgt, V, steps, lr, tau, verbose=False):
    """Per query, the best a(q) in R^r for the fixed offline B.

    Listwise: push the target columns' share of the softmax up. A hard hinge on
    the V-th rival was tried first and was worse than alpha=0 -- it trades one
    target against another and overshoots -- so the iterate that actually
    maximises recovery@V is the one kept, which makes the answer a LOWER bound
    on what rank r can do, never an upper one. B=None means r=P, the
    unconstrained control: it must reach ~1 or the optimiser is the bottleneck
    rather than the rank.
    """
    NQ, P = S.shape
    r = P if B is None else B.shape[1]
    A = torch.zeros(NQ, r, device=S.device, requires_grad=True)
    opt = torch.optim.Adam([A], lr=lr)
    valid = (tgt >= 0).float()
    j = valid.sum(1, keepdim=True)
    best = (recovery(S, tgt, V)[0], None)
    for it in range(steps):
        s = (S + A) if B is None else (S + A @ B.T)
        lse = torch.logsumexp(s / tau, dim=1, keepdim=True)
        st = s.gather(1, tgt.clamp(min=0)) / tau
        loss = -(((st - lse) * valid).sum(1) / j.squeeze(1)).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if it % 10 == 0 or it == steps - 1:
            with torch.no_grad():
                sc = (S + A) if B is None else (S + A @ B.T)
                rc = recovery(sc, tgt, V)[0]
            if rc > best[0]:
                best = (rc, A.detach().clone())
            if verbose and it % max(1, steps // 5) == 0:
                print(f"      step {it:4d}  loss {loss.item():.4f}  recovery {rc:.4f}")
    if best[1] is None:
        return S, torch.zeros(NQ, r, device=S.device)
    A = best[1]
    return ((S + A) if B is None else (S + A @ B.T)), A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("/dev/shm/gpuann/msmarco-1m"))
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--V", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--r", type=int, nargs="+", default=[0, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--pool", default="nest4k")
    ap.add_argument("--P", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--nq", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    man = json.loads((a.data / "manifest.json").read_text())
    ip = man["metric"].startswith("ip")
    X = torch.from_numpy(np.load(a.data / "X.npy")).cuda()
    pool = load_pool(a.data, a.pool, a.P)
    pt = torch.from_numpy(pool.astype(np.int64)).cuda()
    Vp = select_a.pool_vectors(X, pt)

    sets = {}
    for split in ("val", "test"):
        Q = torch.from_numpy(np.load(a.data / f"Q_{split}.npy")).cuda()
        gt = np.load(a.data / f"gt_{split}_ids.npy")[:, : a.k]
        if a.nq:
            Q, gt = Q[: a.nq], gt[: a.nq]
        with select_a.reduced_precision(True):
            S = (Q @ Vp.T) if ip else -torch.cdist(Q, Vp)
        sets[split] = (Q, S.float(), nearest_pool_to_answers(X, Vp, gt, ip))
    del X
    torch.cuda.empty_cache()

    NQ = sets["test"][1].shape[0]
    print(f"{a.data.name}  pool {a.pool} P={a.P}  val {sets['val'][1].shape[0]} / "
          f"test {NQ} queries  k={a.k}\n")

    # ---- where the oracle entries sit under the plain scan -----------------
    S_t, nearT_t = sets["test"][1], sets["test"][2]
    ordr = torch.argsort(S_t, dim=1, descending=True)
    pos = torch.empty_like(ordr)
    pos.scatter_(1, ordr, torch.arange(a.P, device=S_t.device).expand(NQ, a.P))
    tr = pos.gather(1, nearT_t)                                      # [NQ, k]
    q = torch.tensor([0.25, 0.5, 0.75, 0.9]).cuda()
    print("rank of the pool node nearest a true answer, under the plain scan "
          f"(P={a.P}):")
    print(f"  mean {tr.float().mean():.1f}   quartiles "
          + "  ".join(f"p{int(x*100)}={v:.0f}" for x, v in
                      zip(q.tolist(), torch.quantile(tr.float().flatten(), q).tolist())))
    for V in a.V:
        print(f"  share already inside top-{V}: {(tr < V).float().mean():.4f}")
    print()

    # ---- the spectrum of the oracle relation -------------------------------
    rows = []
    for V in a.V:
        tgt_v, _ = target_sets(sets["val"][2], V)
        T = torch.zeros(sets["val"][1].shape[0], a.P, device=S_t.device)
        T.scatter_(1, tgt_v.clamp(min=0), (tgt_v >= 0).float())
        pop = T.mean(0, keepdim=True)                                # pool-node popularity
        sv_raw = torch.linalg.svdvals(T)
        sv_c = torch.linalg.svdvals(T - T.sum(1, keepdim=True) * pop / pop.sum())
        def energy(sv, rr):
            return (sv[:rr].pow(2).sum() / sv.pow(2).sum()).item()
        print(f"V={V}: spectrum of the val oracle relation T [{T.shape[0]}x{a.P}] "
              f"-- share of Frobenius energy in the top r")
        print("      r:      " + "".join(f"{rr:>8d}" for rr in a.r if rr))
        print("  raw   :      " + "".join(f"{energy(sv_raw, rr):8.3f}" for rr in a.r if rr))
        print("  -popul:      " + "".join(f"{energy(sv_c, rr):8.3f}" for rr in a.r if rr))

        # B: the offline document-side augmentation, from the val log only
        _, _, Vh = torch.linalg.svd(T, full_matrices=False)
        tgt_t, nk = target_sets(nearT_t, V)
        print(f"  targets per test query: mean {nk.float().mean():.2f} of V={V}")
        for rr in a.r:
            t0 = time.perf_counter()
            if rr == 0:
                out = [("none", S_t, None)]
            elif rr >= a.P:
                out = [("free", *reversed(fit_alpha(S_t, None, tgt_t, V, a.steps,
                                                    a.lr, a.tau)[::-1]))]
            else:
                out = []
                Bs = torch.nn.functional.normalize(Vh[:rr].T.contiguous(), dim=0)
                out.append(("svd", *fit_alpha(S_t, Bs, tgt_t, V, a.steps, a.lr,
                                              a.tau)))
                g = torch.Generator(device="cuda").manual_seed(0)
                Br = torch.randn(a.P, rr, device=S_t.device, generator=g)
                Br = torch.nn.functional.normalize(torch.linalg.qr(Br)[0], dim=0)
                out.append(("rand", *fit_alpha(S_t, Br, tgt_t, V, a.steps, a.lr,
                                               a.tau)))
            for nm, sc, A in out:
                rec, got, tot = recovery(sc, tgt_t, V)
                an = 0.0 if A is None else A.norm(dim=1).mean().item()
                print(f"    r={rr:<5d} B={nm:<5s} recovery@{V} {rec:.4f}"
                      f"  ({got:.0f}/{tot:.0f})  |a| {an:7.3f}")
                rows.append(dict(V=V, r=rr, B=nm, recovery=rec, got=got, tot=tot,
                                 anorm=an,
                                 energy_raw=energy(sv_raw, rr) if rr else 0.0,
                                 energy_cent=energy(sv_c, rr) if rr else 0.0))
            print(f"          ({time.perf_counter()-t0:5.1f}s)")
        print()
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(dict(data=a.data.name, pool=a.pool, P=a.P,
                                         k=a.k, steps=a.steps, lr=a.lr,
                                         tau=a.tau, rows=rows), indent=2))
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
