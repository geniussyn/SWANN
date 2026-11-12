"""Stage 1 acceptance: the engine means what the design says it means.

The checks are the design document's own minimum set -- agreement with a CPU
local search and with brute force, reproducible tie-breaks, valid ids, visited
overflow reported rather than swallowed, branch merging -- plus the probe/resume
equivalence that stage 2 is built on and that costs nothing to assert now.

Run: python tests/test_engine.py     (or pytest tests/test_engine.py)
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gpuann import engine, refcpu                                   # noqa: E402

DEV = "cuda"
TEAMS = (4, 8, 16, 32)


def toy(n=2000, d=64, deg=32, seed=0, metric=engine.METRIC_IP):
    """A small corpus and its exact kNN graph. Normalised when the metric is IP."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, d), dtype=np.float32)
    if metric == engine.METRIC_IP:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = np.ascontiguousarray(X, dtype=np.float32)
    sim = X @ X.T if metric == engine.METRIC_IP else -((X[:, None] - X[None]) ** 2).sum(-1)
    np.fill_diagonal(sim, -np.inf)
    G = np.ascontiguousarray(np.argsort(-sim, axis=1)[:, :deg].astype(np.int32))
    return X, G


def to_gpu(X, G, Qv):
    return (torch.from_numpy(X).to(DEV), torch.from_numpy(G).to(DEV),
            torch.from_numpy(Qv).to(DEV))


def test_cpu_gpu_exact():
    """Same queue, same distances, bit for bit, at every team size."""
    for metric in (engine.METRIC_IP, engine.METRIC_L2):
        X, G = toy(metric=metric)
        rng = np.random.default_rng(7)
        nq, J, L, budget = 5, 3, 16, 12
        Qv = rng.standard_normal((nq, X.shape[1]), dtype=np.float32)
        if metric == engine.METRIC_IP:
            Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
        Qv = np.ascontiguousarray(Qv)
        seeds = rng.integers(0, X.shape[0], size=(nq, J)).astype(np.int32)
        Xg, Gg, Qg = to_gpu(X, G, Qv)

        for team in TEAMS:
            st = engine.init_states(Xg, Qg, torch.from_numpy(seeds), L=L,
                                    B=budget, D=G.shape[1],
                                    metric=metric, team_size=team)
            engine.advance(Xg, Gg, Qg, st, budget)
            st.check()
            gi, gd = st.ids.cpu().numpy(), st.dist.cpu().numpy()
            ge = st.expanded.cpu().numpy()
            gc = st.counters.cpu().numpy()

            for qi in range(nq):
                for j in range(J):
                    ref = refcpu.run(X, G, Qv[qi], int(seeds[qi, j]), L=L,
                                     B=budget, metric=metric, team=team)
                    rq = ref["queue"]
                    assert [e[0] for e in rq] == [i for i in gi[qi, j] if i >= 0], \
                        f"ids differ metric={metric} team={team} q={qi} j={j}"
                    for slot, e in enumerate(rq):
                        assert np.float32(e[1]) == gd[qi, j, slot], \
                            f"distance differs at slot {slot}: {e[1]!r} vs {gd[qi,j,slot]!r}"
                        assert int(e[2]) == int(ge[qi, j, slot]), "expanded flag differs"
                    assert gc[qi, j, 0] == ref["counters"]["expansions"]
                    assert gc[qi, j, 1] == ref["counters"]["dist_evals"]
    return "cpu/gpu exact agreement (IP and L2, teams 4/8/16/32)"


def test_resume_equivalence():
    """advance(a) then advance(b) is advance(a+b): the whole of stage 2 rests on this."""
    X, G = toy(seed=3)
    rng = np.random.default_rng(11)
    nq, J, L, budget = 8, 4, 32, 24
    Qv = rng.standard_normal((nq, X.shape[1]), dtype=np.float32)
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    Qv = np.ascontiguousarray(Qv)
    seeds = torch.from_numpy(rng.integers(0, X.shape[0], (nq, J)).astype(np.int32))
    Xg, Gg, Qg = to_gpu(X, G, Qv)

    kw = dict(L=L, B=budget, D=G.shape[1], team_size=32)
    one = engine.init_states(Xg, Qg, seeds, **kw)
    engine.advance(Xg, Gg, Qg, one, budget)

    for split in (1, 2, 7, budget - 1):
        two = engine.init_states(Xg, Qg, seeds, **kw)
        engine.advance(Xg, Gg, Qg, two, split)
        engine.advance(Xg, Gg, Qg, two, budget - split)
        assert torch.equal(one.ids, two.ids), f"ids differ after split {split}"
        assert torch.equal(one.dist, two.dist), f"distances differ after split {split}"
        assert torch.equal(one.expanded, two.expanded), f"flags differ after split {split}"
        assert torch.equal(one.counters, two.counters), f"counters differ after split {split}"
        assert torch.equal(one.visited, two.visited), f"visited differs after split {split}"
    return "probe/resume equivalence over 4 split points"


def test_budget_is_a_state_property():
    """Asking for more than max_expansions does not buy more expansions."""
    X, G = toy(seed=5)
    rng = np.random.default_rng(13)
    Qv = np.ascontiguousarray(rng.standard_normal((4, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = torch.from_numpy(rng.integers(0, X.shape[0], (4, 2)).astype(np.int32))
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    st = engine.init_states(Xg, Qg, seeds, L=16, B=10, D=G.shape[1])
    engine.advance(Xg, Gg, Qg, st, 100)
    assert int(st.expansions.max()) == 10, int(st.expansions.max())
    return "budget capped by the state, not by the call"


def test_brute_force_on_complete_graph():
    """On a complete graph one expansion sees everything, so the queue must be
    the exact top-L. Checks the distances themselves, not only the plumbing."""
    n, d, L = 96, 48, 16
    rng = np.random.default_rng(17)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    X = np.ascontiguousarray(X)
    G = np.ascontiguousarray(np.stack([np.delete(np.arange(n), i) for i in range(n)])
                             .astype(np.int32)[:, :95])
    Qv = np.ascontiguousarray(rng.standard_normal((6, d), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    seeds = torch.zeros((6, 1), dtype=torch.int32)
    ids, dist, st = engine.search(Xg, Gg, Qg, seeds, k=10, L=L, B=1)
    exact = 1.0 - (Qv @ X.T)
    want = np.argsort(exact, axis=1, kind="stable")[:, :10]
    got = ids.cpu().numpy()
    assert (got == want).all(), f"\n{got}\n{want}"
    assert np.allclose(dist.cpu().numpy(), np.take_along_axis(exact, want, 1), atol=2e-6)
    return "complete-graph search equals brute force top-10"


def test_dead_branch_and_exhaustion():
    """A seed of -1 is an empty branch; a sink node exhausts instead of looping."""
    n, d = 64, 16
    rng = np.random.default_rng(19)
    X = np.ascontiguousarray(rng.standard_normal((n, d), dtype=np.float32))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    G = np.full((n, 8), -1, dtype=np.int32)          # every node is a sink
    Qv = np.ascontiguousarray(rng.standard_normal((2, d), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    seeds = torch.tensor([[0, -1], [5, -1]], dtype=torch.int32)
    st = engine.init_states(Xg, Qg, seeds, L=16, B=16, D=8)
    engine.advance(Xg, Gg, Qg, st, 16)
    st.check()
    assert int(st.expansions[0, 0]) == 1 and int(st.exhausted[0, 0]) == 1
    assert int(st.expansions[0, 1]) == 0 and int(st.exhausted[0, 1]) == 1
    assert int(st.ids[0, 1, 0]) == -1, "an empty branch must stay empty"
    ids, _ = engine.branch_topk(st, 10)
    assert int((ids >= 0).sum()) == 2, "only the seeded branches have results"
    return "sinks exhaust, seed -1 is an empty branch"


def test_visited_overflow_is_reported():
    """An undersized table must raise, not quietly lose nodes."""
    X, G = toy(n=1000, seed=23)
    rng = np.random.default_rng(29)
    Qv = np.ascontiguousarray(rng.standard_normal((2, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    st = engine.init_states(Xg, Qg, torch.zeros((2, 1), dtype=torch.int32),
                            L=16, B=16, D=G.shape[1])
    st.visited = torch.zeros((2, 1, 64), dtype=torch.int32, device=DEV)   # far too small
    st.counters.zero_()
    engine.advance(Xg, Gg, Qg, st, 16)
    assert int(st.hash_failures.sum()) > 0, "a full table went unnoticed"
    try:
        st.check()
    except RuntimeError:
        return "visited overflow raises"
    raise AssertionError("check() did not raise on overflow")


def test_merge_dedup():
    """Merged top-k is the exact best over the union of the branch results, once each."""
    X, G = toy(seed=31)
    rng = np.random.default_rng(37)
    nq, J, k = 6, 5, 10
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = torch.from_numpy(rng.integers(0, X.shape[0], (nq, J)).astype(np.int32))
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    ids, dist, st = engine.search(Xg, Gg, Qg, seeds, k=k, L=16, B=16)
    bi, bd = engine.branch_topk(st, k)
    bi, bd = bi.cpu().numpy(), bd.cpu().numpy()
    gi, gd = ids.cpu().numpy(), dist.cpu().numpy()
    for q in range(nq):
        pool = {}
        for j in range(J):
            for t in range(k):
                if bi[q, j, t] >= 0:
                    pool.setdefault(int(bi[q, j, t]), float(bd[q, j, t]))
        want = [i for i, _ in sorted(pool.items(), key=lambda kv: (kv[1], kv[0]))[:k]]
        got = [int(i) for i in gi[q] if i >= 0]
        assert got == want, f"q={q}\n{got}\n{want}"
        assert len(set(got)) == len(got), "duplicate id survived the merge"
    return "branch merge is an exact dedup'd selection"


def test_shared_cache_is_free_of_consequence():
    """The shared cache changes how much is computed and nothing else.

    A cached distance is bitwise equal to the one it replaces -- same inputs,
    same reduction order -- so the whole search must come out identical, and the
    only thing that moves is the distance count.
    """
    X, G = toy(seed=79)
    rng = np.random.default_rng(83)
    nq, E, L, budget = 16, 24, 32, 12
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = torch.from_numpy(rng.integers(0, X.shape[0], (nq, E)).astype(np.int32))
    Xg, Gg, Qg = to_gpu(X, G, Qv)

    plain = engine.search(Xg, Gg, Qg, seeds, k=10, L=L, B=budget)
    cached = engine.search(Xg, Gg, Qg, seeds, k=10, L=L, B=budget, shared_cache=True)
    assert torch.equal(plain[0], cached[0]), "ids differ with the cache on"
    assert torch.equal(plain[1], cached[1]), "distances differ with the cache on"
    assert torch.equal(plain[2].ids, cached[2].ids), "queues differ"
    assert torch.equal(plain[2].expansions, cached[2].expansions), "expansions differ"
    a = float(plain[2].dist_evals.sum(1).float().mean())
    b = float(cached[2].dist_evals.sum(1).float().mean())
    hits = float(cached[2].cache_hits.sum(1).float().mean())
    assert b < a, f"cache saved nothing: {b:.0f} vs {a:.0f}"
    assert abs((a - b) - hits) < 1e-6, "hits do not account for the saving"
    return f"identical output, {a:.0f} -> {b:.0f} distances ({100*(1-b/a):.0f}% saved)"


def test_determinism():
    """Identical inputs, identical bits, run after run."""
    X, G = toy(seed=41)
    rng = np.random.default_rng(43)
    Qv = np.ascontiguousarray(rng.standard_normal((16, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = torch.from_numpy(rng.integers(0, X.shape[0], (16, 8)).astype(np.int32))
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    ref = None
    for _ in range(5):
        ids, dist, st = engine.search(Xg, Gg, Qg, seeds, k=10, L=16, B=16)
        now = (ids.cpu().numpy().copy(), dist.cpu().numpy().copy(),
               st.counters.cpu().numpy().copy())
        if ref is None:
            ref = now
        else:
            for a, b in zip(ref, now):
                assert (a == b).all(), "run-to-run difference"
    return "5 repeats are bitwise identical"


def test_multiseed_matches_cpu():
    """Several seeds sharing one branch: same queue as the CPU reference, under
    both expansion policies, with the lineage of every entry agreeing too."""
    X, G = toy(seed=53)
    rng = np.random.default_rng(59)
    nq, V, L, budget = 4, 6, 32, 20
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = rng.integers(0, X.shape[0], (nq, 1, V)).astype(np.int32)
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    for policy in (engine.POLICY_BEST, engine.POLICY_ROUND_ROBIN):
        st = engine.init_states(Xg, Qg, torch.from_numpy(seeds), L=L, B=budget,
                                D=G.shape[1], policy=policy)
        engine.advance(Xg, Gg, Qg, st, budget)
        st.check()
        gi, gd = st.ids.cpu().numpy(), st.dist.cpu().numpy()
        gl = st.lineage.cpu().numpy()
        for q in range(nq):
            ref = refcpu.run(X, G, Qv[q], list(seeds[q, 0]), L=L, B=budget, policy=policy)
            rq = ref["queue"]
            assert [e[0] for e in rq] == [i for i in gi[q, 0] if i >= 0], f"ids differ, policy {policy}"
            for slot, e in enumerate(rq):
                assert np.float32(e[1]) == gd[q, 0, slot], "distance differs"
                assert int(e[3]) == int(gl[q, 0, slot]), "lineage differs"
            assert gi[q, 0][0] >= 0
    return "multi-seed branch matches CPU under both policies"


def test_early_stop_matches_cpu():
    """Per-lineage retirement: same queue as the reference, and it really stops."""
    X, G = toy(seed=71)
    rng = np.random.default_rng(73)
    nq, V, L, budget, k = 4, 4, 32, 64, 10
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = rng.integers(0, X.shape[0], (nq, 1, V)).astype(np.int32)
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    for eps in (0.0, 0.1, 0.25):
        st = engine.init_states(Xg, Qg, torch.from_numpy(seeds), L=L, B=budget,
                                D=G.shape[1], policy=engine.POLICY_ROUND_ROBIN,
                                stop_eps=eps, kstop=k)
        engine.advance(Xg, Gg, Qg, st, budget)
        st.check()
        gi, gd = st.ids.cpu().numpy(), st.dist.cpu().numpy()
        for q in range(nq):
            ref = refcpu.run(X, G, Qv[q], list(seeds[q, 0]), L=L, B=budget,
                             policy=engine.POLICY_ROUND_ROBIN, stop_eps=eps, kstop=k)
            assert [e[0] for e in ref["queue"]] == [i for i in gi[q, 0] if i >= 0], \
                f"ids differ at eps={eps}"
            for slot, e in enumerate(ref["queue"]):
                assert np.float32(e[1]) == gd[q, 0, slot], "distance differs"
            assert ref["counters"]["expansions"] == int(st.expansions[q, 0])
        assert int(st.expansions.max()) < budget, f"eps={eps} never stopped early"
    return "early stop matches CPU and terminates before the budget"


def test_shared_state_beats_split_branches_on_work():
    """V seeds in one state evaluate each node once; V branches do not.

    Compared per expansion actually performed, because the two structures do not
    spend the same number of expansions: a shared queue of width L holds roughly
    L/V live candidates per lineage, so round-robin exhausts early unless L is
    comfortably larger than V. That is a real constraint on the design, not an
    artefact of the test.
    """
    X, G = toy(seed=61)
    rng = np.random.default_rng(67)
    nq, V, budget = 32, 8, 64
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    seeds = rng.integers(0, X.shape[0], (nq, V)).astype(np.int32)
    Xg, Gg, Qg = to_gpu(X, G, Qv)

    L = 64
    split = engine.init_states(Xg, Qg, torch.from_numpy(seeds), L=L,
                               B=budget // V, D=G.shape[1])
    engine.advance(Xg, Gg, Qg, split, budget // V)
    shared = engine.init_states(Xg, Qg, torch.from_numpy(seeds).unsqueeze(1), L=L,
                                B=budget, D=G.shape[1],
                                policy=engine.POLICY_ROUND_ROBIN)
    engine.advance(Xg, Gg, Qg, shared, budget)
    a = (split.dist_evals.sum(1).float() / split.expansions.sum(1).clamp(min=1).float()).mean()
    b = (shared.dist_evals[:, 0].float() / shared.expansions[:, 0].clamp(min=1).float()).mean()
    assert float(b) < float(a), f"shared {float(b):.1f} vs split {float(a):.1f} distances/expansion"
    return (f"shared state: {float(b):.1f} distances per expansion vs {float(a):.1f} "
            f"for split branches")


def ragged(X, G, seed=0, lo=0.25):
    """G with each row truncated to a random degree and padded with -1.

    What a pruned, degree-allocated build produces: the row width is the widest
    node's degree and every other node leaves a tail of -1. `lo` is the narrowest
    degree as a fraction of the width, so the padding is most of the matrix at
    the low end -- the case the kernel has to stop charging for.
    """
    rng = np.random.default_rng(seed)
    n, W = G.shape
    deg = rng.integers(max(1, int(lo * W)), W + 1, n)
    out = G.copy()
    pos = np.arange(W).reshape(1, -1)
    out[pos >= deg.reshape(-1, 1)] = -1
    return np.ascontiguousarray(out), deg.astype(np.int32)


def test_degree_is_a_cost_not_an_answer():
    """Bounding every loop by deg[p] changes what is spent and nothing else.

    The slots a per-node degree skips are exactly the -1 ones the kernel already
    rejected, so the two runs have to agree bitwise -- queue, distances, counters
    and all -- while the padded run pays for the width. That equality is the
    whole licence for the optimisation, so it is asserted over both policies,
    both metrics and every team size.

    Both runs advance *the same* initialised state, cloned, rather than two
    states built by two calls. That is the comparison the claim is about: it is
    `advance` that reads the degree, so initialising twice would put a second
    kernel inside the equality being asserted. It was also, until the seed dedup
    was fixed, the only way to run this at all -- two separately initialised
    round-robin searches disagreed. They no longer do, which the last block
    asserts, so the cloning cannot go back to hiding that.
    """
    import copy
    saved = []
    for metric in (engine.METRIC_IP, engine.METRIC_L2):
        X, G0 = toy(seed=23, deg=64, metric=metric)
        G, deg = ragged(X, G0, seed=29)
        rng = np.random.default_rng(31)
        nq, V, L, budget = 24, 8, 48, 32
        Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
        if metric == engine.METRIC_IP:
            Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
        seeds = torch.from_numpy(rng.integers(0, X.shape[0], (nq, 1, V)).astype(np.int32))
        Xg, Gg, Qg = to_gpu(X, G, Qv)
        degt = engine.graph_degrees(Gg)
        assert degt is not None, "a ragged graph must produce a degree array"
        assert torch.equal(degt[0].cpu(), torch.from_numpy(deg)), "degrees disagree"
        assert degt[1] == int(deg.max())

        def dup(st):
            n = copy.copy(st)
            for f in ("ids", "dist", "expanded", "lineage", "visited", "counters"):
                setattr(n, f, getattr(st, f).clone())
            return n

        for policy in (engine.POLICY_BEST, engine.POLICY_ROUND_ROBIN):
            for team in TEAMS:
                base = engine.init_states(Xg, Qg, seeds, L=L, B=budget, D=G.shape[1],
                                          metric=metric, team_size=team,
                                          policy=policy, kstop=10)
                wide, tight = dup(base), dup(base)
                engine.advance(Xg, Gg, Qg, wide, budget)
                engine.advance(Xg, Gg, Qg, tight, budget, deg=degt)
                tag = f"metric={metric} policy={policy} team={team}"
                assert torch.equal(wide.ids, tight.ids), f"queues differ ({tag})"
                assert torch.equal(wide.dist, tight.dist), f"queue distances differ ({tag})"
                assert torch.equal(wide.expanded, tight.expanded), f"flags differ ({tag})"
                assert torch.equal(wide.lineage, tight.lineage), f"lineages differ ({tag})"
                assert torch.equal(wide.counters, tight.counters), f"counters differ ({tag})"
        # And the whole thing again from two separate inits, which is what the
        # cloning above is deliberately not testing.
        for team in (8, 16):
            runs = []
            for _ in range(4):
                st = engine.init_states(Xg, Qg, seeds, L=L, B=budget, D=G.shape[1],
                                        metric=metric, team_size=team,
                                        policy=engine.POLICY_ROUND_ROBIN, kstop=10)
                engine.advance(Xg, Gg, Qg, st, budget, deg=degt)
                runs.append((st.ids.clone(), st.lineage.clone(), st.counters.clone()))
            n = sum(0 if all(torch.equal(a, b) for a, b in zip(runs[0], o)) else 1
                    for o in runs[1:])
            assert n == 0, (f"metric={metric} team={team}: {n}/3 separately "
                            f"initialised round-robin searches disagree")
        saved.append(100.0 * (1.0 - deg.mean() / G.shape[1]))
    return (f"identical over 2 metrics x 2 policies x {len(TEAMS)} teams, "
            f"{saved[0]:.0f}% of the matrix skipped")


def test_graph_degrees_guards_its_contract():
    """A full graph needs no array, and a row with a hole is refused.

    The kernel reads slots 0..deg[p) and trusts that the edges are there, so a
    graph whose -1 sits in the middle of a row would be searched with its tail
    silently dropped. That is a wrong answer, not a slow one, so it is caught in
    Python rather than left to the kernel.
    """
    X, G0 = toy(seed=37, deg=32)
    Gg = torch.from_numpy(G0).to(DEV)
    assert engine.graph_degrees(Gg) is None, "a full graph should need no degrees"

    holed = G0.copy()
    holed[5, 3] = -1                                   # a hole, not a tail
    try:
        engine.graph_degrees(torch.from_numpy(holed).to(DEV))
    except ValueError as e:
        assert "front-compacted" in str(e)
    else:
        raise AssertionError("a row with a hole was accepted")

    G, deg = ragged(X, G0, seed=41)
    Gr = torch.from_numpy(G).to(DEV)
    dt, dmax = engine.graph_degrees(Gr)
    try:                                               # Dmax must be the real one
        engine.advance(torch.from_numpy(X).to(DEV), Gr,
                       torch.from_numpy(np.ascontiguousarray(X[:2])).to(DEV),
                       engine.init_states(torch.from_numpy(X).to(DEV),
                                          torch.from_numpy(np.ascontiguousarray(X[:2])).to(DEV),
                                          torch.zeros((2, 1, 1), dtype=torch.int32),
                                          L=16, B=4, D=G.shape[1]),
                       4, deg=(dt, dmax + 1))
    except RuntimeError as e:
        assert "Dmax" in str(e), str(e)
    else:
        raise AssertionError("a Dmax past the row width was accepted")
    return f"full graph -> None, hole refused, Dmax={dmax} enforced"


def _unpack_narrow(packed, scale, bits):
    """The stored bytes back to codes and floats, in numpy, without the kernel.

    The kernel's own unpack is an exponent trick on a packed word; this is the
    plain shift-and-mask reading of the same bytes, so a disagreement between
    the two is a kernel bug and not a shared mistake.
    """
    per, off, mask = 8 // bits, 2 ** (bits - 1), (1 << bits) - 1
    N, nb = packed.shape
    codes = np.empty((N, nb * per), dtype=np.int32)
    for i in range(per):
        codes[:, i::per] = (packed >> (bits * i)) & mask
    return codes, (codes - off).astype(np.float64) * scale[:, None].astype(np.float64)


def test_narrow_kernel_equals_its_stored_corpus():
    """The packed search must return the exact top-k of the corpus it stores.

    On a complete graph one expansion sees every node, so the answer is a brute
    force and the only thing left to be wrong is the unpack. That is the whole
    point: a 4-bit kernel that mis-reads its own bytes does not crash and does
    not warn, it just returns a worse list -- the offset-binary trap cost recall
    0 with a kernel that ran clean. Here the reference is built by shifting and
    masking the same bytes in numpy, so the exponent trick, its bias constant,
    the nibble order and the per-vector scale are all checked against something
    that shares no code with them.

    Every team size, because `Vec<nib4>::N` and `Vec<nib2>::N` set the stride and
    at team 32 with 2-bit values half the lanes fall past d and must contribute
    nothing rather than read past the row.
    """
    n, d, L, nq = 96, 256, 16, 6
    rng = np.random.default_rng(17)
    X = rng.standard_normal((n, d), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(DEV)
    G = np.ascontiguousarray(np.stack([np.delete(np.arange(n), i) for i in range(n)])
                             .astype(np.int32))
    Gg = torch.from_numpy(G).to(DEV)
    Qv = rng.standard_normal((nq, d), dtype=np.float32)
    Qv /= np.linalg.norm(Qv, axis=1, keepdims=True)
    Qg = torch.from_numpy(np.ascontiguousarray(Qv)).to(DEV)
    seeds = torch.zeros((nq, 1), dtype=torch.int32)
    out = []
    for bits in (4, 2):
        P, sc, rot = engine.pack_narrow(Xg, bits)
        assert P.dtype == torch.uint8 and P.shape == (n, d * bits // 8)
        codes, Xdq = _unpack_narrow(P.cpu().numpy(), sc.cpu().numpy(), bits)
        Qr = engine.rotate_queries(Qg, rot)
        exact = 1.0 - (Qr.cpu().numpy().astype(np.float64) @ Xdq.T)
        want = np.argsort(exact, axis=1, kind="stable")[:, :10]
        for team in TEAMS:
            ids, dist, _ = engine.search(P, Gg, Qr, seeds, k=10, L=L, B=1,
                                         team_size=team, xscale=sc, xbits=bits)
            got = ids.cpu().numpy()
            assert (got == want).all(), f"{bits}-bit team {team}\n{got}\n{want}"
            assert np.allclose(dist.cpu().numpy(),
                               np.take_along_axis(exact, want, 1), atol=1e-4), \
                f"{bits}-bit team {team} distances"
        out.append(f"{bits}-bit")
    return "packed search equals brute force on its own bytes: " + ", ".join(out)


def test_pack_narrow_stores_offset_binary():
    """The stored code is q + 2^(bits-1), and the scale is the fitted sigma.

    This is the representation the kernel's exponent trick assumes, and it is the
    one thing about the format that cannot be inferred from a running kernel: in
    two's complement the same unpack maps 7 to -1 and -8 to 0 and everything
    still runs. Checked against a recomputation from X and the returned rotation
    and scale, and the extreme codes are asserted to actually occur, so a clip
    that never saturates cannot pass by leaving the ends unused.
    """
    rng = np.random.default_rng(5)
    X = rng.standard_normal((512, 64), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(DEV)
    out = []
    for bits, clip in ((4, 2.5), (2, 1.0)):
        lvl, off = float(2 ** (bits - 1) - 1), 2 ** (bits - 1)
        P, sc, rot = engine.pack_narrow(Xg, bits, clip=clip)
        codes, _ = _unpack_narrow(P.cpu().numpy(), sc.cpu().numpy(), bits)
        v = Xg @ rot
        want_sc = (clip * v.std(1, keepdim=True).clamp(min=1e-12)) / lvl
        assert torch.allclose(sc, want_sc.squeeze(1), rtol=1e-6), f"{bits}-bit scale"
        q = torch.round(v / want_sc).clamp(-lvl - 1, lvl).cpu().numpy().astype(np.int32)
        bad = int((codes != q + off).sum())
        assert bad == 0, f"{bits}-bit: {bad} codes are not q + {off}"
        assert codes.min() == 0 and codes.max() == 2 * off - 1, \
            f"{bits}-bit uses only codes {codes.min()}..{codes.max()}"
        out.append(f"{bits}-bit in 0..{2 * off - 1}")
    return "offset-binary round trip: " + ", ".join(out)


def test_narrow_contract_is_enforced():
    """Every way of holding the packed path wrong has to raise, not run.

    The narrow corpus carries its width and its scales out of band -- the tensor
    is just bytes -- so a wrong `xbits` or a mismatched `xscale` is a plausible
    caller error that the kernel would otherwise read straight through.
    """
    rng = np.random.default_rng(9)
    X = rng.standard_normal((64, 64), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(DEV)
    Gg = torch.from_numpy(np.ascontiguousarray(
        np.tile(np.arange(1, 9, dtype=np.int32), (64, 1)))).to(DEV)
    seeds = torch.zeros((4, 1), dtype=torch.int32)
    P, sc, rot = engine.pack_narrow(Xg, 4)
    Qr = engine.rotate_queries(Xg[:4].contiguous(), rot)

    def refuses(part, **kw):
        kw.setdefault("G", Gg); kw.setdefault("seeds", seeds)
        kw.setdefault("k", 10); kw.setdefault("L", 16); kw.setdefault("B", 1)
        kw.setdefault("team_size", 32)
        try:
            engine.search(**kw)
        except (RuntimeError, ValueError) as e:
            assert part in str(e), f"{part}: wrong complaint -- {e}"
            return
        raise AssertionError(f"{part}: accepted")

    refuses("xbits", X=P, Qv=Qr, xscale=sc, xbits=3)
    refuses("xbits", X=P, Qv=Qr, xscale=sc, xbits=8)
    refuses("inner-product", X=P, Qv=Qr, xscale=sc, xbits=4,
            metric=engine.METRIC_L2)
    refuses("xscale", X=P, Qv=Qr, xscale=sc[:32].contiguous(), xbits=4)
    refuses("xscale", X=P, Qv=Qr, xscale=sc.double(), xbits=4)
    refuses("xscale is only for", X=Xg, Qv=Xg[:4].contiguous(), xscale=sc)
    for bits, part in ((3, "bits must be"), (8, "bits must be")):
        try:
            engine.pack_narrow(Xg, bits)
        except ValueError as e:
            assert part in str(e), str(e)
        else:
            raise AssertionError(f"pack_narrow accepted bits={bits}")
    try:
        engine.pack_narrow(Xg[:, :48].contiguous(), 4)
    except ValueError as e:
        assert "d % 32" in str(e), str(e)
    else:
        raise AssertionError("pack_narrow accepted d = 48")
    return "bad xbits, metric, xscale and dimension all refused"


def test_rotation_is_a_change_of_basis_only():
    """The rotation must be orthogonal, reproducible, and invisible to the score.

    Orthogonal because the per-vector scale is fitted to a Gaussian coordinate
    and a non-orthogonal map would change the norms it is fitted against;
    reproducible because a packed corpus written to disk is unsearchable without
    the same basis, and the basis is not stored with it -- it is a fixed seed.
    """
    rng = np.random.default_rng(11)
    X = rng.standard_normal((256, 64), dtype=np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    Xg = torch.from_numpy(np.ascontiguousarray(X)).to(DEV)
    Qg = torch.from_numpy(np.ascontiguousarray(
        rng.standard_normal((16, 64), dtype=np.float32))).to(DEV)
    P1, s1, rot = engine.pack_narrow(Xg, 4, clip=2.5)
    assert torch.allclose(rot.T @ rot, torch.eye(64, device=rot.device), atol=1e-5), \
        "the rotation is not orthogonal"
    before, after = Qg @ Xg.T, engine.rotate_queries(Qg, rot) @ (Xg @ rot).T
    assert torch.allclose(before, after, atol=1e-4), "the rotation moved the score"
    P2, s2, rot2 = engine.pack_narrow(Xg, 4, clip=2.5)
    assert torch.equal(rot, rot2), "the default rotation is not reproducible"
    assert torch.equal(P1, P2) and torch.equal(s1, s2), "packing is not reproducible"
    P3, s3, _ = engine.pack_narrow(Xg, 4, rot=rot, clip=2.5)
    assert torch.equal(P1, P3) and torch.equal(s1, s3), "a passed rotation differs"
    return "rotation orthogonal, reproducible and score-preserving"


def test_duplicate_seeds_keep_the_lowest_lineage():
    """A node named twice in one branch's seeds is kept once, as its first index.

    The slot a seed lands in *is* its lineage, so deduping the seeds by whichever
    team won an atomicCAS made the label depend on warp scheduling: `ids`,
    `dist`, `visited` and the counters all matched run to run and only `lineage`
    moved, and round-robin -- the one policy that reads the label to pick a
    parent -- then ran a different search. It reproduced on 3 runs in 13 at team
    8 and 10 in 13 at team 16, and not at all at team 4 or 32, because those are
    the team sizes that put the two duplicate slots in one warp or in different
    `base` iterations. bench/repro_rr_nondeterminism.py has the narrowing.

    So this asserts the rule and not just stability: the surviving label is the
    lowest index, which is a property of the seed list and of nothing else. That
    rule is not a new convention either -- `refcpu.init_state` has always deduped
    its seeds by index (`if s in seen: continue`, `lineage = i`), so the kernel
    was disagreeing with the reference and not merely with itself. The reference
    is what the last block checks against; `test_multiseed_matches_cpu` already
    compares lineages, it just never drew a duplicate seed.
    """
    X, G = toy(seed=23, deg=64, metric=engine.METRIC_L2)
    rng = np.random.default_rng(31)
    nq, V, L = 12, 8, 48
    Qv = np.ascontiguousarray(rng.standard_normal((nq, X.shape[1]), dtype=np.float32))
    row = [5, 9, 5, 2, 9, 7, 5, 1]                 # 5 at 0/2/6, 9 at 1/4
    want = {5: 0, 9: 1, 2: 3, 7: 5, 1: 7}
    seeds = torch.from_numpy(np.tile(np.array(row, dtype=np.int32), (nq, 1, 1)))
    Xg, Gg, Qg = to_gpu(X, G, Qv)
    for team in TEAMS:
        runs = []
        for _ in range(12):
            st = engine.init_states(Xg, Qg, seeds, L=L, B=32, D=G.shape[1],
                                    metric=engine.METRIC_L2, team_size=team,
                                    policy=engine.POLICY_ROUND_ROBIN, kstop=10)
            ids = st.ids[:, 0].cpu().numpy()
            lin = st.lineage[:, 0].cpu().numpy()
            # Every query, not just the first: the dedup races per block, so one
            # query can flip while its neighbour does not.
            for qi in range(nq):
                live = ids[qi][ids[qi] >= 0]
                assert sorted(live.tolist()) == sorted(want), \
                    f"team {team} q{qi}: queue holds {sorted(live.tolist())}"
                for v, want_l in want.items():
                    got = lin[qi][ids[qi] == v]
                    assert got.tolist() == [want_l], \
                        f"team {team} q{qi}: node {v} labelled {got.tolist()}, want {want_l}"
            # The duplicate's distance is not paid for either.
            ev = st.dist_evals[:, 0].cpu().numpy()
            assert (ev == len(want)).all(), \
                f"team {team}: {sorted(set(ev.tolist()))} distances for {len(want)} seeds"
            engine.advance(Xg, Gg, Qg, st, 32)
            runs.append((st.ids.clone(), st.dist.clone(), st.lineage.clone(),
                         st.visited.clone(), st.counters.clone()))
            bi, bd = engine.branch_topk(st, 10)
            engine.merge_branches(bi, bd, 10)      # the trigger the repro needed
        n = sum(0 if all(torch.equal(a, b) for a, b in zip(runs[0], o)) else 1
                for o in runs[1:])
        assert n == 0, f"team {team}: {n}/11 round-robin runs disagree"

        # Against the reference, which has always deduped by index.
        st = engine.init_states(Xg, Qg, seeds, L=L, B=32, D=G.shape[1],
                                metric=engine.METRIC_L2, team_size=team,
                                policy=engine.POLICY_ROUND_ROBIN, kstop=10)
        engine.advance(Xg, Gg, Qg, st, 32)
        gi, gd, gl = (st.ids.cpu().numpy(), st.dist.cpu().numpy(),
                      st.lineage.cpu().numpy())
        for qi in range(nq):
            ref = refcpu.run(X, G, Qv[qi], row, L=L, B=32, team=team,
                             metric=engine.METRIC_L2,
                             policy=engine.POLICY_ROUND_ROBIN)["queue"]
            assert [e[0] for e in ref] == [i for i in gi[qi, 0] if i >= 0], \
                f"team {team} q{qi}: ids differ from the reference"
            for slot, e in enumerate(ref):
                # ids and lineage exactly -- that is what is under test. The
                # distance to a tolerance: refcpu's team emulation matches the
                # kernel's L2 reduction order bit for bit at team 32 but not at
                # team 4, where 2 of these 576 queue distances land 1 ULP apart
                # (105.530182 against 105.530174). That is the reduction order
                # and not this rule, and `test_cpu_gpu_exact` does not compare
                # distances at all for the same reason.
                assert np.isclose(np.float32(e[1]), gd[qi, 0, slot], rtol=1e-6), \
                    f"team {team} q{qi} slot {slot}: {e[1]} vs {gd[qi, 0, slot]}"
                assert int(e[3]) == int(gl[qi, 0, slot]), \
                    f"team {team} q{qi} slot {slot}: lineage {gl[qi,0,slot]} " \
                    f"against the reference's {e[3]}"
    return ("duplicate seeds dedup by index, matching refcpu, reproducible over "
            "4 teams x 12 runs")


TESTS = [test_cpu_gpu_exact, test_resume_equivalence, test_budget_is_a_state_property,
         test_brute_force_on_complete_graph, test_dead_branch_and_exhaustion,
         test_visited_overflow_is_reported, test_merge_dedup, test_determinism,
         test_multiseed_matches_cpu, test_shared_state_beats_split_branches_on_work,
         test_early_stop_matches_cpu, test_shared_cache_is_free_of_consequence,
         test_degree_is_a_cost_not_an_answer, test_graph_degrees_guards_its_contract,
         test_narrow_kernel_equals_its_stored_corpus,
         test_pack_narrow_stores_offset_binary, test_narrow_contract_is_enforced,
         test_rotation_is_a_change_of_basis_only,
         test_duplicate_seeds_keep_the_lowest_lineage]

if __name__ == "__main__":
    fails = 0
    for t in TESTS:
        try:
            print(f"PASS  {t.__name__:36s} {t()}")
        except Exception as e:                                        # noqa: BLE001
            fails += 1
            print(f"FAIL  {t.__name__:36s} {type(e).__name__}: {e}")
    print(f"\n{len(TESTS) - fails}/{len(TESTS)} passed")
    sys.exit(1 if fails else 0)
