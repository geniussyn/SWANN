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
from swann import _engine as engine
from swann import _refcpu as refcpu                                   # noqa: E402

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


TESTS = [test_cpu_gpu_exact, test_resume_equivalence, test_budget_is_a_state_property,
         test_brute_force_on_complete_graph, test_dead_branch_and_exhaustion,
         test_visited_overflow_is_reported, test_merge_dedup, test_determinism,
         test_multiseed_matches_cpu, test_shared_state_beats_split_branches_on_work,
         test_early_stop_matches_cpu, test_shared_cache_is_free_of_consequence]

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
