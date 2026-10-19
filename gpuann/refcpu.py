"""The same local search, in Python, as the thing the GPU is checked against.

This is not an approximation of the kernel: it reproduces the kernel's
arithmetic, not only its algorithm. Each distance is summed in the lane-strided
order of a TEAM-wide team and then folded by the same halving tree, in float32,
with the multiply and the add rounded separately -- which is why the kernel is
built with -fmad=false. Agreement is therefore exact, and a test may assert
equality of distances rather than closeness, so that a real reordering bug
cannot hide under a tolerance.

Slow on purpose. It is for small graphs in tests, never for measurement.
"""
from __future__ import annotations

import numpy as np

METRIC_IP, METRIC_L2 = 0, 1
F32 = np.float32


def team_distance(q, x, metric=METRIC_IP, team=32):
    """One distance, in the kernel's reduction order and rounding."""
    D = len(q)
    lanes = []
    for lane in range(team):
        acc = F32(0.0)
        for t in range(lane, D, team):
            term = F32(q[t] * x[t]) if metric == METRIC_IP else F32(F32(q[t] - x[t]) ** 2)
            acc = F32(acc + term)
        lanes.append(acc)
    off = team // 2
    while off > 0:
        for lane in range(off):
            lanes[lane] = F32(lanes[lane] + lanes[lane + off])
        off >>= 1
    return F32(F32(1.0) - lanes[0]) if metric == METRIC_IP else lanes[0]


def _key(entry):
    return (entry[1], entry[0])          # (distance, id): the kernel's total order


POLICY_BEST, POLICY_ROUND_ROBIN = 0, 1


def init_state(X, q, seeds, m=16, max_expansions=16, metric=METRIC_IP, team=32,
               policy=POLICY_BEST):
    """One branch seeded with one or several entries. Mirrors init_states.

    Entries are (id, distance, expanded, lineage); lineage is the index of the
    seed an entry descends from, and it is what round-robin expansion keys on.
    """
    X = np.asarray(X, dtype=np.float32); q = np.asarray(q, dtype=np.float32)
    if np.isscalar(seeds) or isinstance(seeds, (int, np.integer)):
        seeds = [int(seeds)]
    cand, seen = [], set()
    for i, s in enumerate(seeds):
        s = int(s)
        if s < 0 or s in seen:
            continue
        seen.add(s)
        cand.append((s, team_distance(q, X[s], metric, team), False, i))
    cand.sort(key=_key)
    return {"queue": cand[:m], "visited": set(seen), "m": m, "metric": metric, "team": team,
            "max_expansions": max_expansions, "n_seeds": max(1, len(seeds)), "policy": policy,
            "counters": {"expansions": 0, "dist_evals": len(cand), "exhausted": 0}}


def advance(X, G, q, state, delta_expansions):
    """Mirrors advance in the kernel, including the stop rule, the expansion
    policy and the merge."""
    X = np.asarray(X, dtype=np.float32); q = np.asarray(q, dtype=np.float32)
    m, metric, team = state["m"], state["metric"], state["team"]
    S, policy = state.get("n_seeds", 1), state.get("policy", POLICY_BEST)
    queue, visited, ctr = state["queue"], state["visited"], state["counters"]

    for _ in range(delta_expansions):
        if ctr["expansions"] >= state["max_expansions"]:
            break
        slot = None
        if policy == POLICY_ROUND_ROBIN and S > 1:
            want = ctr["expansions"] % S
            slot = next((i for i, e in enumerate(queue) if not e[2] and e[3] == want), None)
        if slot is None:
            slot = next((i for i, e in enumerate(queue) if not e[2]), None)
        if slot is None:
            ctr["exhausted"] = 1
            break
        p, plin = queue[slot][0], queue[slot][3]
        queue[slot] = (p, queue[slot][1], True, plin)
        ctr["expansions"] += 1

        cand = []
        for v in G[p]:
            v = int(v)
            if v < 0 or v in visited:
                continue
            visited.add(v)
            cand.append((v, team_distance(q, X[v], metric, team), False, plin))
            ctr["dist_evals"] += 1
        cand.sort(key=_key)

        merged, a, b = [], 0, 0
        while len(merged) < m and (a < len(queue) or b < len(cand)):
            if a < len(queue) and (b >= len(cand) or _key(queue[a]) < _key(cand[b])):
                merged.append(queue[a]); a += 1
            else:
                merged.append(cand[b]); b += 1
        queue = merged
        state["queue"] = queue
    return state


def run(X, G, q, seeds, m=16, budget=16, metric=METRIC_IP, team=32, policy=POLICY_BEST):
    """init + one full-budget advance, the single-shot path."""
    st = init_state(X, q, seeds, m=m, max_expansions=budget, metric=metric, team=team,
                    policy=policy)
    return advance(X, G, q, st, budget)


def topk(state, k):
    return [(e[0], e[1]) for e in state["queue"][:k]]
