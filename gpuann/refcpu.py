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


def init_state(X, q, seed, m=16, max_expansions=16, metric=METRIC_IP, team=32):
    """One branch, one seed. Mirrors init_states in the kernel."""
    X = np.asarray(X, dtype=np.float32); q = np.asarray(q, dtype=np.float32)
    if seed < 0:
        return {"queue": [], "visited": set(), "m": m, "max_expansions": max_expansions,
                "metric": metric, "team": team,
                "counters": {"expansions": 0, "dist_evals": 0, "exhausted": 0}}
    d0 = team_distance(q, X[seed], metric, team)
    return {"queue": [(int(seed), d0, False)], "visited": {int(seed)},
            "m": m, "max_expansions": max_expansions, "metric": metric, "team": team,
            "counters": {"expansions": 0, "dist_evals": 1, "exhausted": 0}}


def advance(X, G, q, state, delta_expansions):
    """Mirrors advance in the kernel, including the stop rule and the merge."""
    X = np.asarray(X, dtype=np.float32); q = np.asarray(q, dtype=np.float32)
    m, metric, team = state["m"], state["metric"], state["team"]
    queue, visited, ctr = state["queue"], state["visited"], state["counters"]

    for _ in range(delta_expansions):
        if ctr["expansions"] >= state["max_expansions"]:
            break
        slot = next((i for i, e in enumerate(queue) if not e[2]), None)
        if slot is None:
            ctr["exhausted"] = 1
            break
        p = queue[slot][0]
        queue[slot] = (p, queue[slot][1], True)
        ctr["expansions"] += 1

        cand = []
        for v in G[p]:
            v = int(v)
            if v < 0 or v in visited:
                continue
            visited.add(v)
            cand.append((v, team_distance(q, X[v], metric, team), False))
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


def run(X, G, q, seed, m=16, budget=16, metric=METRIC_IP, team=32):
    """init + one full-budget advance, the single-shot path."""
    st = init_state(X, q, seed, m=m, max_expansions=budget, metric=metric, team=team)
    return advance(X, G, q, st, budget)


def topk(state, k):
    return [(e[0], e[1]) for e in state["queue"][:k]]
