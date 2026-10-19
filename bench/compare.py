"""Our engine against native CAGRA, at matched recall, on the same everything.

A latency number only means something next to the recall it bought, so neither
side is reported alone: each of our configurations is placed against the time
native CAGRA needs to reach *that same recall*, read off CAGRA's own
recall-latency frontier by linear interpolation between its two bracketing
points. Extrapolation is refused -- a configuration outside the measured CAGRA
range is reported as such and not scored.

Reads the result JSONs the two benches write; it runs no search of its own, so
it can be re-run after either side is remeasured.
"""
import argparse, glob, json
from pathlib import Path


def ms(r):
    """Our rows carry selection and search separately; CAGRA's carry one number.
    The comparable quantity is everything the query pays for, so selection is in."""
    return r.get("total_ms", r.get("batch_ms"))


def pareto(rows):
    """(recall, ms) kept only where no other point is both better and faster."""
    pts = sorted({(r["recall"], ms(r)) for r in rows})
    out, best = [], float("inf")
    for rec, t in sorted(pts, key=lambda p: (-p[0], p[1])):
        if t < best:
            out.append((rec, t)); best = t
    return sorted(out)


def time_at(frontier, recall):
    """Interpolated time for a recall, or None outside the measured range."""
    if recall < frontier[0][0] or recall > frontier[-1][0]:
        return None
    for (r0, t0), (r1, t1) in zip(frontier, frontier[1:]):
        if r0 <= recall <= r1:
            if r1 == r0:
                return min(t0, t1)
            return t0 + (t1 - t0) * (recall - r0) / (r1 - r0)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default="results/cagra_graph_entry_arms_m*_val.json")
    ap.add_argument("--extra-ours", default="results/cagra_graph_entry_arms_val.json")
    ap.add_argument("--native", default="results/cagra_graph_native_val.json")
    ap.add_argument("--algo", default="single_cta", help="native arm to compare against")
    a = ap.parse_args()

    ours = []
    for f in sorted(glob.glob(a.ours)) + ([a.extra_ours] if Path(a.extra_ours).exists() else []):
        j = json.loads(Path(f).read_text())
        for r in j["rows"]:
            ours.append(dict(r, m=j["m"], budget=j["total_budget"]))
    nat = json.loads(Path(a.native).read_text())
    nrows = [r for r in nat["rows"] if r["algo"] == a.algo]
    front = pareto(nrows)

    print(f"{nat['manifest']['corpus']} n={nat['manifest']['n']} graph={nat['manifest']['graph']} "
          f"q={nat['nq']} k={nat['k']}\n")
    print(f"native CAGRA ({a.algo}) frontier: " +
          "  ".join(f"{r:.4f}@{t:.1f}ms" for r, t in front) + "\n")

    print(f"{'ours':26s} {'recall@10':>10s} {'ms':>7s} {'CAGRA ms':>9s} {'speedup':>8s}")
    scored = []
    for r in sorted(ours, key=lambda x: x["recall"]):
        name = f"m={r['m']:<3d} S={r['E']:<3d} {r['arm']}"
        t, mine = time_at(front, r["recall"]), ms(r)
        if t is None:
            print(f"{name:26s} {r['recall']:10.4f} {mine:7.2f} {'--':>9s} "
                  f"{'out of range':>8s}")
            continue
        sp = t / mine
        scored.append((sp, name))
        print(f"{name:26s} {r['recall']:10.4f} {mine:7.2f} {t:9.2f} {sp:7.2f}x")

    if scored:
        best = max(scored)
        print(f"\nbest of ours: {best[1].strip()} at {best[0]:.2f}x native CAGRA "
              f"(>1 means ours is faster at the same recall)")


if __name__ == "__main__":
    main()
