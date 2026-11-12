"""Is the dense catalogue's extra recall real, or is it the size of the noise?

With the stopping rule removed the sparse catalogue saturates: on all three
corpora, relaxing eps from 0.30 to unbounded moves recall by less than 0.001, so
whatever is still missing is not a budget problem. The dense catalogue sits above
that plateau -- but by 0.0006 on msmarco, which is twelve answers out of 20,000,
and on GIST by six answers out of 5,000. Treated as two independent samples those
are nothing.

They are not independent samples. Both arms ran the same queries on the same
graph, so the comparison is paired, and the question to ask of a pair is which
way it moved. A sign test over the queries that disagree is the right statistic
and it is far sharper than the means suggest.
"""
import argparse, json
from pathlib import Path
from math import comb

import numpy as np


def sign_test(better, worse):
    """Two-sided exact probability that this many queries would split this way
    if the catalogue made no difference and each disagreeing query were a coin."""
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    tail = sum(comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("results/ceiling"))
    ap.add_argument("--corpora", nargs="+",
                    default=["msmarco-1m", "gist-960", "glove-100"])
    a = ap.parse_args()

    print(f"{'corpus':12} {'queries':>8} {'sparse':>8} {'dense':>8} {'delta':>8} "
          f"{'better':>7} {'worse':>6} {'same':>6} {'p':>9}")
    out = []
    for ds in a.corpora:
        sp = a.dir / f"{ds}_nest4k.npz"
        dn = a.dir / f"{ds}_nest128k.npz"
        if not (sp.exists() and dn.exists()):
            print(f"{ds:12} {'(missing)':>8}")
            continue
        s = np.load(sp)["perq"][0]
        d = np.load(dn)["perq"][0]
        assert s.shape == d.shape, f"{ds}: {s.shape} vs {d.shape}"
        better = int((d > s).sum()); worse = int((d < s).sum())
        same = int((d == s).sum())
        p = sign_test(better, worse)
        print(f"{ds:12} {len(s):8,} {s.mean():8.4f} {d.mean():8.4f} "
              f"{d.mean()-s.mean():+8.4f} {better:7,} {worse:6,} {same:6,} {p:9.2e}")
        out.append(dict(corpus=ds, n=len(s), sparse=float(s.mean()),
                        dense=float(d.mean()), delta=float(d.mean() - s.mean()),
                        better=better, worse=worse, same=same, p=float(p)))
    (a.dir / "paired.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.dir / 'paired.json'}")
    print("'better' counts queries the dense catalogue answered more completely;\n"
          "p is a two-sided sign test over the queries that disagree, which is\n"
          "the only set that carries information about the direction.")


if __name__ == "__main__":
    main()
