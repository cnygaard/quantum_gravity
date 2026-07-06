"""
Large-D_int anchor for the LQG volume-chaos result (SLOW: ~19 min, N0=15722).

Reproduces the biggest tractable single node -- a 7-valent vertex with distinct spins
[1, 1.5, 2, 2.5, 3, 3.5, 4.5] -- giving the intertwiner dimension D_int=276 (138 positive
volume levels).  This is the statistically cleanest Tier-1 point: r~ = 0.617, above the
Poisson value 0.386 and at/above the chaotic (antisymmetric-random) value ~0.594 -- the
finite-size trend of the faster D_int = 49, 75, 95 cases (see scrambling.py) confirmed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scrambling as S            # noqa: E402

S.MAXN0 = 20000                   # allow the N0=15722 build


def main():
    spins = [1, 1.5, 2, 2.5, 3, 3.5, 4.5]
    print(f"7-valent node {spins}  (this takes ~19 min)...", flush=True)
    lv, D_int, N0 = S.volume_spectrum(spins)
    rb = S.rbar(lv)
    anti = S._antisym_rbar()
    print(f"  N0={N0}  D_int={D_int}  positive levels={lv.size}")
    print(f"  r~ = {rb:.4f}   (Poisson 0.386 | chaotic ~{anti:.3f})  -> "
          f"{'CHAOTIC' if rb > 0.5 else 'not chaotic'}")


if __name__ == "__main__":
    main()
