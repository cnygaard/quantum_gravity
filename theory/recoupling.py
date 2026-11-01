"""
SU(2) recoupling primitives for the Thiemann Hamiltonian: Clebsch-Gordan, 3j, 6j.
=================================================================================

The exact Thiemann Hamiltonian acts on spin-network states by HOLONOMY GRASPING (coupling a
fundamental j=1/2 to an edge -> Clebsch-Gordan, the +/-1/2 spin change) and VERTEX RECOUPLING
(the 6j symbols).  The suite has no recoupling machinery, so we build it here (exact rational
sums via Python big-int factorials; only the final sqrt is floating point).  Spins and
projections are multiples of 1/2; all factorial arguments below are integers by the triangle
and projection selection rules.

Verified in __main__ against known values and the orthogonality relations.
"""
import math
from fractions import Fraction


def _fac(x):
    n = round(x)
    if abs(x - n) > 1e-9 or n < 0:                     # non-integer or negative -> invalid term
        return None                                     # (enforces the integer-perimeter selection rule)
    return math.factorial(int(n))


def _tri(a, b, c):
    """Triangle coefficient Delta(a,b,c)^2 = (a+b-c)!(a-b+c)!(-a+b+c)!/(a+b+c+1)! as a Fraction."""
    fs = [_fac(a + b - c), _fac(a - b + c), _fac(-a + b + c), _fac(a + b + c + 1)]
    if any(f is None for f in fs):
        return None
    return Fraction(fs[0] * fs[1] * fs[2], fs[3])


def cg(j1, m1, j2, m2, J, M):
    """Clebsch-Gordan <j1 m1 j2 m2 | J M> (Racah's single-sum formula)."""
    if abs(m1) > j1 + 1e-9 or abs(m2) > j2 + 1e-9 or abs(M) > J + 1e-9:
        return 0.0
    if round(2 * (m1 + m2)) != round(2 * M):
        return 0.0
    if J < abs(j1 - j2) - 1e-9 or J > j1 + j2 + 1e-9:
        return 0.0
    tri = _tri(j1, j2, J)
    if tri is None or tri == 0:
        return 0.0
    facs = [_fac(J + M), _fac(J - M), _fac(j1 - m1), _fac(j1 + m1), _fac(j2 - m2), _fac(j2 + m2)]
    if any(f is None for f in facs):
        return 0.0
    s = Fraction(0)
    for k in range(0, int(round(j1 + j2 + J)) + 2):
        args = [k, j1 + j2 - J - k, j1 - m1 - k, j2 + m2 - k, J - j2 + m1 + k, J - j1 - m2 + k]
        fs = [_fac(a) for a in args]
        if any(f is None for f in fs):
            continue
        denom = 1
        for f in fs:
            denom *= f
        s += Fraction((-1) ** k, denom)
    # numeric assembly: sqrt((2J+1) * Delta^2 * prod facs) * sum
    prodfac = facs[0] * facs[1] * facs[2] * facs[3] * facs[4] * facs[5]
    pref = math.sqrt((2 * J + 1) * float(tri) * float(prodfac))
    return pref * float(s)


def threej(j1, m1, j2, m2, j3, m3):
    """Wigner 3j symbol from the Clebsch-Gordan coefficient."""
    if round(2 * (m1 + m2 + m3)) != 0:
        return 0.0
    return (-1) ** int(round(j1 - j2 - m3)) / math.sqrt(2 * j3 + 1) * cg(j1, m1, j2, m2, j3, -m3)


def sixj(a, b, c, d, e, f):
    """Wigner 6j symbol {a b c; d e f} (Racah's single-sum formula)."""
    tris = [_tri(a, b, c), _tri(c, d, e), _tri(a, e, f), _tri(b, d, f)]
    if any(t is None or t == 0 for t in tris):
        return 0.0
    s = Fraction(0)
    lo = int(round(max(a + b + c, c + d + e, a + e + f, b + d + f)))
    hi = int(round(min(a + b + d + e, a + c + d + f, b + c + e + f)))
    for t in range(lo, hi + 1):
        args = [t - a - b - c, t - c - d - e, t - a - e - f, t - b - d - f,
                a + b + d + e - t, a + c + d + f - t, b + c + e + f - t]
        fs = [_fac(x) for x in args]
        if any(x is None for x in fs):
            continue
        ftp1 = _fac(t + 1)
        denom = 1
        for x in fs:
            denom *= x
        s += Fraction((-1) ** t * ftp1, denom)
    pref = math.sqrt(float(tris[0] * tris[1] * tris[2] * tris[3]))
    return pref * float(s)


def _check():
    ok = True

    def close(x, y, tol=1e-10):
        return abs(x - y) < tol

    # known Clebsch-Gordan values
    ok &= close(cg(0.5, 0.5, 0.5, -0.5, 0, 0), 1 / math.sqrt(2))
    ok &= close(cg(0.5, 0.5, 0.5, 0.5, 1, 1), 1.0)
    ok &= close(cg(0.5, 0.5, 0.5, -0.5, 1, 0), 1 / math.sqrt(2))
    ok &= close(cg(1, 1, 1, -1, 0, 0), 1 / math.sqrt(3))
    ok &= close(cg(1, 0, 1, 0, 2, 0), math.sqrt(2.0 / 3.0))
    # CG orthogonality: sum_{m1,m2} <j1 m1 j2 m2|J M><j1 m1 j2 m2|J' M> = delta_JJ'
    j1, j2 = 1.0, 1.5
    for J in (0.5, 1.5, 2.5):
        for Jp in (0.5, 1.5, 2.5):
            M = 0.5
            tot = 0.0
            m1 = -j1
            while m1 <= j1 + 1e-9:
                m2 = M - m1
                if abs(m2) <= j2 + 1e-9:
                    tot += cg(j1, m1, j2, m2, J, M) * cg(j1, m1, j2, m2, Jp, M)
                m1 += 1
            ok &= close(tot, 1.0 if J == Jp else 0.0)
    # 6j known value and orthogonality
    ok &= close(sixj(1, 1, 1, 1, 1, 1), 1.0 / 6.0)
    # sum_x (2x+1){a b x; c d p}{c d x; a b q} = delta_pq/(2p+1)
    a, b, c, d = 1.0, 1.0, 1.0, 1.0
    for p in (0.0, 1.0, 2.0):
        for q in (0.0, 1.0, 2.0):
            tot = 0.0
            x = 0.0
            while x <= 4.0 + 1e-9:
                tot += (2 * x + 1) * sixj(a, b, x, c, d, p) * sixj(c, d, x, a, b, q)
                x += 1
            ok &= close(tot, (1.0 / (2 * p + 1)) if p == q else 0.0)
    return ok


if __name__ == "__main__":
    print("recoupling.py self-check:", "PASS" if _check() else "FAIL")
    print(f"  cg(1/2 1/2; 1/2 -1/2 | 0 0) = {cg(0.5,0.5,0.5,-0.5,0,0):.6f}  (1/sqrt2 = {1/math.sqrt(2):.6f})")
    print(f"  6j {{1 1 1;1 1 1}} = {sixj(1,1,1,1,1,1):.6f}  (1/6 = {1/6:.6f})")
