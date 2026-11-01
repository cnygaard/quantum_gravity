"""
The VERBATIM Borissov-De Pietri-Rovelli / Gaul-Rovelli m=1 trivalent Thiemann matrix elements.
==============================================================================================

Implements, verbatim from Gaul-Rovelli gr-qc/0011106 (CQG 18, 1593), the action of Thiemann's
(m=1, Euclidean) Hamiltonian constraint on a gauge-invariant trivalent vertex |v(p,q,r)>
(colors = twice spins).  GR Appendix A evaluates the amplitudes in closed form and proves them
IDENTICAL to Borissov-De Pietri-Rovelli gr-qc/9703090 (CQG 14, 2793) -- modulo two documented
BDR errata (GR footnotes 12/13: BDR Eq. (A.10) has a wrong square root; the middle two cases of
BDR Eq. (5.9) are printed in reversed order).  So GR App. A *is* the corrected verbatim BDR.

WHAT IS IMPLEMENTED (equation numbers = GR):
  * Kauffman-Lins chromatic nets at A=-1 (App. B):  Delta_n (97), Theta(a,b,c) (98),
    Tet[a b e; c d f] (100, single-sum, exact integer arithmetic), twist lambda_c^{ab} (107).
  * The 4-valent volume element (62): Wtilde(p,q,m,c) between virtual colors t-1,t+1 -- reused
    from theory/dipole.py (certified against the first-principles J-operator grasp, +/-sqrt3/4).
    For m=1 the intertwiner space is 2-dim and the volume is diagonal (86):
    V(p,q,1,r+eps) = sqrt(w),  w = |Wtilde(p,q,1,r+eps)_{t=r+eps}|.
  * The general amplitude, Eq. (87):  Atilde^(1)(p,p+eb | q,q+et | r, r+eps) built from
    lambda * Delta/Theta * V * [TetTet/ThetaTheta - TetTet/ThetaTheta].
  * The eight closed forms i.)-viii.) of App. A (the BDR amplitudes), e.g.
      i.)  Atilde(p,p+1|q,q+1|r,r+1) =  V(r+1) * i l0 (p-q) / (12(r+1)).

VERIFICATION CHAIN (run this file):
  (1) lambda identity (89) and Delta/Theta identity (90) on a grid;
  (2) the four fundamental Tet/Theta quotients (91)-(94) on a grid  [validates Tet+Theta code];
  (3) Eq. 62 volume vs FIRST-PRINCIPLES J-operator grasp on 4-valent nodes (p,q,1,c) at several
      non-monochromatic colors  [w_Eq62 = 2 * |eig(q_hat)| uniformly: the binor color factor];
  (4) THE MAIN EVENT: Eq. (87) == closed forms i)-viii) for all admissible (p,q,r) up to color 8
      and all sign combinations  [two independent routes within the paper -> transcription and
      therefore the BDR matrix elements are reproduced verbatim].

Conventions: l0 = 1; amplitudes are returned as the REAL number multiplying (i*l0); the full
H_Delta on |v(p,q,r)> is (-1)^m (i l0 /(12 C(m))) Sum A^(m) with C(1)=1/2, i.e. -2 * Atilde
summed over the three edge pairs (Eq. 74/80-81).  Matrix elements are i * (real): the bare
Euclidean operator is anti-Hermitian-like and is symmetrized before use as a generator.
"""
import math
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dipole import W_elem                                  # noqa: E402  (GR Eq. 62; certified)


# ------------------------------------------------------------------ admissibility + nets
def adm(a, b, c):
    """KL admissibility of a color triple (App. B): parity + triangle inequalities."""
    if (a + b + c) % 2:
        return False
    return a + b - c >= 0 and b + c - a >= 0 and c + a - b >= 0


def Delta(n):
    """Symmetrizer loop value, Eq. (97):  Delta_n = (-1)^n (n+1)."""
    return Fraction((-1) ** n * (n + 1))


def Theta(a, b, c):
    """Theta net, Eq. (98):  (-1)^{m+n+p} (m+n+p+1)! m! n! p! / (a! b! c!)."""
    if not adm(a, b, c):
        return Fraction(0)
    m = (a + b - c) // 2
    n = (b + c - a) // 2
    p = (c + a - b) // 2
    num = (-1) ** (m + n + p) * math.factorial(m + n + p + 1) * \
        math.factorial(m) * math.factorial(n) * math.factorial(p)
    den = math.factorial(a) * math.factorial(b) * math.factorial(c)
    return Fraction(num, den)


def Tet(a, b, e, c, d, f):
    """Tetrahedral net Tet[a b e; c d f], Eq. (100) verbatim (single sum, exact)."""
    # vertex triples (from the tetrahedral symmetry set): {a,e,d},{b,a,f},{f,d,c},{c,b,e}
    for tri in ((a, e, d), (b, a, f), (f, d, c), (c, b, e)):
        if not adm(*tri):
            return Fraction(0)
    a1 = (a + d + e) // 2
    a2 = (b + c + e) // 2
    a3 = (a + b + f) // 2
    a4 = (c + d + f) // 2
    b1 = (b + d + e + f) // 2
    b2 = (a + c + e + f) // 2
    b3 = (a + b + c + d) // 2
    E = math.factorial(a) * math.factorial(b) * math.factorial(c) * \
        math.factorial(d) * math.factorial(e) * math.factorial(f)
    I = 1
    for bj in (b1, b2, b3):
        for ai in (a1, a2, a3, a4):
            I *= math.factorial(bj - ai)
    s_lo, s_hi = max(a1, a2, a3, a4), min(b1, b2, b3)
    S = Fraction(0)
    for s in range(s_lo, s_hi + 1):
        den = 1
        for ai in (a1, a2, a3, a4):
            den *= math.factorial(s - ai)
        for bj in (b1, b2, b3):
            den *= math.factorial(bj - s)
        S += Fraction((-1) ** s * math.factorial(s + 1), den)
    return Fraction(I, E) * S


def lam(a, b, c):
    """Twist factor lambda_c^{ab}, Eq. (107): (-1)^{(a'+b'-c')/2}, x' = x(x+3)."""
    ap, bp, cp = a * (a + 3), b * (b + 3), c * (c + 3)
    return (-1) ** ((ap + bp - cp) // 2)


# ------------------------------------------------------------------ volume (Eq. 62/83-86)
def w_vol(p, q, c):
    """w(p,q,1,c) = |Wtilde^(4)(p,q,1,c)_{t=c}| (Eq. 83, via Eq. 62 with alpha=c-1,beta=c+1)."""
    if not adm(p, q, c - 1) or not adm(c, 1, c - 1) or not adm(p, q, c + 1) or not adm(c, 1, c + 1):
        # 2-dim virtual space {c-1,c+1} requires both admissible; else volume element absent
        return 0.0
    return abs(W_elem(p, q, 1, c, c - 1))


def V_vol(p, q, c):
    """Diagonal m=1 volume element, Eq. (86): V(p,q,1,c) = sqrt(w)."""
    return math.sqrt(w_vol(p, q, c))


# ------------------------------------------------------------------ the amplitudes
def A_general(p, q, r, eb, et, eps):
    """One eps-term of Eq. (87), verbatim.  Returns the REAL coefficient of (i l0):
    Atilde^(1)(p,p+eb | q,q+et | r,r+eps) / (i l0)."""
    a, b, c = p + eb, q + et, r + eps
    if min(a, b, c) < 0:
        return 0.0
    for tri in ((p, 1, a), (q, 1, b), (r, 1, c)):
        if not adm(*tri):
            return 0.0
    pref = Fraction(lam(1, p, a) * lam(1, q, b) * lam(1, r, c)) * \
        Delta(a) * Delta(b) * Delta(c) / (Theta(p, 1, a) * Theta(q, 1, b) * Theta(r, 1, c))
    th_rab = Theta(r, a, b)
    if th_rab == 0:
        return 0.0
    t1 = t2 = Fraction(0)
    th_qca = Theta(q, c, a)
    if th_qca != 0:
        t1 = Tet(a, q, p, r, 1, c) * Tet(a, b, q, 1, c, r) / (th_qca * th_rab)
    th_pcb = Theta(p, c, b)
    if th_pcb != 0:
        t2 = Tet(p, b, q, 1, r, c) * Tet(a, b, p, c, 1, r) / (th_pcb * th_rab)
    return float(pref * (t1 - t2)) * V_vol(p, q, c) / 12.0


def A_closed(p, q, r, eb, et, eps):
    """GR App. A closed forms i)-viii) (= the corrected-verbatim BDR amplitudes).
    Returns the REAL coefficient of (i l0)."""
    a, b, c = p + eb, q + et, r + eps
    if min(a, b, c) < 0 or not (adm(p, 1, a) and adm(q, 1, b) and adm(r, 1, c)):
        return 0.0
    V = V_vol(p, q, c)
    if eps == +1:
        if eb == +1 and et == +1:      # i.
            x = Fraction(p - q, 12 * (r + 1))
        elif eb == -1 and et == +1:    # ii.
            x = Fraction((p - q + r) * (2 + p + q), 24 * (r + 1) * (p + 1))
        elif eb == +1 and et == -1:    # iii.
            x = -Fraction((q - p + r) * (2 + p + q), 24 * (r + 1) * (q + 1))
        else:                          # iv.
            x = Fraction((p + q - r) * (p - q) * (2 + p + q + r),
                         48 * (r + 1) * (p + 1) * (q + 1))
    else:
        if eb == +1 and et == +1:      # v.
            x = Fraction(q - p, 12 * (r + 1))
        elif eb == -1 and et == +1:    # vi.
            x = -Fraction((p - q + r) * (2 + p + q), 24 * (r + 1) * (p + 1))
        elif eb == +1 and et == -1:    # vii.
            x = Fraction((q - p + r) * (2 + p + q), 24 * (r + 1) * (q + 1))
        else:                          # viii.
            x = Fraction((p + q - r) * (q - p) * (2 + p + q + r),
                         48 * (r + 1) * (p + 1) * (q + 1))
    return float(x) * V


def A_total(p, q, r, eb, et):
    """Physical amplitude for new inner colors (p+eb, q+et): internal-color sum over eps
    (Eq. 88; the state does not depend on c).  Coefficient of (i l0)."""
    return A_closed(p, q, r, eb, et, +1) + A_closed(p, q, r, eb, et, -1)


# ------------------------------------------------------------------ verification
def _check_lam_delta():
    ok = True
    for n in range(1, 12):
        ok &= lam(1, n, n + 1) == (-1) ** n                                  # (89) sigma=+1
        if n >= 1:
            ok &= lam(1, n, n - 1) == (-1) ** (n + 1)                        # (89) sigma=-1
            ok &= Delta(n - 1) / Theta(n, n - 1, 1) == Fraction(-n, n + 1)   # (90) sigma=-1
        ok &= Delta(n + 1) / Theta(n, n + 1, 1) == 1                         # (90) sigma=+1
    return ok


def _check_tet_quotients():
    """GR Eqs. (91)-(94): the four fundamental Tet/Theta quotients, on a grid."""
    ok = True
    for aa in range(1, 8):
        for bb in range(1, 8):
            for cc in range(1, 8):
                if not adm(aa, bb, cc):
                    continue
                for tau in (-1, 0, 1):
                    t2 = tau * tau
                    # the identities presuppose the Tet's 4th vertex triple {C,B,E} admissible
                    # (91)
                    th = Theta(aa + 1, bb + tau, cc + 1 - t2)
                    if th != 0 and cc - t2 >= 0 and adm(cc - t2, bb + tau, aa):
                        ok &= (Tet(aa + 1, bb + tau, aa, cc - t2, 1, cc + 1 - t2) / th == 1)
                    # (92)
                    th = Theta(aa - 1, bb + tau, cc + 1 - t2)
                    if th != 0 and cc - t2 >= 0 and adm(cc - t2, bb + tau, aa):
                        ok &= (Tet(aa - 1, bb + tau, aa, cc - t2, 1, cc + 1 - t2) / th ==
                               Fraction(tau * (tau + 1) + aa + bb - cc, 2 * aa))
                    # (94)
                    th = Theta(aa - 1, bb + tau, cc - 1 + t2)
                    if th != 0 and cc - 1 + t2 >= 0 and adm(cc + t2, bb + tau, aa):
                        ok &= (Tet(aa - 1, bb + tau, aa, cc + t2, 1, cc - 1 + t2) / th ==
                               -Fraction((tau * (tau - 1) + aa - bb + cc) *
                                         (tau * (tau + 1) + 2 + aa + bb + cc),
                                         4 * aa * (t2 + cc)))
                for sig in (-1, 1):
                    # (93)
                    th = Theta(aa + 1, bb + sig, cc)
                    if th != 0 and adm(cc + 1, bb + sig, aa):
                        ok &= (Tet(aa + 1, bb + sig, aa, cc + 1, 1, cc) / th ==
                               Fraction(sig + 1 + bb - aa + cc, 2 + 2 * cc))
    return ok


def _gens_general(spins):
    """Embedded (Jx,Jy,Jz) per site for arbitrary spins (self-contained, numpy)."""
    import numpy as np
    dims = [int(round(2 * j + 1)) for j in spins]

    def gen(j):
        d = int(round(2 * j + 1))
        m = np.array([j - k for k in range(d)], float)
        Jz = np.diag(m).astype(complex)
        Jp = np.zeros((d, d), complex)
        for k in range(1, d):
            Jp[k - 1, k] = np.sqrt((j - m[k]) * (j + m[k] + 1))
        Jm = Jp.conj().T
        return (Jp + Jm) / 2, (Jp - Jm) / 2j, Jz

    def emb(op, a):
        M = np.array([[1.0 + 0j]])
        for k, d in enumerate(dims):
            M = np.kron(M, op if k == a else np.eye(d, dtype=complex))
        return M

    return [tuple(emb(g, a) for g in gen(j)) for a, j in enumerate(spins)]


def _check_volume_first_principles():
    """Eq. 62 magnitude vs first-principles J-operator grasp: w = 2*|eig(q_hat)| uniformly."""
    import numpy as np
    from thiemann import triple, singlet_projector
    out = []
    for (p, q, c) in [(1, 1, 1), (2, 1, 2), (2, 2, 1), (3, 2, 2), (2, 2, 3)]:
        J = _gens_general([p / 2, q / 2, 0.5, c / 2])      # edges p,q,m=1,c
        P = singlet_projector(J, 4)
        if P.shape[1] != 2:
            continue
        g = P.conj().T @ triple(J, 0, 1, 3) @ P            # grasp the p,q,c edges (W_[pqc])
        v = float(np.max(np.abs(np.linalg.eigvalsh((g + g.conj().T) / 2))))
        out.append((p, q, c, w_vol(p, q, c), 2 * v))
    ok = all(abs(w - tv) < 1e-12 for *_, w, tv in out)
    return ok, out


def _check_closed_vs_general(cmax=8):
    """THE MAIN EVENT: Eq. (87) == App. A closed forms on the full admissible grid."""
    worst = 0.0
    n = 0
    for p in range(1, cmax + 1):
        for q in range(1, cmax + 1):
            for r in range(1, cmax + 1):
                if not adm(p, q, r):
                    continue
                for eb in (1, -1):
                    for et in (1, -1):
                        for eps in (1, -1):
                            g = A_general(p, q, r, eb, et, eps)
                            cl = A_closed(p, q, r, eb, et, eps)
                            worst = max(worst, abs(g - cl))
                            n += 1
    return worst, n


if __name__ == "__main__":
    print("=" * 76)
    print("VERBATIM BDR / GAUL-ROVELLI m=1 TRIVALENT MATRIX ELEMENTS -- verification")
    print("=" * 76)
    print(f"  [1] lambda (89) + Delta/Theta (90) identities: "
          f"{'PASS' if _check_lam_delta() else 'FAIL'}")
    print(f"  [2] Tet/Theta quotients (91)-(94) on grid:     "
          f"{'PASS' if _check_tet_quotients() else 'FAIL'}")
    okv, tab = _check_volume_first_principles()
    print(f"  [3] Eq.62 volume vs first-principles J-grasp:  {'PASS' if okv else 'FAIL'}"
          f"   (w = 2|eig q_hat|, binor color factor)")
    for p, q, c, w, tv in tab:
        print(f"        (p,q,1,c)=({p},{q},1,{c}):  w_Eq62={w:.6f}   2|eig|={tv:.6f}")
    worst, n = _check_closed_vs_general()
    print(f"  [4] MAIN: Eq.(87) general == App.A closed (BDR), {n} amplitudes, colors<=8:")
    print(f"        max |difference| = {worst:.3e}   -> "
          f"{'VERBATIM REPRODUCED' if worst < 1e-12 else 'MISMATCH'}")
    print(f"\n  sample: A~(2,3|2,3|2,3) = i*l0 * {A_closed(2,2,2,1,1,1):+.6f}"
          f"  (i., p=q -> 0);  A~(2,1|2,3|2,3) = i*l0 * {A_closed(2,2,2,-1,1,1):+.6f}  (ii.)")
    print(f"  total (eps-summed) A(2,2->1,3|2.) = i*l0 * {A_total(2,2,2,-1,1):+.6f}")
