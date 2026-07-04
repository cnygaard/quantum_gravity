"""
Phase 2 -- Berry / Kähler curvature.

B = 2 Im Q from the same 6x6 kernel QGT, cross-checked by an INDEPENDENT
Wilson-loop (Fukui-Hatsugai-Suzuki) plaquette curvature, the Pfaffian cross-lock
|Pf B| = (1/8) sqrt(det G), and the Kähler relation B = 1/2 J^T G with
J = [[0,I],[-I,0]] on (b, theta).  Symbolic proof of the Kähler relation for
arbitrary L is S1 (proofs/kahler_S1.sage).

Basis: (dp_x,dp_y,dp_z,phi_x,phi_y,phi_z) = (b_x,b_y,b_z, th_x,th_y,th_z).
Anchors (p_bar=1, t=0.1):
  B_{th3 b3} = B[5,2] = +L''/2        = 9.862031
  B_perp     = B[3,0] = +L'coth/2     = 12.924840
  B_{th1th2} = B[3,4] = -L'/2         = -9.843482   (= B_{b1b2} = B[0,1])
  <p_hat>    = t/2 L'                 = 0.984348
  loop phase = -2 pi L'              = -123.696847
  |Pf B|     = L'' L'^2/(8 sinh^2 p) = 691.895 = (1/8) sqrt(det G)
"""
from mpmath import mp, mpf, cosh, sinh, pi, sqrt, fabs, matrix

from ..core.dictionary import T_BENCH
from ..core.kernel import (boost, displace, qgt, G_from_Q, B_from_Q, L_derivs,
                           berry_plaquette, det, pfaffian)
from .phase1 import edge_family
from ..harness import rec, boolrec, phase_status

PH = "Phase2"


def kahler_J():
    """J = [[0, I3],[-I3, 0]] on (b, theta):  b <- theta,  theta <- -b."""
    J = matrix(6, 6)
    for i in range(3):
        J[i, 3 + i] = 1
        J[3 + i, i] = -1
    return J


def run(pbar=mpf(1), t=T_BENCH, verbose=True):
    _, Lp, Lpp = L_derivs(pbar, t)
    g_of = edge_family(pbar, t)
    Q = qgt(g_of, 6, t)
    G = G_from_Q(Q)
    B = B_from_Q(Q)

    B_th3b3 = B[5, 2]
    B_perp = B[3, 0]
    B_th1th2 = B[3, 4]
    B_b1b2 = B[0, 1]

    c_th3b3 = Lpp / 2
    c_perp = Lp * cosh(pbar) / sinh(pbar) / 2
    c_th1th2 = -Lp / 2
    phat = t / 2 * Lp
    loop = -2 * pi * Lp

    rec("P2.B_th3b3.vs_closed", PH, B_th3b3, c_th3b3, 'NUM', note="B_th3b3 = L''/2")
    rec("P2.B_perp.vs_closed", PH, B_perp, c_perp, 'NUM', note="B_perp = L'coth/2")
    rec("P2.B_th1th2.vs_closed", PH, B_th1th2, c_th1th2, 'NUM', note="B_th1th2 = -L'/2")
    rec("P2.B_b1b2.vs_closed", PH, B_b1b2, c_th1th2, 'NUM', note="B_b1b2 = -L'/2")

    rec("P2.B_th3b3.anchor", PH, B_th3b3, mpf('9.862031'), 'T2')
    rec("P2.B_perp.anchor", PH, B_perp, mpf('12.924840'), 'T2')
    rec("P2.B_th1th2.anchor", PH, B_th1th2, mpf('-9.843482'), 'T2')
    rec("P2.phat.anchor", PH, phat, mpf('0.984348'), 'T2', note="<p> = t/2 L'")
    rec("P2.loop.anchor", PH, loop, mpf('-123.696847'), 'T2', note="loop phase -2pi L'")

    # Pfaffian cross-lock (antisymmetrize B for the Pfaffian)
    Ba = matrix(6, 6)
    for i in range(6):
        for j in range(6):
            Ba[i, j] = (B[i, j] - B[j, i]) / 2
    pf = fabs(pfaffian(Ba))
    dG = det(G)
    lock_closed = Lpp * Lp ** 2 / (8 * sinh(pbar) ** 2)
    rec("P2.pfaff.vs_closed", PH, pf, lock_closed, 'NUM',
        note="|Pf B| = L''L'^2/(8 sinh^2 p)")
    rec("P2.pfaff.vs_sqrtdetG", PH, pf, sqrt(dG) / 8, 'NUM',
        note="|Pf B| = (1/8) sqrt(det G)")
    rec("P2.pfaff.anchor", PH, pf, mpf('691.895'), 'T2')

    # B two independent ways: plaquette vs 2 Im Q
    twoway = []
    for (a, b) in [(2, 5), (0, 3), (3, 4)]:
        F = berry_plaquette(g_of, 6, a, b, delta=mpf('1e-8'), t=t)
        twoway.append((a, b, F, B[a, b]))
        rec(f"P2.B2way.{a}{b}", PH, fabs(F), fabs(B[a, b]), 'T2',
            note=f"|plaquette F_{a}{b}| = |2 Im Q_{a}{b}|")

    # Kähler relation B = 1/2 J^T G
    J = kahler_J()
    JtG = J.T * G
    kahler_err = max(fabs(B[i, j] - JtG[i, j] / 2) for i in range(6) for j in range(6))
    rec("P2.kahler.numeric", PH, kahler_err, mpf(0), 'NUM',
        note="max|B - 1/2 J^T G| over 6x6")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  B_th3b3 = {mp.nstr(B_th3b3,10)}  (L''/2   = {mp.nstr(c_th3b3,10)})")
        print(f"  B_perp  = {mp.nstr(B_perp,10)}  (L'coth/2= {mp.nstr(c_perp,10)})")
        print(f"  B_th1th2= {mp.nstr(B_th1th2,10)}  (-L'/2   = {mp.nstr(c_th1th2,10)})")
        print(f"  |Pf B|  = {mp.nstr(pf,12)}  (1/8)sqrt(detG) = {mp.nstr(sqrt(dG)/8,12)}  closed = {mp.nstr(lock_closed,12)}")
        print(f"  Kahler  max|B - 1/2 J^T G| = {float(kahler_err):.2e}")
        for a, b, F, Bab in twoway:
            print(f"  plaquette F_{a}{b} = {mp.nstr(F,8)}   2ImQ = {mp.nstr(Bab,8)}")
    return st


if __name__ == "__main__":
    run()
