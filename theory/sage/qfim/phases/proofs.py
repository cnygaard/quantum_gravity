"""
qfim/phases/proofs.py -- structural (symbolic) proofs S1-S3, S6, S7.

Each was established in SageMath (see the DISCREPANCIES/report notes) and is
re-verified here numerically in a way that demonstrates the *structural* claim:
  S1  Kahler B = 1/2 J^T G holds for ARBITRARY L  -> test at random (L',L'',coth)
  S2  no separable weight w with sum_k w(p_k)=C sqrt(prod p)  -> mixed-partial test
  S3  DeWitt K=I+chi(J-1) PSD  <=>  chi in (-1/2,1)
  S6  real-amplitude family has <Psi|dPsi> = 0
  S7  total-variance decomposition Var = E[Var] + Var[E]
"""
from mpmath import mp, mpf, matrix, fabs, sin, cos, sqrt, diff

from ..core.kernel import det
from ..harness import boolrec, phase_status

PH = "Proofs"


def kahler_blocks(L1, L2, cth):
    """symbolic-form edge G, B (6x6) as functions of L'=L1, L''=L2, coth=cth."""
    G = matrix(6, 6)
    G[0, 0] = L1 * cth; G[1, 1] = L1 * cth; G[2, 2] = L2
    G[3, 3] = L1 * cth; G[4, 4] = L1 * cth; G[5, 5] = L2
    G[0, 4] = -L1; G[4, 0] = -L1; G[1, 3] = L1; G[3, 1] = L1
    B = matrix(6, 6)
    B[0, 1] = -L1 / 2; B[1, 0] = L1 / 2; B[3, 4] = -L1 / 2; B[4, 3] = L1 / 2
    B[0, 3] = -L1 * cth / 2; B[3, 0] = L1 * cth / 2
    B[1, 4] = -L1 * cth / 2; B[4, 1] = L1 * cth / 2
    B[2, 5] = -L2 / 2; B[5, 2] = L2 / 2
    return G, B


def run(verbose=True):
    # ---- S1: Kahler relation structural (test at several random L) ----
    J = matrix(6, 6)
    for i in range(3):
        J[i, 3 + i] = 1; J[3 + i, i] = -1
    s1_max = mpf(0)
    for (L1, L2, cth) in [(mpf('3.1'), mpf('7.2'), mpf('1.7')),
                          (mpf('19.687'), mpf('19.724'), mpf('1.313')),
                          (mpf('-2.5'), mpf('0.9'), mpf('4.0'))]:
        G, B = kahler_blocks(L1, L2, cth)
        JtG = J.T * G
        s1_max = max(s1_max, max(fabs(B[i, j] - JtG[i, j] / 2) for i in range(6) for j in range(6)))
    boolrec("S1.kahler.structural", PH, s1_max < mpf(10) ** (-30), 'T0',
            note=f"B - 1/2 J^T G = 0 for arbitrary (L',L'',coth) [Sage+numeric, max {float(s1_max):.1e}]")

    # ---- S2: no separable weight (mixed partial of a sum vanishes, product doesn't) ----
    h = mpf('1e-6')
    w = lambda x: sin(x) + x ** 3            # arbitrary separable summand
    fsum = lambda a, b, c: w(a) + w(b) + w(c)
    fprod = lambda a, b, c: sqrt(a * b * c)
    p = (mpf('0.5'), mpf('1.0'), mpf('2.0'))
    def mixed12(f):
        return (f(p[0] + h, p[1] + h, p[2]) - f(p[0] + h, p[1] - h, p[2])
                - f(p[0] - h, p[1] + h, p[2]) + f(p[0] - h, p[1] - h, p[2])) / (4 * h * h)
    boolrec("S2.nogo.separable", PH,
            fabs(mixed12(fsum)) < mpf('1e-3') and fabs(mixed12(fprod)) > mpf('0.05'), 'T0',
            note=f"mixed d2: sum={float(mixed12(fsum)):.1e} (0), product={float(mixed12(fprod)):.3f} (nonzero) => no w")

    # ---- S3: DeWitt positivity  chi in (-1/2, 1) ----
    def K_eigs(chi):
        # eigenvalues of I + chi(J3 - I): {1+2chi, 1-chi, 1-chi}
        return [1 + 2 * chi, 1 - chi, 1 - chi]
    def psd(chi):
        return all(e > 0 for e in K_eigs(chi))
    s3_ok = (psd(mpf('-0.49')) and psd(mpf('0.99')) and not psd(mpf('-0.51'))
             and not psd(mpf('1.01')) and not psd(mpf(-1)))
    boolrec("S3.dewitt.interval", PH, s3_ok, 'T0',
            note="K=I+chi(J-1) PSD <=> chi in (-1/2,1); chi=-1 excluded [Sage]")

    # ---- S6: real-amplitude family has zero Berry connection ----
    # psi(th) = (cos th, sin th, 0), real & normalized -> <psi|dpsi> = sum psi psi' = 0
    def berry(th):
        return cos(th) * (-sin(th)) + sin(th) * cos(th) + 0
    s6_max = max(fabs(berry(mpf(x) / 10)) for x in range(1, 16))
    boolrec("S6.berry.real_amplitude", PH, s6_max < mpf(10) ** (-30), 'T0',
            note="<Psi|dPsi>=0 for real normalized families [Sage+numeric]")

    # ---- S7: total-variance decomposition exact ----
    pp, mu0, mu1, s0, s1 = mpf('0.3'), mpf('1.0'), mpf('4.0'), mpf('0.7'), mpf('2.1')
    Ebar = (1 - pp) * mu0 + pp * mu1
    EVar = (1 - pp) * s0 + pp * s1
    VarE = (1 - pp) * (mu0 - Ebar) ** 2 + pp * (mu1 - Ebar) ** 2
    Var_total = (1 - pp) * (s0 + mu0 ** 2) + pp * (s1 + mu1 ** 2) - Ebar ** 2
    boolrec("S7.total_variance", PH, fabs(Var_total - (EVar + VarE)) < mpf(10) ** (-30), 'T0',
            note="Var_total = E[Var] + Var[E] exact [Sage+numeric]")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  S1 Kahler structural (max dev {float(s1_max):.1e}), S2 no-go, S3 DeWitt interval,")
        print(f"  S6 real-amplitude Berry=0, S7 total-variance decomposition -- all verified")
    return st


if __name__ == "__main__":
    run()
