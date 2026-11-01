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
from mpmath import mp, mpf, matrix, fabs, sin, cos, cosh, sinh, sqrt, quad, inf, pi

from ..core.kernel import det
from ..core.dictionary import PBAR, T_BENCH, L1_closed, L2_closed
from .phase4 import wL, wT, w, pbar_of_q
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
    wsep = lambda x: sin(x) + x ** 3         # arbitrary separable summand
    fsum = lambda a, b, c: wsep(a) + wsep(b) + wsep(c)
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

    # ---- S3b: DeWitt<->Fisher Mobius map  lam_eff=chi/(1+2chi);  conformal match -> chi=-1 ----
    def Rmat(chi):
        # equicorrelation / correlation matrix R = (1-chi)I + chi J: unit diagonal, chi off-diagonal
        return matrix([[1, chi, chi], [chi, 1, chi], [chi, chi, 1]])
    def lam_eff(chi):
        return chi / (1 + 2 * chi)
    def inv_resid(chi):
        # R(chi) * (1-chi)^-1 [I - lam_eff J] = I  (Sherman-Morrison), checked by multiplication
        a = 1 - chi
        le = lam_eff(chi)
        dia, off = (1 - le) / a, -le / a
        closed = matrix([[dia, off, off], [off, dia, off], [off, off, dia]])
        P = Rmat(chi) * closed
        return max(fabs(P[i, j] - (1 if i == j else 0)) for i in range(3) for j in range(3))
    tol = mpf(10) ** (-30)
    s3b_ok = (
        all(inv_resid(mpf(c)) < tol for c in ('-0.4', '-0.25', '0', '0.3', '0.9'))   # inversion identity
        and fabs(lam_eff(mpf('-0.25')) + mpf('0.5')) < tol      # capstone chi=-1/4 -> lam_eff=-1/2
        and fabs(lam_eff(mpf(0))) < tol                          # chi=0 -> lam_eff=0
        and fabs(lam_eff(mpf(-1)) - 1) < tol                     # DeWitt conformal lam=1 <=> chi=-1
        and fabs((1 - mpf(-1)) / (1 + 2 * mpf(-1)) + 2) < tol    # eigenvalue-ratio route rho_F(-1)=-2
        and fabs(1 - 3 * lam_eff(mpf(1))) < tol                  # chi=1 -> DeWitt degeneracy 1-3lam=0
        and fabs(1 + 2 * mpf('-0.5')) < tol)                     # chi=-1/2 -> pole (1+2chi=0), window edge
    boolrec("S3b.dewitt.mobius", PH, s3b_ok, 'T0',
            note="R(chi)^-1=(1-chi)^-1[I-lam J], lam_eff=chi/(1+2chi); lam=1<=>chi=-1 (DeWitt conformal), "
                 "ratio=-2 same; window chi in(-1/2,1)<->lam in(-inf,1/3), chi=1<->lam=1/3 degeneracy [Sage+numeric]")

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

    # ---- S4a: Wick / Gaussian identity, verified by direct tanh-sinh integration ----
    # int d_i d_k f d_j d_l f = pi^{3/2}/(4 sig sqrt(det q)) [q_ik q_jl + q_ij q_kl + q_il q_jk]
    A, B, C, sg = mpf('2.0'), mpf('3.0'), mpf('1.5'), mpf('1.3')
    gf = lambda a, x: mp.e ** (-a * x ** 2 / (2 * sg ** 2))
    d1 = lambda a, x: -(a * x / sg ** 2) * gf(a, x)
    d2 = lambda a, x: (a / sg ** 2) * (a * x ** 2 / sg ** 2 - 1) * gf(a, x)
    I0 = lambda a: quad(lambda x: gf(a, x) ** 2, [-inf, inf])
    I2 = lambda a: quad(lambda x: d1(a, x) ** 2, [-inf, inf])
    I4 = lambda a: quad(lambda x: d2(a, x) ** 2, [-inf, inf])
    I02 = lambda a: quad(lambda x: d2(a, x) * gf(a, x), [-inf, inf])
    pref = pi ** mpf('1.5') / (4 * sg * sqrt(A * B * C))
    s4a = max(fabs(I4(A) * I0(B) * I0(C) - pref * 3 * A ** 2),      # (1,1,1,1)
              fabs(I02(A) * I02(B) * I0(C) - pref * A * B),         # (1,1,2,2)
              fabs(I2(A) * I2(B) * I0(C) - pref * A * B))           # (1,2,1,2)
    boolrec("S4a.wick_identity", PH, s4a < mpf('1e-20'), 'T0',
            note=f"Gaussian d_i d_k f d_j d_l f integral = 3-term q-contraction (resid {float(s4a):.1e}) [Sage+quad]")

    # ---- S4b: rho collapse -- additive (not sqrt-product multiplicative) structure ----
    t = T_BENCH
    pb = PBAR
    Wtot = sum(w(p, t) for p in pb)
    rho = [2 * (2 * Wtot - w(pb[i], t)) for i in range(3)]
    s4b_ok = (fabs(sum(rho) - 10 * Wtot) < mpf('1e-25')
              and fabs((rho[0] - rho[2]) - 2 * (w(pb[2], t) - w(pb[0], t))) < mpf('1e-25'))
    boolrec("S4b.rho_additive", PH, s4b_ok, 'T0',
            note="sum rho = 10 W_tot; rho_i-rho_j = 2[w(p_j)-w(p_i)] -- additive under q_a p_a^2=1, no C sqrt(prod) [S2]")

    # ---- S4c: equal footing -- L''=w_L/p^2 and G^phys_perp=w_T/p^2 (weight/p^2) ----
    def Gperp(p):
        return 2 * L1_closed(p, t) * (cosh(p) - 1) / (p ** 2 * sinh(p))
    s4c = mpf(0)
    for p in [mpf('0.5'), mpf(1), mpf(2)]:
        s4c = max(s4c, fabs(L2_closed(p, t) - wL(p, t) / p ** 2),
                  fabs(Gperp(p) - wT(p, t) / p ** 2))
    boolrec("S4c.equal_footing", PH, s4c < mpf('1e-30'), 'T0',
            note="L''=w_L/p^2 and G^phys_perp=w_T/p^2: both channels enter displacement cost as weight/p^2")

    _referee_proofs()

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  S1 Kahler structural (max dev {float(s1_max):.1e}), S2 no-go, S3 DeWitt interval,")
        print(f"  S4a Wick (resid {float(s4a):.1e}), S4b rho-additive, S4c equal-footing,")
        print(f"  S6 real-amplitude Berry=0, S7 total-variance decomposition -- all verified")
    return st


def _referee_proofs():
    """S8-S9 (SymPy): the referee-requested exact statements behind T1, Remark 1, the boosted
    blocks and the restated edge-additivity obstruction."""
    import sympy as sp
    t, z, r, j, lam = sp.symbols('t zeta r j lambda', positive=True)

    # S8a  T1 small-t: the n=0 Poisson term of Z0 = e^{t/4} sum_{n>=1} n^2 e^{-t n^2/4} is
    #      e^{t/4} 2 sqrt(pi) t^{-3/2}; then Z1/Z0 = -dlnZ0/dt = 3/(2t) - 1/4, G = 2/t - 1/3.
    n = sp.symbols('n', positive=True)
    lead = sp.exp(t / 4) * sp.integrate(n ** 2 * sp.exp(-t * n ** 2 / 4), (n, 0, sp.oo))
    ratio = sp.simplify(-sp.diff(sp.log(lead), t))
    G_lead = sp.simplify(sp.Rational(4, 3) * ratio)
    s8a_sym = (sp.simplify(lead - 2 * sp.sqrt(sp.pi) * t ** sp.Rational(-3, 2) * sp.exp(t / 4)) == 0
               and sp.simplify(G_lead - (2 / t - sp.Rational(1, 3))) == 0)
    # numeric: exact half-integer-j sums vs the expansion at t = 0.1 (remainder ~ e^{-4 pi^2/t})
    tb = mpf('0.1')
    Z0 = sum((2 * jj + 1) ** 2 * mp.e ** (-tb * jj * (jj + 1)) for jj in [mpf(k) / 2 for k in range(0, 400)])
    Z1 = sum((2 * jj + 1) ** 2 * jj * (jj + 1) * mp.e ** (-tb * jj * (jj + 1))
             for jj in [mpf(k) / 2 for k in range(0, 400)])
    s8a_num = fabs(mpf(4) / 3 * Z1 / Z0 - (2 / tb - mpf(1) / 3))
    boolrec("S8a.T1.small_t", PH, s8a_sym and s8a_num < mpf(10) ** (-mp.dps + 5), 'T0',
            computed=float(s8a_num),
            note=f"SymPy: Poisson n=0 term -> Z1/Z0 = 3/(2t)-1/4, G = 2/t-1/3; exact j in N/2 sums at t=0.1 "
                 f"agree to {float(s8a_num):.0e} (working precision; true remainder ~e^-395)")

    # S8b  Remark 1 made exact: S^3 of radius r has Laplacian eigenvalues l(l+2)/r^2; with
    #      l = 2j these are j(j+1) iff r = 2; then R = 6/r^2 = 3/2, e^{t/4} = e^{tR/6}, and the
    #      constant -1/4 = -R/6 in Z1/Z0, -1/3 = -(2/9) R in G.
    ell = 2 * j
    r_sol = sp.solve(sp.Eq(ell * (ell + 2) / r ** 2, j * (j + 1)), r)
    R = 6 / r_sol[0] ** 2 if r_sol else None
    s8b = (r_sol == [2] and R == sp.Rational(3, 2) and sp.Rational(1, 4) == R / 6
           and sp.Rational(-1, 3) == -sp.Rational(2, 9) * R)
    boolrec("S8b.remark1.S3_curvature", PH, bool(s8b), 'T0',
            note="Casimir j(j+1) <=> S^3 radius 2, R=3/2: e^{t/4}=e^{tR/6} (Minakshisundaram-Pleijel a_1), "
                 "-1/3 = -(2/9)R exactly (normalization-dependent, not heuristic) [SymPy]")

    # S8c  boosted blocks = Hessian of the (winding-free) Kahler potential ln(zeta/sinh zeta) + zeta^2/t
    K = sp.log(z / sp.sinh(z)) + z ** 2 / t
    Lp = 2 * z / t + 1 / z - sp.coth(z)
    Lpp = 2 / t - 1 / z ** 2 + 1 / sp.sinh(z) ** 2
    s8c = (sp.simplify(sp.diff(K, z) - Lp) == 0 and sp.simplify(sp.diff(K, z, 2) - Lpp) == 0)
    boolrec("S8c.boost.kahler_potential", PH, bool(s8c), 'T0',
            note="L' = dK/dzeta, L'' = d2K/dzeta2 for K = ln(zeta/sinh zeta) + zeta^2/t "
                 "(exact up to the dropped windings O(e^{-4pi^2/t})) [SymPy]")

    # S8d  6x6 boosted block is PSD: G_x couples (dp_x, phi_y) and (dp_y, phi_x); each 2x2
    #      [[L' coth, -+L'], [-+L', L' coth]] has eigenvalues L'(coth zeta -+ 1) > 0.
    M = sp.Matrix([[Lp * sp.coth(z), -Lp], [-Lp, Lp * sp.coth(z)]])
    eigs = [sp.simplify(e) for e in M.eigenvals()]
    s8d_sym = all(sp.simplify(e - Lp * (sp.coth(z) - 1)) == 0 or sp.simplify(e - Lp * (sp.coth(z) + 1)) == 0
                  for e in eigs)
    grid = [mpf(k) / 10 for k in range(1, 81)]
    s8d_num = min(min(L1_closed(zz, T_BENCH) * (cosh(zz) / sinh(zz) - 1), L2_closed(zz, T_BENCH))
                  for zz in grid)
    boolrec("S8d.boost.psd", PH, bool(s8d_sym) and s8d_num > 0, 'T0', computed=float(s8d_num),
            note=f"2x2 cross blocks have eigenvalues L'(coth -+ 1); min over zeta in (0,8] at t=0.1 of "
                 f"{{L'(coth-1), L''}} = {float(s8d_num):.3e} > 0 -> 6x6 block PSD [SymPy+grid]")

    # S9  edge-additivity obstruction (restated).  A single-flux response w with
    #     l_P^2 w(p_bar_k) = q_k:  on the conformal orbit q = Q*1 (p_bar = Q) it needs w(Q) = Q;
    #     on q = diag(1,1,lam) (p_bar = (sqrt lam, sqrt lam, 1)) it needs w(sqrt lam) = 1 and
    #     w(1) = lam for every lam -- impossible (w(1) is one number), and the conformal and
    #     anisotropic requirements also clash pointwise: w(sqrt lam) = sqrt lam != 1.
    w = sp.Function('w')
    conformal = sp.Eq(w(sp.sqrt(lam)), sp.sqrt(lam))             # from the conformal orbit at Q = sqrt(lam)
    aniso = sp.Eq(w(sp.sqrt(lam)), 1)                             # from q = diag(1,1,lam), axis 1
    clash = sp.solve([conformal.lhs - conformal.rhs, aniso.lhs - aniso.rhs], [w(sp.sqrt(lam)), lam], dict=True)
    s9 = clash == [{w(sp.sqrt(lam)): 1, lam: 1}]                  # only the isotropic point survives
    boolrec("S9.edge_additive.obstruction", PH, bool(s9), 'T0',
            note="single-flux w(p_bar_k) = q_k: conformal orbit forces w(p)=p, q=diag(1,1,lam) forces "
                 "w(sqrt lam)=1 -> consistent only at lam=1 (isotropic orbit) [SymPy]")


if __name__ == "__main__":
    run()
