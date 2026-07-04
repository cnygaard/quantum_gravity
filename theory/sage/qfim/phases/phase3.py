"""
Phase 3 -- connector & symplectic.

The connector maps PHYSICAL displacements (dp = flux, du = holonomy angle) to the
GENERATOR coordinates (beta, theta) of the kernel QGT:
    beta_par = dp_par
    beta_perp = (sinh p / p) dp_perp
    theta     = du + ((cosh p - 1)/p) (n x dp_perp)
Its Jacobian J_c pulls the QGT back to physical coords: G^phys = J_c^T G^gen J_c.

Verified here:
  * connector identity  1/sinh p + tanh(p/2) = coth p                 (numeric; T0 in Sage)
  * physical Berry coefficients (L''/2, L'/(2p), L'/2), lock L'/(2p) = <p>/(t p)
  * dp^dp block of B^phys == 0  (flux subspace Lagrangian), swept p in [0.1,3]
  * Lagrangian identity  Xi_P1^T B^phys Xi_P2 = 0 for random flux pairs
  * connector coefficient p/sinh p = 0.850918 via explicit polar decomposition,
    explicit(beta) -> formula with O(beta^2) residual  (adjudicates 8.506 vs 8.509)
  * anisotropy D(p) = L''/2 - L'/(2p); D(0.2) exact vs series  (adjudicates 0.0008852)
"""
from mpmath import (mp, mpf, cosh, sinh, tanh, coth, matrix, fabs, sqrt, log,
                    mpc)

from ..core.dictionary import T_BENCH, L1_closed, L2_closed
from ..core.kernel import boost, displace, qgt, G_from_Q, B_from_Q, flux_of
from .phase1 import edge_family
from ..harness import rec, boolrec, phase_status

PH = "Phase3"


def connector_J(pbar):
    """J_c: d(beta,theta)/d(dp,du) at base p=pbar zhat, n=zhat."""
    c, s, p = cosh(pbar), sinh(pbar), pbar
    J = matrix(6, 6)
    J[0, 0] = s / p          # beta_x = (s/p) dp_x
    J[1, 1] = s / p          # beta_y = (s/p) dp_y
    J[2, 2] = mpf(1)         # beta_z = dp_z
    J[3, 3] = mpf(1)         # theta_x <- du_x
    J[4, 4] = mpf(1)         # theta_y <- du_y
    J[5, 5] = mpf(1)         # theta_z <- du_z
    J[3, 1] = -(c - 1) / p   # theta_x += -((c-1)/p) dp_y   [ (zxdp)_x = -dp_y ]
    J[4, 0] = (c - 1) / p    # theta_y +=  ((c-1)/p) dp_x   [ (zxdp)_y =  dp_x ]
    return J


def phys_QGT(pbar, t):
    g_of = edge_family(pbar, t)
    Q = qgt(g_of, 6, t)
    Jc = connector_J(pbar)
    G = G_from_Q(Q)
    B = B_from_Q(Q)
    return Jc.T * G * Jc, Jc.T * B * Jc


def anisotropy_D(pbar, t):
    Lp, Lpp = L1_closed(pbar, t), L2_closed(pbar, t)
    return Lpp / 2 - Lp / (2 * pbar)


def run(pbar=mpf(1), t=T_BENCH, verbose=True):
    Lp, Lpp = L1_closed(pbar, t), L2_closed(pbar, t)

    # --- connector identity (numeric; symbolic T0 lives in proofs) ---
    idmax = max(fabs(1 / sinh(p) + tanh(p / 2) - coth(p))
                for p in [mpf('0.3'), mpf(1), mpf('2.5')])
    rec("P3.identity.1sinh_tanh_coth", PH, idmax, mpf(0), 'Z',
        note="1/sinh p + tanh(p/2) - coth p = 0")

    # --- physical-coordinate Berry blocks ---
    Gp, Bp = phys_QGT(pbar, t)
    rec("P3.Bphys.long", PH, Bp[5, 2], Lpp / 2, 'NUM', note="B^phys long = L''/2")
    rec("P3.Bphys.trans", PH, Bp[3, 0], Lp / (2 * pbar), 'NUM', note="B^phys trans = L'/(2p)")
    rec("P3.Bphys.hol", PH, Bp[4, 3], Lp / 2, 'NUM', note="B^phys hol = L'/2")
    phat_over_tp = (t / 2 * Lp) / (t * pbar)
    rec("P3.Bphys.lock", PH, Lp / (2 * pbar), phat_over_tp, 'NUM',
        note="L'/(2p) = <p>/(t p)")

    # --- dp^dp block == 0, swept over p in [0.1, 3] (flux subspace Lagrangian) ---
    dpdp_max = mpf(0)
    for pv in [mpf('0.1'), mpf('0.5'), mpf(1), mpf(2), mpf(3)]:
        _, Bpp = phys_QGT(pv, t)
        dpdp_max = max(dpdp_max, max(fabs(Bpp[i, j]) for i in range(3) for j in range(3)))
    rec("P3.dpdp.zero_sweep", PH, dpdp_max, mpf(0), 'Znum',
        note="max|B^phys[dp,dp]| over p in [0.1,3]")

    # --- Lagrangian identity: random flux pairs Xi1^T B^phys Xi2 = 0 ---
    def flux_vec(seed):
        # deterministic pseudo-random transverse+long flux vector (params 0..2)
        v = [mpf(0)] * 6
        for k in range(3):
            v[k] = mpf(((seed * 131 + k * 17) % 97) - 48) / 50
        return v
    lag_max = mpf(0)
    for sd in range(6):
        X1, X2 = flux_vec(sd), flux_vec(sd + 100)
        val = sum(X1[i] * Bp[i, j] * X2[j] for i in range(6) for j in range(6))
        lag_max = max(lag_max, fabs(val))
    rec("P3.lagrangian.flux_pairs", PH, lag_max, mpf(0), 'Znum',
        note="Xi_P1^T B^phys Xi_P2 = 0 (random flux pairs)")

    # --- connector coefficient via explicit polar decomposition, O(beta^2) ---
    coeff_formula = pbar / sinh(pbar)         # dp_perp / beta  = p/sinh p
    gbase = boost([0, 0, pbar])
    resid = []
    for beta in [mpf('0.02'), mpf('0.01'), mpf('0.005')]:
        g = displace(gbase, [beta, 0, 0])      # transverse boost generator beta
        dp_perp = flux_of(g)[0]                # measured transverse flux
        ratio = dp_perp / beta
        resid.append((beta, fabs(ratio - coeff_formula)))
    # quadratic convergence: residual ratio ~ 4 when beta halves
    slope = log(resid[0][1] / resid[2][1]) / log(resid[0][0] / resid[2][0])
    bt = mpf('1e-11')
    rec("P3.connector.coeff", PH, flux_of(displace(gbase, [bt, 0, 0]))[0] / bt,
        coeff_formula, 'NUM', note="dp_perp/beta -> p/sinh p = 0.850918")
    boolrec("P3.connector.quadratic", PH, mpf('1.9') < slope < mpf('2.1'), 'Znum',
            note=f"|explicit-formula| ~ beta^{float(slope):.3f} (adjudicates 8.506 vs 8.509)")

    # --- anisotropy D(0.2): exact vs 2-term series; refutes 0.0008852 ---
    D02 = anisotropy_D(mpf('0.2'), t)
    series2 = mpf('0.2') ** 2 / 45 - 4 * mpf('0.2') ** 4 / 945
    rec("P3.D02.exact_vs_series", PH, D02, series2, 'T2',
        note="D(0.2) exact vs 2-term series p^2/45-4p^4/945")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  connector identity max|.| = {float(idmax):.2e}")
        print(f"  B^phys (long,trans,hol) = {float(Bp[5,2]):.5f},{float(Bp[3,0]):.5f},{float(Bp[4,3]):.5f}"
              f"  (L''/2={float(Lpp/2):.5f}, L'/2p={float(Lp/(2*pbar)):.5f}, L'/2={float(Lp/2):.5f})")
        print(f"  max|B^phys[dp,dp]| swept = {float(dpdp_max):.2e}   Lagrangian max = {float(lag_max):.2e}")
        print(f"  connector coeff p/sinh p = {mp.nstr(coeff_formula,8)}  (explicit ratios {[mp.nstr(r[1],3) for r in resid]}, slope {float(slope):.3f})")
        print(f"  D(0.2) exact = {mp.nstr(D02,10)}  series2 = {mp.nstr(series2,10)}  [refutes 0.0008852]")
    return st


if __name__ == "__main__":
    run()
