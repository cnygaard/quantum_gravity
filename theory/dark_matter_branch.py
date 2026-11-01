"""
Dark-matter C_ab ratio: is the HARD (discreteness-floor) branch physical?
========================================================================

v12 (app:holo-Cab) leaves this door open: the soft covariance-strain gives an
Immirzi-independent  M_DM/M_b = kappa/(4 ln2) ~ 0.18-0.36, and "recovering the
observed 5.36 would require pinning saturation at the discreteness floor [hard],
left to future work."  This script tests whether hard can be that route.

Hard needs each pinned cell squeezed to the area floor so its inverse-variance
(Fisher) 1/f blows up and, summed over cells, restores a 1/gamma_0 enhancement.
Three independent, framework-checkable objections are evaluated:

  (1) CONTINUUM LIMIT.  Write the three explicit routes' Immirzi scalings and
      take the discreteness scale a_1 = 8 pi gamma0 lP^2 sqrt(j(j+1)) -> 0.
      A genuine quantum-geometry (discreteness) effect must stay finite or
      vanish.  soft ~ gamma0^0 (finite), screen G_eff ~ gamma0^+1 (->0), but
      hard ~ gamma0^-1 DIVERGES -> unphysical.

  (2) SUSCEPTIBILITY.  Matter couples diagonally to the area operator, so the
      source parameter is a NATURAL (exponential-family) parameter lambda with
      tilt e^{lambda A}.  Then the source-Fisher is F_lambda = Var(A), and the
      identity  d<A>/dlambda = Var(A)  holds exactly.  Hence the response is
      governed by Var, not 1/Var: pinning (Var down) can only REDUCE the sourced
      strain.  The hard 1/f enhancement uses 1/Var (unit susceptibility), which
      is the wrong Fisher for a diagonal coupling.  Verified numerically here.

  (3) EIGENSTATE INACCESSIBILITY.  Hard needs the pinned cell near an area
      eigenstate (Var -> floor).  On the heat-kernel state the hard model's
      variance collapses ~ t (phaseDM), i.e. it approaches a sharp-spin state,
      whose conjugate (extrinsic-curvature) fluctuation diverges -- destroying
      the balanced coherent state that reproduces classical geometry.

Conclusion (verified): the hard branch is closed; soft ~0.4 is the honest
magnitude; the 5.36 gap is real for this mechanism.
"""
from mpmath import mp, mpf, sqrt, exp, log, pi, sin

mp.dps = 40
G0 = mpf('0.274')     # Immirzi (framework value)
LN2 = log(mpf(2))


# ---------------------------------------------------------------- heat-kernel moments
def moments(t, lam=mpf(0), band=None, nmax=4000):
    """<1>, <A>, <A^2> under P(n) ~ n^2 e^{-t (n^2-1)/4} e^{lam A}, A=sqrt(j(j+1)).
    band=(n0,W) restricts to n in [n0-W, n0+W] (the hard fixed-j projection)."""
    lo, hi = 1, nmax
    if band is not None:
        n0, W = band
        lo, hi = max(1, n0 - W), n0 + W
    m0 = m1 = m2 = mpf(0)
    for n in range(lo, hi + 1):
        s = mpf(n * n - 1) / 4
        A = sqrt(s)
        w = mpf(n * n) * exp(-t * s + lam * A)
        m0 += w; m1 += w * A; m2 += w * A * A
    return m0, m1, m2


def mean_var(t, lam=mpf(0), band=None):
    m0, m1, m2 = moments(t, lam, band)
    mean = m1 / m0
    return mean, m2 / m0 - mean * mean


def run():
    print("=" * 70)
    print("DARK-MATTER C_ab RATIO: IS THE HARD (DISCRETENESS-FLOOR) BRANCH PHYSICAL?")
    print("=" * 70)

    # ---- baseline: the soft, Immirzi-free magnitude -------------------------
    for kappa in (mpf(1) / 2, mpf('1.11')):
        print(f"  soft  M_DM/M_b = kappa/(4 ln2) = {float(kappa/(4*LN2)):.3f}   (kappa={float(kappa)})")
    print("  observed Omega_c/Omega_b = 5.36 +/- 0.07")

    # ---- (1) continuum-limit scaling of the three explicit routes -----------
    print("\n[1] CONTINUUM LIMIT  (discreteness scale a_1 = 8 pi gamma0 lP^2 sqrt(j(j+1)) -> 0):")
    print("    route                    Immirzi power   value@0.274    a_1->0 (gamma0->0)")
    Om = mpf('0.315')
    sinc = float(sin(sqrt(Om)) / sqrt(Om))
    soft = float(mpf('1.11') / (4 * LN2))                       # ~0.4, gamma0^0
    screen = float(G0 * pi / 2)                                 # 0.43, gamma0^+1
    hard = float(pi / (2 * G0)) * sinc                          # 5.44, gamma0^-1 (restored headline)
    for name, val, p in [("soft C_ab", soft, 0), ("screen G_eff", screen, +1),
                          ("hard = pi/(2 g0) sinc", hard, -1)]:
        lim = "finite" if p == 0 else ("-> 0 (OK)" if p > 0 else "-> INF  (UNPHYSICAL)")
        print(f"    {name:24s} gamma0^{p:+d}        {val:6.3f}        {lim}")
    print("    -> a matter-sourced gravity effect must not blow up as space becomes")
    print("       smooth; only hard does.  DECISIVE against hard.")

    # ---- (2) susceptibility identity: source-Fisher = Var, not 1/Var --------
    print("\n[2] SUSCEPTIBILITY  (matter couples diagonally to A -> tilt e^{lambda A}):")
    print("    check the exponential-family identity  d<A>/dlambda = Var(A):")
    h = mpf('1e-6')
    for t in (mpf('0.1'), mpf('0.02')):
        mean0, var0 = mean_var(t)
        mp_, _ = mean_var(t, lam=h)
        mm_, _ = mean_var(t, lam=-h)
        dmean = (mp_ - mm_) / (2 * h)
        print(f"    t={float(t):5.3f}:  d<A>/dlambda={float(dmean):.6f}  Var(A)={float(var0):.6f}"
              f"  rel.diff={float(abs(dmean-var0)/var0):.1e}")
    print("    -> source-Fisher F_lambda = Var(A).  Pinning lowers Var -> lowers the")
    print("       sourced strain.  The hard 1/f (=1/Var) is the WRONG Fisher for a")
    print("       diagonal coupling; its enhancement is sign-inverted.")

    # ---- (3) eigenstate inaccessibility: hard needs Var -> floor (~t) --------
    print("\n[3] EIGENSTATE INACCESSIBILITY  (hard pins toward a sharp-spin area eigenstate):")
    print("    vacuum vs fixed-j-band (W=1) variance on the heat-kernel state:")
    for t in (mpf('0.1'), mpf('0.02'), mpf('0.004')):
        _, var_vac = mean_var(t)
        n0 = int(round(float(2 / sqrt(t))))
        _, var_hard = mean_var(t, band=(n0, 1))
        print(f"    t={float(t):5.3f}:  Var_vac={float(var_vac):8.3f}  Var_hard(W=1)={float(var_hard):.4f}"
              f"  ratio={float(var_hard/var_vac):.4f}")
    print("    -> hard drives Var -> O(1) floor (ratio ~ t -> 0): a near area-eigenstate,")
    print("       whose conjugate extrinsic-curvature fluctuation diverges, destroying the")
    print("       balanced coherent state that reproduces smooth classical geometry.")
    print("       Weak galactic coupling (G rho ~ 1e-120) cannot drive Var to this floor.")

    print("\n" + "=" * 70)
    print("VERDICT: the HARD branch is CLOSED on three independent grounds (continuum")
    print("divergence, wrong-Fisher/susceptibility, coherent-state pointer).  The C_ab")
    print("mechanism robustly gives the soft, Immirzi-free ~0.4 (a G_eff-like source")
    print("renormalisation, not a halo); the observed 5.36 is NOT recovered.  v12's")
    print("'hard, left to future work' door is closed in the negative.")
    print("=" * 70)


if __name__ == "__main__":
    run()
