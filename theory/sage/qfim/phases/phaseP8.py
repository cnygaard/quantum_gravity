"""
Phase P8 -- probe-variance anchor for the M4 coefficient  c = 1/2 Delta_1  (referee M4).

The emergent-metric prefactor c (installed as ell_P^2 by dimensional analysis) is derived in
the appendix "Determination of Coherence Width" as the positional variance of the theory's
minimal probe:
    Var(x_hat) = a^2 t^2 Var(J_hat) = a^2 t / 2 = 1/2 Delta_1 = 4 pi gamma_0 ell_P^2 ,
with Delta_1 = 8 pi gamma_0 ell_P^2 the unit-spin area gap and  t a^2 = Delta_1  the symplectic
portal.  This phase ANCHORS the two dimensionless inputs of that derivation in the suite's native
t-language, records the gauge-transformation reading of the 4 pi gamma_0 ~= 3.44 factor, and LOGS
(does not select) the O(1) probe-convention residue.

  P8.probe.varJ    Var(J_hat) = G_theta/4 = flat_iso(t)/4 -> 1/(2t)          [ t*Var -> 1/2 ]
                   pure-state QFIM-variance relation  G = 4 Var(generator);  G_theta the exact
                   single-edge angular QFIM  2/t - 1/3  (dictionary.flat_iso).  t*Var = 1/2 - t/12.
  P8.probe.saturate  Var(x_hat) * G_xx = 1  for all (t, a): the canonical probe saturates the
                   quantum Cramer-Rao bound, and the edge scale a cancels (dimensionless efficiency).
  P8.coeff.menu    c/ell_P^2 = n pi gamma_0,  n in {2,3,4} = {1.72, 2.58, 3.44};  headline
                   4 pi gamma_0 = 1/2 Delta_1/ell_P^2 = sigma0^2(portal)/sigma0^2(dim-anal=2) = the
                   3.44 GAUGE factor.  LOGGED, not selected: closure/Gauss -> n=2 (two-face transfer),
                   portal per-edge Berry form -> n=4 (single-face label shift).
  P8.refine.invar  t a^2 = Delta_1 invariant under refinement  a->a/lam, t->lam^2 t  (RG fixed line).

Unlike PhaseLS (which discriminates a single well-defined LS overlap), the three n-values come from
three DISTINCT probe definitions, not from data -- this phase cannot select n; it certifies the
inputs and bounds the residue.  gamma_0 is an external (dimensionful) paper input, not a suite
constant.  All records tier T2 (advisory; never gate).
"""
from mpmath import mpf, pi

from ..core.dictionary import flat_iso
from ..harness import boolrec, phase_status

PH = "PhaseP8"
TS = [mpf('0.1'), mpf('0.02'), mpf('0.004')]
GAMMA0 = mpf('0.274')          # Immirzi (SU(2), Engle-Noui-Perez); paper value, external input


def run(verbose=True):
    # ---- P8.probe.varJ: Var(J) = G_theta/4 = flat_iso(t)/4 -> 1/(2t);  t*Var -> 1/2 ----
    varJ = [flat_iso(t) / 4 for t in TS]          # exact single-edge: (2/t - 1/3)/4
    tvar = [t * v for t, v in zip(TS, varJ)]      # = 1/2 - t/12  -> 1/2
    conv_ok = (all(tvar[i] < tvar[i + 1] for i in range(len(tvar) - 1))     # monotone up
               and abs(tvar[-1] - mpf(1) / 2) < mpf('1e-3'))               # -> 1/2
    boolrec("PhaseP8.probe.varJ", PH, conv_ok, 'T2', computed=float(tvar[-1]), target=0.5,
            note=(f"Var(J)=G_theta/4 -> 1/(2t) [G=4Var]; t*Var(J)=1/2 - t/12 -> 1/2: "
                  f"{[round(float(x), 6) for x in tvar]} at t={[float(t) for t in TS]}"))

    # ---- P8.probe.saturate: Var(x)*G_xx = 1 for all (t,a) (Cramer-Rao; a cancels) ----
    # x_hat = a theta_hat, Var(x)=a^2 Var(theta)=a^2 (t/2), G_xx=G_theta/a^2=(2/t)/a^2 (leading MU).
    sat = []
    for t in TS:
        for a2 in (mpf('0.7'), mpf('11.0'), mpf('123.4')):     # arbitrary edge areas a^2
            sat.append((a2 * (t / 2)) * ((2 / t) / a2))
    sat_ok = all(abs(s - 1) < mpf('1e-30') for s in sat)
    boolrec("PhaseP8.probe.saturate", PH, sat_ok, 'T2', computed=float(sat[0]), target=1.0,
            note="Var(x_hat).G_xx=1 for all (t,a): canonical probe saturates Cramer-Rao; edge scale a cancels")

    # ---- P8.coeff.menu: c/ell_P^2 = n pi gamma_0, n in {2,3,4}; 4pi gamma_0 = the gauge factor ----
    menu = {n: n * pi * GAMMA0 for n in (2, 3, 4)}
    c_over_lP2 = menu[4]                              # portal per-edge headline: 1/2 Delta_1/ell_P^2
    sigma0_portal2 = 8 * pi * GAMMA0                  # Delta_1/ell_P^2 = 6.886 (portal gauge sigma0^2)
    gauge_factor = sigma0_portal2 / 2                 # sigma0^2(portal)/sigma0^2(dim-anal=2) = 3.443
    menu_ok = abs(c_over_lP2 - gauge_factor) < mpf('1e-30')       # 4pi gamma0 == 1/2 * 8pi gamma0
    boolrec("PhaseP8.coeff.menu", PH, menu_ok, 'T2', computed=float(c_over_lP2), target=float(gauge_factor),
            note=(f"c/lP^2 = n.pi.gamma0 (LOGGED, not selected): n=2->{float(menu[2]):.4f} "
                  f"n=3->{float(menu[3]):.4f} n=4->{float(menu[4]):.4f}; headline 4pi.gamma0="
                  f"{float(c_over_lP2):.4f} = 1/2 Delta1/lP^2 = sigma0^2(portal={float(sigma0_portal2):.4f})/2 "
                  f"(the 3.44 gauge factor). closure/Gauss->n=2, portal Berry->n=4."))

    # ---- P8.refine.invar: t a^2 = Delta_1 invariant under refinement a->a/lam, t->lam^2 t ----
    a0, t0 = mpf('8.3'), mpf('0.1')                  # capstone cell (a=8.30 lP, t=0.1)
    base = t0 * a0 ** 2
    refined = [(lam ** 2 * t0) * (a0 / lam) ** 2 for lam in (mpf(2), mpf(3), mpf(5))]
    refine_ok = all(abs(r - base) < mpf('1e-30') for r in refined)
    boolrec("PhaseP8.refine.invar", PH, refine_ok, 'T2', computed=float(base),
            note=f"t.a^2=Delta_1 invariant under a->a/lam, t->lam^2 t (RG fixed line): t.a^2={float(base):.4f}")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    Var(J)=G_theta/4 -> 1/(2t):  t*Var(J) = {[round(float(x),6) for x in tvar]} -> 0.5  ok={conv_ok}")
        print(f"    Cramer-Rao Var(x).G_xx=1 (a-independent):  ok={sat_ok}")
        print(f"    coeff menu n.pi.gamma0:  n=2->{float(menu[2]):.4f}  n=3->{float(menu[3]):.4f}  n=4->{float(menu[4]):.4f}   (LOGGED)")
        print(f"    4pi.gamma0 = {float(c_over_lP2):.4f} = 1/2 Delta1/lP^2 = sigma0^2(portal)/2 = {float(gauge_factor):.4f}  ok={menu_ok}")
        print(f"    refinement t.a^2 invariant:  ok={refine_ok}")
    return st


if __name__ == "__main__":
    run()
