"""
Phase Page -- Page-curve bookkeeping for a finite LQG horizon (information-paradox programme).

An LQG horizon of area A is a set of N punctures with spins j_p, area
a_p = 8 pi gamma0 lP^2 sqrt(j_p(j_p+1)) and FINITE multiplicity d_p = 2 j_p + 1. Finiteness gives
the ceiling S_rad <= ln m and the exact return S_rad(end) = 0; the Page CURVE additionally needs
typicality of the global state (Page's theorem), which this phase assumes and does not derive.

Capacity vs area: one fixed configuration has ln D = sum ln d_p, equal to A/4 only for
monochromatic punctures at the matched gamma0 (H2(i)); the Ghosh-Perez route (H2(ii)) gives
S = A/4 + N sigma(gamma) for the DIRECT SUM over configurations (the label entropy is ~58% of
A/4 at gamma*). Heights use the exact Page average with a centre (Bianchi-Dona): SU(2) (J_R)
and, for the direct sum, the area split.

Records (all T2 advisory -- this phase LOGS the bookkeeping, it never gates):
  PhasePage.finite_dim       : ln D = A/4 at the ABCK-matched gamma0 (H2(i))
  PhasePage.peak_half        : unconstrained-register Page peak = ln m - 1/2
  PhasePage.returns          : S_rad(end) = 0
  PhasePage.qes_cross        : the two branches of min(C_emitted, C_remaining) cross at half-area
  PhasePage.gamma_indep      : capacity-fraction shape independent of gamma0 (manifest)
  PhasePage.gp_area_law      : GP identity S/N = A/(4N) + sigma, sigma(gamma*) = 0
  PhasePage.gp_capacity_split: <a>/4 = <ln d> + H(label) at gamma* (fixed config = 42% of A/4)
  PhasePage.gp_page_gamma    : exchangeable order: Page point at half the area, every gamma
  PhasePage.gp_order         : Page point vs emission order (random/composition/small/large)
  PhasePage.closure_log      : ln Catalan(N/2) = N ln 2 + c1 ln N + c0, c1 = -3/2
  PhasePage.singlet_page_exact: exact Haar-singlet curve (n = 14) vs Monte Carlo
  PhasePage.centre_deficit   : SU(2)-centre midpoint deficit below ln m - 1/2, saturating ~0.289
  PhasePage.centre_area_split: direct-sum area split: -alpha sqrt(N) midpoint term, fit vs theory
"""
import os
import sys

from mpmath import mp, mpf, log, sqrt, pi, e

_THEORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _THEORY not in sys.path:
    sys.path.insert(0, _THEORY)                # for page_curve (gp_* functions)

from ..harness import boolrec, phase_status

PH = "PhasePage"


def _page_entropy(ln_m, ln_n):        # ln m <= ln n; S = ln m - m/(2n), in log space
    return ln_m - e ** (ln_m - ln_n) / 2


def _curve(spins, gamma0):
    """(area_frac, S_rad, ln d_B) as punctures emit largest-area-first."""
    spins = sorted(spins, key=lambda j: -float(sqrt(mpf(j) * (j + 1))))
    lnd = [log(mpf(2 * j + 1)) for j in spins]
    area = [8 * pi * gamma0 * sqrt(mpf(j) * (j + 1)) for j in spins]
    N, A = len(spins), sum(8 * pi * gamma0 * sqrt(mpf(j) * (j + 1)) for j in spins)
    out = []
    for k in range(N + 1):
        lnR, lnB = sum(lnd[:k]), sum(lnd[k:])
        m, n = (lnR, lnB) if lnR <= lnB else (lnB, lnR)
        S = _page_entropy(m, n) if k not in (0, N) else mpf(0)
        out.append((float(sum(area[:k]) / A), float(S), float(lnB)))
    return out, float(sum(lnd)), float(A)


def run(verbose=True):
    dps0 = mp.dps                      # never clobber the global gate precision
    mp.dps = max(dps0, 30)
    try:
        return _run(verbose)
    finally:
        mp.dps = dps0


def _run(verbose):
    N = 200
    spins = [mpf(1) / 2] * N
    curve, S0, A0 = _curve(spins, mpf('0.274'))
    peak = max(curve, key=lambda r: r[1])

    # (1) finiteness + the honest capacity-area dictionary (referee-corrected): the
    #     monochromatic j=1/2 MATCHING ln2 = 2 pi gamma0 sqrt(3)/2 fixes gamma0=0.1273
    #     (hypothesis H2(i)); ln d_p is NOT proportional to a_p across mixed spins.
    g_match = float(log(mpf(2)) / (pi * sqrt(mpf(3))))
    _, S0m, A0m = _curve(spins, g_match)
    boolrec("PhasePage.finite_dim", PH, abs(S0m / (A0m / 4) - 1) < 1e-9, 'T2',
            computed=S0m / (A0m / 4),
            note=f"H2(i) monochromatic matching: ln D = A/4 at gamma0={g_match:.4f} (j=1/2); "
                 "NOT a gamma0-free identity -- see gp_* records for the ensemble route")

    # (2) unconstrained-register bookkeeping: peak = ln m - 1/2 (Page's finite-size correction).
    #     The SU(2)-invariant height is centre_deficit below (an extra O(1) ~0.29 nat).
    peak_ratio = peak[1] / (S0 / 2)
    boolrec("PhasePage.peak_half", PH, 0.99 < peak_ratio <= 1.0, 'T2', computed=peak_ratio,
            note=f"unconstrained registers: peak S_rad={peak[1]:.3f} vs ln m={S0/2:.3f}; deficit={S0/2-peak[1]:.3f} "
                 "nat = Page's 1/2 (the SU(2) centre adds centre_deficit)")

    # (3) information returns: S_rad(end)=0
    boolrec("PhasePage.returns", PH, curve[-1][1] < 1e-6, 'T2', computed=curve[-1][1],
            note="S_rad(N)=0 -> pure state recovered -> information preserved (unitary)")

    # (4) QES saddle degeneracy at half-area: no-island A_emit/4 == island A_horiz/4
    mid = curve[N // 2]
    s_isl = mid[2] / S0                       # island saddle A_remaining/4
    s_noi = 1 - s_isl                          # no-island saddle A_emitted/4
    boolrec("PhasePage.qes_cross", PH, abs(s_isl - s_noi) < 1e-9, 'T2', computed=abs(s_isl - s_noi),
            note=f"at area-frac {mid[0]:.3f} the branches of min(C_emit, C_rem) cross ({s_noi:.3f}={s_isl:.3f}); "
                 "no extremization is performed")

    # (5) gamma0-independence of the curve shape IN CAPACITY FRACTION (trivial by
    #     normalization for a fixed spin profile -- the referee's point; the nontrivial
    #     area-fraction statement is gp_page_gamma below)
    shapes = []
    for g in (mpf('0.2375'), mpf('0.274'), mpf('0.5'), mpf('1')):
        c, s0, _ = _curve(spins, g)
        shapes.append(max(c, key=lambda r: r[1])[1] / s0)
    spread = max(shapes) - min(shapes)
    boolrec("PhasePage.gamma_indep", PH, spread < 1e-9, 'T2', computed=spread,
            note=f"capacity-fraction shape spread over gamma0 = {spread:.1e} (manifest for fixed "
                 "spin profile; area-fraction version = gp_page_gamma)")

    # (6) Ghosh-Perez ensemble (PRL 107,241301): S = A/4 + N sigma(gamma) with UNIVERSAL
    #     1/4 coefficient for every gamma; sigma(gamma*)=0 at gamma* = the framework value.
    from page_curve import (gp_area_law, gp_gamma_star, gp_page, closure_log_fit, gp_page_order,
                            gp_split, singlet_page_exact, centre_deficit, area_split_fit)
    res_max = max(abs(gp_area_law(g)[0]) for g in (0.2, 0.2375, 0.274, 0.5))
    gstar = gp_gamma_star()
    boolrec("PhasePage.gp_area_law", PH, res_max < 1e-30 and abs(gstar - 0.274) < 1e-3, 'T2',
            computed=gstar,
            note=f"GP identity S/N=A/(4N)+sigma exact (max res {res_max:.1e}); universal 1/4 "
                 f"coefficient all gamma; sigma=0 at gamma*={gstar:.6f} = Ghosh-Mitra/ENP "
                 "state-counting value (zero chemical potential, per Ghosh-Perez)")
    # (7) GP Page curves at several gamma, EXCHANGEABLE (random) emission order: peak at half
    #     the area up to O(N^-1/2) -- by the law of large numbers, not by the dictionary.
    afs = [gp_page(g, N=400, seed=1)[0] for g in (0.2, 0.274, 0.5)]
    off = max(abs(a - 0.5) for a in afs)
    boolrec("PhasePage.gp_page_gamma", PH, off < 0.05, 'T2', computed=off,
            note=f"GP-sampled spins, exchangeable order: peak area-frac {['%.3f' % a for a in afs]} "
                 "for gamma in (0.2,0.274,0.5): half-area to O(N^-1/2), every gamma (LLN)")
    # (7b) emission-order sensitivity (referee): composition-preserving (quasi-static) order keeps
    #      the half-area point; spin-selective orders move it.
    tab = {g: {o: gp_page_order(g, o, N=400, seed=1) for o in ('random', 'composition', 'small', 'large')}
           for g in (0.2, 0.274067, 0.5)}
    comp_off = max(abs(t['composition'] - 0.5) for t in tab.values())
    sel = max(max(abs(t['small'] - 0.5), abs(t['large'] - 0.5)) for t in tab.values())
    boolrec("PhasePage.gp_order", PH, comp_off < 0.01 and sel > 0.03, 'T2', computed=sel,
            note="Page-point area fraction (random/composition/small-first/large-first): " +
                 "; ".join(f"gamma={g:.3f}: " + "/".join(f"{t[o]:.3f}" for o in ('random', 'composition', 'small', 'large'))
                           for g, t in tab.items()))
    a4, lnd, H, fr = gp_split(gstar)
    boolrec("PhasePage.gp_capacity_split", PH, abs(a4 - lnd - H) < 1e-9 and 0.40 < fr < 0.44, 'T2',
            computed=fr,
            note=f"at gamma*: <a>/4={a4:.4f} = <ln d> {lnd:.4f} + H(label) {H:.4f}; one fixed configuration "
                 f"holds {fr:.1%} of A/4 -> route (ii) needs the direct sum over configurations")
    # (8) closure (gauge-invariant) counting: ln dim Inv = N ln2 - (3/2) ln N + O(1)
    c1, c0 = closure_log_fit(Ns=(160, 320, 640, 1280, 2560))
    boolrec("PhasePage.closure_log", PH, abs(c1 + 1.5) < 0.05, 'T2', computed=c1,
            note=f"ln Catalan(N/2) = N ln2 + c1 ln N + c0: fitted c1={c1:.3f} (Kaul-Majumdar "
                 "-3/2 closure log); dim Inv((C^2)^14)=C_7=429, ln=6.06 vs 14ln2=9.70")

    # (9) exact Haar-singlet curve (Bianchi-Dona centre formula) vs Monte Carlo (n=14)
    import evaporation as ev
    ex = [singlet_page_exact(14, k) for k in range(1, 8)]
    idx, _ = ev.jz0_basis(14)
    _, J2, _ = ev.build_sector(14, 0)
    mc, sd = ev.page_value(ev.singlet_P(J2), idx, 7, 14, nsamp=400, seed=3)
    boolrec("PhasePage.singlet_page_exact", PH, abs(mc - ex[-1]) < 4 * sd / 20 + 1e-3, 'T2', computed=ex[-1],
            note="exact Haar-singlet S(R_k), n=14, k=1..7: " + " ".join(f"{x:.4f}" for x in ex) +
                 f"; Monte Carlo (400 draws) k=7: {mc:.4f} +/- {sd / 20:.4f}")
    # (10) SU(2) centre: midpoint deficit below the unconstrained ln m - 1/2 is O(1), saturating
    defs = {n: centre_deficit(n) for n in (14, 60, 240, 960)}
    boolrec("PhasePage.centre_deficit", PH, abs(defs[960][1] - 0.289) < 0.002 and defs[14][1] < defs[960][1],
            'T2', computed=defs[960][1],
            note="midpoint deficit below ln m - 1/2 (n=14/60/240/960): " +
                 "/".join(f"{d[1]:.4f}" for d in defs.values()) + "; below ln m: " +
                 "/".join(f"{d[2]:.4f}" for d in defs.values()) + " -- the closure log does not enter the height")
    # (11) option (a): direct sum with the area split as an abelian centre (two-species model)
    a, b, c, res, a_pred, ok = area_split_fit(0.25)
    a2, b2, c2, res2, a2_pred, ok2 = area_split_fit(0.5)
    boolrec("PhasePage.centre_area_split", PH, ok and ok2 and abs(a - a_pred) < 0.02 and abs(a2 - a2_pred) < 0.02,
            'T2', computed=a,
            note=f"S_mid - ln D_tot/2 = -{a:.3f} sqrt(N) + {b:.2f} ln N + {c:.2f} (N=32..512, n1=N/4, max res "
                 f"{res:.1e}); n1=N/2: -{a2:.3f} sqrt(N) + {b2:.2f} ln N + {c2:.2f} (max res {res2:.1e}); predicted "
                 f"alpha {a_pred:.3f} / {a2_pred:.3f} from area-split fluctuations; exact dims {ok and ok2}")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    finite dim: ln D=A/4 at the ABCK-matched gamma0; unconstrained peak/ln m={peak_ratio:.4f} at area-frac {peak[0]:.3f}")
        print(f"    S_rad(end)={curve[-1][1]:.1e}; branch crossing |C_emit-C_rem|={abs(s_isl-s_noi):.1e} at half-area")
        print(f"    GP: sigma=0 at gamma*={gstar:.4f}; split <a>/4={a4:.3f}=<ln d> {lnd:.3f}+H {H:.3f} ({fr:.1%} in one config)")
        print(f"    emission order: exchangeable off {off:.3f}, composition-preserving off {comp_off:.3f}, spin-selective up to {sel:.3f}")
        print(f"    closure: c1={c1:.3f}; exact singlet S(R_7)={ex[-1]:.4f} (MC {mc:.4f}); SU(2)-centre deficit -> {defs[960][1]:.4f}")
        print(f"    direct-sum area split: -{a:.3f} sqrt(N) (predicted {a_pred:.3f}) + {b:.2f} ln N")
        print(f"    VERDICT: finite capacity + typicality (assumed) -> centre-corrected Page curve; half-area Page point")
        print(f"             for exchangeable or composition-preserving emission, every gamma0")
    return st


if __name__ == "__main__":
    run()
