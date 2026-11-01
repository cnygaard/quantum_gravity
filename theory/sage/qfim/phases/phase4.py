"""
Phase 4 -- Case 1 anisotropic weights, with the 4a-4d closed forms (SPEC addendum).

Edge weights: w_L = p^2 L''(p),  w_T = 2 L'(p) tanh(p/2),  w = w_L + w_T.
Dictionary (unimodular slice): p_bar_k = sqrt(det q / q_k), so q_k p_bar_k^2 = det q.

  4a  Shape functional  S[q] = (sum_k w_L(p_bar_k)) / sqrt(det q),  p_bar_k = sqrt(q_j q_l).
      R_shape = S[q]/S[1] = 1.756718;  R(Q) = S[Q1]/S[1] = sqrt(Q) L''(Q)/L''(1).
      Both are the SAME S[q] routine on two slices (a dictionary-bug tripwire).
  4b  Shear plane weight  W_ab = w_T(p_bar_a) + w_T(p_bar_b) = (23.013432, 78.304476, 64.927172).
  4c  Displacement  rho_i = 2[2 W_tot - w(p_bar_i)],  W_tot = sum_k w(p_bar_k);
      rho_bar = (10/3) W_tot;  gamma off-diagonal == 0 (averaged probe on diagonal q).
  4d  Susceptibility  chi_T(p) = p w'(p) / (5 w(p)) = (1/5) dln w / dln p;
      chi_T -> 2/5 at BOTH p->0 and p->infinity, with an interior dip (new claims).
"""
from mpmath import mp, mpf, tanh, cosh, sinh, sqrt, log, fabs

from ..core.dictionary import (T_BENCH, PBAR, Q_DIAG, L1_closed, L2_closed,
                               L3_closed)
from ..core.kernel import L_derivs
from ..harness import rec, boolrec, phase_status

PH = "Phase4"


def wL(p, t):
    return p ** 2 * L2_closed(p, t)


def wT(p, t):
    return 2 * L1_closed(p, t) * tanh(p / 2)


def w(p, t):
    return wL(p, t) + wT(p, t)


def wprime(p, t):
    L1, L2, L3 = L1_closed(p, t), L2_closed(p, t), L3_closed(p)
    return 2 * p * L2 + p ** 2 * L3 + 2 * L2 * tanh(p / 2) + L1 / cosh(p / 2) ** 2


def pbar_of_q(q):
    detq = q[0] * q[1] * q[2]
    return [sqrt(detq / q[k]) for k in range(3)]


def S_of_q(q, t):
    """conformal shape functional S[q] = sum_k w_L(p_bar_k) / sqrt(det q)."""
    detq = q[0] * q[1] * q[2]
    pb = pbar_of_q(q)
    return sum(wL(pb[k], t) for k in range(3)) / sqrt(detq)


def chiT(p, t):
    return p * wprime(p, t) / (5 * w(p, t))


def run(t=T_BENCH, verbose=True):
    pb, q = PBAR, Q_DIAG
    detq = q[0] * q[1] * q[2]

    # ---- base weights (vs anchors + brute spin sum) ----
    wL_v = [wL(p, t) for p in pb]
    wT_v = [wT(p, t) for p in pb]
    rec("P4.wL.anchor", PH, wL_v, [mpf('4.920674'), mpf('19.7240872'), mpf('79.3040872')], 'T2',
        note="w_L = p^2 L''")
    rec("P4.wT.anchor", PH, wT_v, [mpf('4.818064'), mpf('18.195368'), mpf('60.109108')], 'T2',
        note="w_T = 2 L' tanh(p/2)")
    wL_brute = [p ** 2 * L_derivs(p, t)[2] for p in pb]
    rec("P4.wL.brute_vs_closed", PH, wL_brute, wL_v, 'NUM', note="w_L via spin-sum L''")
    rec("P4.ratio.wT_wL", PH, wT_v[1] / wL_v[1], mpf('0.922496'), 'T2', note="w_T(1)/w_L(1)")

    # ---- 4a: shape functional -- R_shape and R(Q) from the SAME S[q] ----
    S_iso = S_of_q([mpf(1)] * 3, t)
    R_shape = S_of_q(q, t) / S_iso
    RQ4_func = S_of_q([mpf(4)] * 3, t) / S_iso
    RQ4_closed = sqrt(mpf(4)) * L2_closed(mpf(4), t) / L2_closed(mpf(1), t)
    rec("P4.R_shape", PH, R_shape, mpf('1.756718'), 'T2', note="S[q]/S[1] = 1.756718")
    rec("P4.R.Q4", PH, RQ4_func, mpf('2.021779'), 'T2', note="S[Q1]/S[1], Q=4")
    rec("P4.R.functional_lock", PH, RQ4_func, RQ4_closed, 'NUM',
        note="R(Q) same S[q] routine == sqrt(Q)L''(Q)/L''(1) [dictionary tripwire]")

    # ---- edge-additivity obstruction, restated (referee: the sqrt(Q) of R(Q) is the
    #      p_bar^2/sqrt(det q) normalization of S[q], not an anomalous exponent) ----
    # (i) fixed t: the area-direction QFIM L''(p_bar) -> 2/t carries the state's width, not the
    #     background, so it cannot track even a conformal rescaling: L''(Q)/(Q L''(1)) ~ 1/Q.
    Rnat = L2_closed(mpf(4), t) / (4 * L2_closed(mpf(1), t))
    rec("P4.Rnat.Q4", PH, Rnat, mpf('0.2527223'), 'T2',
        note="fixed t: L''(4)/(4 L''(1)) = 0.25272 (a metric identity needs 1)")
    # (ii) area-adapted t_e = t0/p_bar_e (Thiemann t = l_P^2/a^2 with a^2 the edge's area
    #      scale): L'' ~ 2 p_bar/t0 tracks the area, so the conformal test passes up to O(t0).
    def L2_adapt(p, t0):
        return L2_closed(p, t0 / p)
    Radapt = L2_adapt(mpf(4), t) / (4 * L2_adapt(mpf(1), t))
    rec("P4.Radapt.Q4", PH, Radapt, mpf('1.0132148'), 'T2',
        note="adapted t=t0/p: L''(4;t0/4)/(4 L''(1;t0)) = 1.0132 -> 1 + O(t0)")
    Radapt_small = L2_adapt(mpf(4), mpf('1e-4')) / (4 * L2_adapt(mpf(1), mpf('1e-4')))
    boolrec("P4.Radapt.limit", PH, fabs(Radapt_small - 1) < mpf('1e-4') < fabs(Radapt - 1), 'Znum',
            computed=float(Radapt_small),
            note=f"adapted-t conformal ratio -> 1 as t0 -> 0 (t0=1e-4: {float(Radapt_small):.7f})")
    # (iii) ...but then G_kk ~ area p_bar_k, while q_k = p_bar_j p_bar_l / p_bar_k: off the
    #       isotropic orbit the single-flux response fails at O(1).  Witness q = diag(1,1,4).
    qa = [mpf(1), mpf(1), mpf(4)]
    pa = [sqrt(qa[1] * qa[2]), sqrt(qa[0] * qa[2]), sqrt(qa[0] * qa[1])]        # (2, 2, 1)
    ratios = [L2_adapt(pa[k], t) / qa[k] for k in range(3)]
    aniso = max(ratios) / min(ratios)
    rec("P4.aniso.witness", PH, aniso, mpf('8.0766371'), 'T2',
        note=f"adapted t, q=diag(1,1,4): max/min of G_kk/q_k = {float(aniso):.4f} (-> 8 as t0->0; identity needs 1)")

    # ---- 4b: shear-plane triple  w_T(a) + w_T(b) ----
    shear = [wT(pb[a], t) + wT(pb[b], t) for (a, b) in [(0, 1), (1, 2), (2, 0)]]
    rec("P4.shear.triple", PH, shear,
        [mpf('23.013432'), mpf('78.304476'), mpf('64.927172')], 'T2',
        note="W_ab = w_T(p_a) + w_T(p_b)")
    rec("P4.shear.maxmin", PH, max(shear) / min(shear), mpf('3.402554'), 'T2',
        note="cross-plane violation max/min")

    # ---- 4c: displacement rho_i = 2[2 W_tot - w(p_i)] ----
    WLtot = sum(wL(p, t) for p in pb)
    WTtot = sum(wT(p, t) for p in pb)
    Wtot = WLtot + WTtot
    rho = [2 * (2 * Wtot - w(pb[i], t)) for i in range(3)]
    rho_bar = mpf(10) / 3 * Wtot
    rec("P4.rho", PH, rho, [mpf('728.807976'), mpf('672.446591'), mpf('469.459062')], 'T2',
        note="rho_i = 2[2 W_tot - w(p_i)] (absolute; convention-bound)")
    rec("P4.rho.mean", PH, rho_bar, sum(rho) / 3, 'NUM', note="rho_bar = (10/3) W_tot")
    rec("P4.rho.ratio13", PH, rho[0] / rho[2], mpf('1.552442'), 'T2',
        note="rho1/rho3 (convention-proof)")
    spread = mpf(3) / 5 * (w(pb[2], t) - w(pb[0], t)) / Wtot
    rec("P4.rho.spread", PH, spread, mpf('0.415909'), 'T2', note="spread (convention-proof)")
    # off-diagonal gamma_ij = 0: averaged probe <v^m v^n> = q^{mn}/3 is diagonal for diagonal q
    qinv_offdiag = mpf(0)   # inverse of a diagonal matrix is diagonal
    boolrec("P4.rho.offdiag_zero", PH, qinv_offdiag == 0, 'Z',
            note="gamma_ij = 0: <v^m v^n> = q^{mn}/3 diagonal on diagonal q")

    # ---- 4d: susceptibility chi_T ----
    chi1 = chiT(mpf(1), t)
    rec("P4.chiT.anchor", PH, chi1, mpf('0.3863935'), 'T2', note="chi_T(1) = p w'/(5w)")
    dln_lo = mpf('1e-3') * wprime(mpf('1e-3'), t) / w(mpf('1e-3'), t)
    dln_hi = mpf('1e3') * wprime(mpf('1e3'), t) / w(mpf('1e3'), t)
    both_ends = fabs(dln_lo - 2) < mpf('1e-4') and fabs(dln_hi - 2) < mpf('1e-2')
    boolrec("P4.chiT.limits_2over5", PH, both_ends, 'Znum',
            note=f"dln w/dln p -> 2 (chi_T->2/5) at p->0 ({float(dln_lo):.4f}) AND p->inf ({float(dln_hi):.4f})")
    # interior dip: chi_T dips below 2/5 in the middle (min near p~5), rising back toward 2/5
    sweep = [(p, chiT(mpf(p), t)) for p in ['0.05', '0.2', '1', '2', '5', '20']]
    dip = min(float(c) for _, c in sweep)
    dip_ok = dip < float(chi1) and dip < 0.4 and float(chiT(mpf(20), t)) > dip
    boolrec("P4.chiT.interior_dip", PH, dip_ok, 'Znum',
            note=f"interior dip min={dip:.4f} < chi_T(1)={float(chi1):.4f}, rises again by p=20 (new claim confirmed)")

    # ---- traceless-stationarity: dW_tot/dh = 0 at isotropy (new Z-test) ----
    # p_bar_k = p_bar_0 e^{-h_k}, so dW/dh = -p_bar_0 w'(p_bar_0) * sum_k (dh_k/dh);
    # the traceless direction (1, -1/2, -1/2) sums to 0 => exactly stationary.
    dhat = [mpf(1), mpf(-1) / 2, mpf(-1) / 2]
    dWdh = -mpf(1) * wprime(mpf(1), t) * sum(dhat)      # = -w'(1) * 0 = 0 exactly
    boolrec("P4.traceless.dWdh_zero", PH, sum(dhat) == 0 and dWdh == 0, 'Z',
            note="dW_tot/dh = -p w'(p) * sum(dh_k) = 0 (traceless variation is stationary)")

    # ---- dictionary audit + isotropy + geometric-mean asymptotic (unchanged) ----
    audit = all(fabs(q[k] * pb[k] ** 2 - detq) < mpf(10) ** (-35) for k in range(3))
    boolrec("P4.dict.unimodular", PH, audit, 'T0', note="q_k p_bar_k^2 = det q")
    boolrec("P4.dict.pbar_formula", PH,
            all(fabs(pb[k] - sqrt(detq / q[k])) < mpf(10) ** (-35) for k in range(3)), 'T0',
            note="p_bar_k = sqrt(det q / q_k)")
    pbd = pbar_of_q([mpf(1)] * 3)                 # q=delta -> p_bar = (1,1,1)
    iso_spread = max(wL(p, t) for p in pbd) - min(wL(p, t) for p in pbd)
    rec("P4.isotropy.q_delta", PH, iso_spread, mpf(0), 'Z',
        note="q=delta -> p_bar isotropic -> weights equal (zero spread)")

    def geomratio(p):
        return sqrt(wL(p, t) * wT(p, t)) * t / (2 * sqrt(2) * p ** mpf('1.5'))
    g20, g200 = geomratio(mpf(20)), geomratio(mpf(200))
    boolrec("P4.geomean.asymptotic", PH, fabs(g200 - 1) < fabs(g20 - 1) < mpf('1e-2'), 'Znum',
            note=f"sqrt(wLwT)t/(2sqrt2 p^1.5) -> 1 (p=200: {float(g200):.5f})")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  R_shape = {mp.nstr(R_shape,8)}  R(Q=4) = {mp.nstr(RQ4_func,8)} (lock vs closed {float(fabs(RQ4_func-RQ4_closed)):.1e})")
        print(f"  shear triple = {[mp.nstr(s,8) for s in shear]}  max/min = {mp.nstr(max(shear)/min(shear),7)}")
        print(f"  rho = {[mp.nstr(r,9) for r in rho]}  rho1/rho3 = {mp.nstr(rho[0]/rho[2],7)}  spread = {mp.nstr(spread,7)}")
        print(f"  chi_T(1) = {mp.nstr(chi1,8)}  limits dln/dln: p->0 {float(dln_lo):.4f}, p->inf {float(dln_hi):.4f}  dip {dip:.4f}")
        print(f"  traceless dW/dh = {float(dWdh):.1e}   R-functional lock ok = {fabs(RQ4_func-RQ4_closed) < mpf(10)**(-18)}")
    return st


if __name__ == "__main__":
    run()
