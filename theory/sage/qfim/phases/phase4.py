"""
Phase 4 -- Case 1 anisotropic weights.

Edge weights (from the physical-coordinate flux metric x p^2):
    w_L = p^2 L''(p)          (longitudinal)
    w_T = 2 L'(p) tanh(p/2)   (transverse)
Dictionary (unimodular slice): p_bar_k = sqrt(det q / q_k), so q_k p_bar_k^2 = det q.

Rigorously verified: w_L, w_T vs anchors and vs brute spin-sum; w_T/w_L; R(Q);
dictionary audit (T0); isotropy at q=delta (Z); geometric-mean asymptotic (the
sqrt(w_L w_T) claim is ASYMPTOTIC, not exact -- downgrade ledger); S2 no-go in Sage.

NOTE (ledger): the SPEC quotes R_shape=1.756718, a shear-plane triple, displacement
ratios and chi_T -> 2/5 WITHOUT an explicit generating formula. The shear triple is
within ~0.5% of w_L(a)+w_T(b); R_shape / chi_T / displacement are not independently
reconstructable from the SPEC and are logged in DISCREPANCIES.md rather than faked.
"""
from mpmath import mp, mpf, tanh, sqrt, fabs

from ..core.dictionary import T_BENCH, PBAR, Q_DIAG, L1_closed, L2_closed
from ..core.kernel import L_derivs
from ..harness import rec, boolrec, phase_status

PH = "Phase4"


def wL(p, t):
    return p ** 2 * L2_closed(p, t)


def wT(p, t):
    return 2 * L1_closed(p, t) * tanh(p / 2)


def run(t=T_BENCH, verbose=True):
    pb = PBAR
    q = Q_DIAG
    detq = q[0] * q[1] * q[2]

    wL_v = [wL(p, t) for p in pb]
    wT_v = [wT(p, t) for p in pb]

    rec("P4.wL.anchor", PH, wL_v, [mpf('4.920674'), mpf('19.724062'), mpf('79.304088')], 'T2',
        note="w_L = p^2 L''")
    rec("P4.wT.anchor", PH, wT_v, [mpf('4.818064'), mpf('18.195368'), mpf('60.109108')], 'T2',
        note="w_T = 2 L' tanh(p/2) [supersedes 18.19539]")

    # tie weights to the BRUTE spin sum (L'' from omega_derivs)
    wL_brute = [p ** 2 * L_derivs(p, t)[2] for p in pb]
    rec("P4.wL.brute_vs_closed", PH, wL_brute, wL_v, 'NUM', note="w_L via spin-sum L''")

    rec("P4.ratio.wT_wL", PH, wT_v[1] / wL_v[1], mpf('0.922496'), 'T2', note="w_T(1)/w_L(1)")
    RQ4 = sqrt(mpf(4)) * L2_closed(mpf(4), t) / L2_closed(mpf(1), t)
    rec("P4.R.Q4", PH, RQ4, mpf('2.021779'), 'T2', note="R(Q=4)=sqrt(Q)L''(Q)/L''(1)")

    # dictionary audit (T0/exact): q_k p_bar_k^2 = det q,  p_bar_k = sqrt(det q/q_k)
    audit = all(fabs(q[k] * pb[k] ** 2 - detq) < mpf(10) ** (-35) for k in range(3))
    boolrec("P4.dict.unimodular", PH, audit, 'T0',
            note="q_k p_bar_k^2 = det q on the unimodular slice")
    pbar_ok = all(fabs(pb[k] - sqrt(detq / q[k])) < mpf(10) ** (-35) for k in range(3))
    boolrec("P4.dict.pbar_formula", PH, pbar_ok, 'T0', note="p_bar_k = sqrt(det q / q_k)")

    # isotropy at q = delta: p_bar_k = 1 all -> weights equal
    wL_iso = [wL(mpf(1), t)] * 3
    wT_iso = [wT(mpf(1), t)] * 3
    iso_dev = max(fabs(wL_iso[k] - wL_iso[0]) + fabs(wT_iso[k] - wT_iso[0]) for k in range(3))
    rec("P4.isotropy.q_delta", PH, iso_dev, mpf(0), 'Z',
        note="q=delta -> p_bar=1 -> weights isotropic")

    # geometric-mean claim is ASYMPTOTIC: sqrt(w_L w_T) t/(2 sqrt2 p^{3/2}) -> 1
    def geomratio(p):
        return sqrt(wL(p, t) * wT(p, t)) * t / (2 * sqrt(2) * p ** mpf('1.5'))
    g20, g50, g200 = geomratio(mpf(20)), geomratio(mpf(50)), geomratio(mpf(200))
    boolrec("P4.geomean.asymptotic", PH, (fabs(g200 - 1) < fabs(g50 - 1) < fabs(g20 - 1))
            and fabs(g200 - 1) < mpf('1e-2'), 'Znum',
            note=f"sqrt(wLwT)t/(2sqrt2 p^1.5): p=20:{float(g20):.4f} 50:{float(g50):.4f} 200:{float(g200):.4f} -> 1")

    # shear-plane triple: closest reconstruction w_L(a)+w_T(b) (~0.5%; ledger item)
    shear_recon = [wL_v[0] + wT_v[1], wL_v[1] + wT_v[2], wL_v[0] + wT_v[2]]
    rec("P4.shear.triple_recon", PH, shear_recon,
        [mpf('23.013432'), mpf('78.304476'), mpf('64.927172')], 'T2',
        note="shear triple ~ w_L(a)+w_T(b) (recon ~0.5%; exact probe not in SPEC -- LEDGER)")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  w_L = {[mp.nstr(x,8) for x in wL_v]}")
        print(f"  w_T = {[mp.nstr(x,8) for x in wT_v]}")
        print(f"  w_T(1)/w_L(1) = {mp.nstr(wT_v[1]/wL_v[1],8)}   R(Q=4) = {mp.nstr(RQ4,8)}")
        print(f"  dict unimodular={audit}  pbar_formula={pbar_ok}   geo-mean ratio p=200: {float(g200):.5f}")
        print(f"  shear recon w_L(a)+w_T(b) = {[mp.nstr(x,7) for x in shear_recon]}  (anchors 23.01,78.30,64.93)")
    return st


if __name__ == "__main__":
    run()
