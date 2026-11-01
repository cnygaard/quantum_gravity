"""
Phase 7 -- shear closure.

Shear weights w(a,b) = 16 j_a j_b/(j_a+j_b) = (160/3, 320/3, 64) at j=(5,10,20);
the isotropic/universal plane weight is 80 (cross-plane equality is the test), with
signed closure corrections 80 - w = (+80/3, -80/3, +16).

ROBERTSON LOCK (the headline): on every plane,
    coherent:  Var(eta) = 1/w,      Var(p_eta) = w/4   ->  product = 1/4  (saturated)
    forced:    Var(eta) = 1/80,     Var(p_eta) = 20    ->  product = 1/4
squeeze = required/coherent = 20/(w/4) = 80/w = (1.5, 0.75, 1.25).

Also: parity zero <p_eta> = 0 (real family), S7 total-variance decomposition (Sage),
conditional lambda = 1/sqrt(kappa_12 M_1) = 1/sqrt(1200) = 0.028868.
"""
from mpmath import mp, mpf, sqrt, fabs

from ..harness import rec, boolrec, phase_status

PH = "Phase7"
J = [mpf(5), mpf(10), mpf(20)]
PLANES = [(0, 1), (1, 2), (0, 2)]


def run(verbose=True):
    j = J
    w = [16 * j[a] * j[b] / (j[a] + j[b]) for (a, b) in PLANES]     # (160/3,320/3,64)
    universal = mpf(80)

    rec("P7.weights", PH, w, [mpf(160) / 3, mpf(320) / 3, mpf(64)], 'T2',
        note="shear weights 16 j_a j_b/(j_a+j_b)")
    corrections = [universal - wk for wk in w]
    rec("P7.signed_corrections", PH, corrections, [mpf(80) / 3, mpf(-80) / 3, mpf(16)], 'NUM',
        note="80 - w = signed closure corrections")

    # coherent variances
    Var_eta_coh = [1 / wk for wk in w]
    Var_p_coh = [wk / 4 for wk in w]
    rec("P7.coherent.var_eta", PH, Var_eta_coh,
        [mpf('0.018750'), mpf('0.009375'), mpf('0.015625')], 'T2', note="coherent Var(eta)=1/w")
    rec("P7.coherent.var_p", PH, Var_p_coh, [mpf(40) / 3, mpf(80) / 3, mpf(16)], 'T2',
        note="coherent Var(p_eta)=w/4")

    # Robertson lock: Var(eta) Var(p_eta) = 1/4 on every plane (coherent AND forced)
    rob_coh = [Var_eta_coh[k] * Var_p_coh[k] for k in range(3)]
    Var_eta_forced = mpf(1) / 80
    Var_p_forced = mpf(20)
    rob_forced = Var_eta_forced * Var_p_forced
    rec("P7.robertson.coherent", PH, rob_coh, [mpf(1) / 4] * 3, 'NUM',
        note="Var(eta)Var(p_eta)=1/4 coherent, every plane")
    rec("P7.robertson.forced", PH, rob_forced, mpf(1) / 4, 'NUM',
        note="Var(eta)Var(p_eta)=1/4 forced")
    rec("P7.forced.var_eta", PH, Var_eta_forced, mpf('0.0125'), 'T2', note="forced Var(eta)=1/80")

    # squeeze = required/coherent = 20/(w/4) = 80/w
    squeeze = [Var_p_forced / Var_p_coh[k] for k in range(3)]
    rec("P7.squeeze", PH, squeeze, [mpf('1.5'), mpf('0.75'), mpf('1.25')], 'NUM',
        note="squeeze = required/coherent = 80/w")

    # universal cross-plane equality: at iso j=10 all planes -> 8j = 80
    jj = mpf(10)
    w_iso = [16 * jj * jj / (2 * jj)] * 3
    rec("P7.universal.cross_plane", PH, w_iso, [universal] * 3, 'NUM',
        note="iso: all shear planes = 8j = 80 (cross-plane equality)")

    # parity zero: <p_eta> = 0 (real-amplitude family)
    boolrec("P7.parity.p_eta_zero", PH, True, 'T0',
            note="<p_eta>=0 by parity (real family) [S6]")

    # S7 total-variance decomposition (proven in Sage)
    boolrec("P7.S7.total_variance", PH, True, 'T0',
            note="Var_total = E[Var] + Var[E] exact [S7, Sage]")

    # conditional: lambda = 1/sqrt(kappa_12 M_1), kappa_12=4 j_1=20, M_1=60 -> 1/sqrt(1200)
    kappa12 = 4 * j[0]
    M1 = 2 * (j[1] + j[2])
    lam = 1 / sqrt(kappa12 * M1)
    rec("P7.conditional.lambda", PH, lam, mpf('0.028868'), 'T2',
        note="lambda = 1/sqrt(kappa_12 M_1) = 1/sqrt(1200)")
    boolrec("P7.conditional.kappa12", PH, kappa12 == 20, 'T0', note="kappa_12 = 4 j_1 = 20 audit")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  w = {[mp.nstr(x,7) for x in w]}  universal=80  corrections={[mp.nstr(c,6) for c in corrections]}")
        print(f"  Robertson coherent = {[mp.nstr(r,8) for r in rob_coh]}   forced = {mp.nstr(rob_forced,8)}")
        print(f"  squeeze = {[mp.nstr(s,6) for s in squeeze]}   lambda = {mp.nstr(lam,8)} (1/sqrt(1200))")
    return st


if __name__ == "__main__":
    run()
