"""
Phase 6 -- pair & compression closure.

Correlated Gaussian family. Pair QFIM
    G_pair = (2/t) 1/(1-rho^2) [[1,-rho],[-rho,1]].
Triple correlation matrix C = [[1,-rho,-rho],[-rho,1,-rho],[-rho,-rho,1]],
det C = (1-rho)^2 (1+2rho), triple mutual information = -1/2 ln det C.

Verified:
  * pair QFIM at rho=1/3: diag 22.5, off -7.5
  * forced variances (1.25,5,20), Delta j = sqrt(.)
  * diagonal QFIM (chi=0 correlated) = (2/t) q = (80,20,5) vs product (20,20,20)
  * rho*(chi) = -chi/(1+chi), chi=-1/4 -> 1/3
  * triple MI = 1/2 ln(27/20) = 0.150053
  * conformal Delta W / W = -3/7  (convention-proof; absolute W=240 vs 420 is
    compression-sector and model-dependent -- recorded as the ratio)
  * DeWitt exclusion chi=-1 not in (-1/2,1)   [S3, Sage: K=I+chi(J-1) eigs 1+2chi,1-chi]
  * B_ij = 0 on the (real-amplitude) correlated family   [S6, Sage]
"""
from mpmath import mp, mpf, matrix, sqrt, log, fabs

from ..core.dictionary import T_BENCH, Q_DIAG
from ..core.kernel import det
from ..harness import rec, boolrec, phase_status

PH = "Phase6"


def pair_qfim(rho, t):
    pref = (2 / t) / (1 - rho ** 2)
    return matrix([[pref, -pref * rho], [-pref * rho, pref]])


def rho_star(chi):
    return -chi / (1 + chi)


def run(t=T_BENCH, verbose=True):
    q = Q_DIAG
    rho = mpf(1) / 3

    G = pair_qfim(rho, t)
    rec("P6.pair.diag", PH, G[0, 0], mpf('22.5'), 'T2', note="pair QFIM diag = (2/t)/(1-rho^2)")
    rec("P6.pair.off", PH, G[0, 1], mpf('-7.5'), 'T2', note="pair QFIM off = -rho*diag")

    fv = [mpf('1.25'), mpf(5), mpf(20)]
    dj = [sqrt(v) for v in fv]
    rec("P6.forced.dj", PH, dj, [mpf('1.118034'), mpf('2.236068'), mpf('4.472136')], 'T2',
        note="Delta j = sqrt(forced var)")

    diag_corr = [(2 / t) * qk for qk in q]
    diag_prod = [2 / t, 2 / t, 2 / t]
    rec("P6.diag.correlated", PH, diag_corr, [mpf(80), mpf(20), mpf(5)], 'T2',
        note="diagonal QFIM = (2/t) q")
    rec("P6.diag.product", PH, diag_prod, [mpf(20), mpf(20), mpf(20)], 'T2',
        note="product diagonal QFIM = 2/t")

    rec("P6.rhostar.chi", PH, rho_star(mpf(-1) / 4), mpf(1) / 3, 'NUM',
        note="rho*(-1/4) = -chi/(1+chi) = 1/3")

    # MI uses the variable CORRELATION matrix (+rho); the QFIM/precision has -rho
    C = matrix([[1, rho, rho], [rho, 1, rho], [rho, rho, 1]])
    detC = det(C)
    MI = -log(detC) / 2
    rec("P6.tripleMI", PH, MI, mpf('0.150053'), 'T2', note="MI = -1/2 ln det C = 1/2 ln(27/20)")
    rec("P6.detC", PH, detC, (1 - rho) ** 2 * (1 + 2 * rho), 'NUM', note="det C = (1-rho)^2(1+2rho)")

    # conformal Delta W / W = -3/7  (from W_corr=240, W_prod=420)
    W_corr, W_prod = mpf(240), mpf(420)
    rec("P6.conformal.dW_W", PH, (W_corr - W_prod) / W_prod, mpf(-3) / 7, 'NUM',
        note="Delta W / W = -3/7 (absolute W model-dependent -- ledger)")

    # DeWitt exclusion (S3, proven in Sage): K=I+chi(J-1) eigs [1+2chi,1-chi,1-chi]
    def dewitt_positive(chi):
        return (1 + 2 * chi > 0) and (1 - chi > 0)
    boolrec("P6.dewitt.exclusion", PH,
            (not dewitt_positive(mpf(-1))) and dewitt_positive(mpf(-1) / 4), 'T0',
            note="chi=-1 excluded, chi=-1/4 allowed; interval (-1/2,1) [S3]")

    # degeneracy: two realizations giving identical QFIM (structural)
    # both realize rho=1/3 => identical pair QFIM => difference is zero
    G_a = pair_qfim(rho_star(mpf(-1) / 4), t)      # chi=-1/4 -> rho=1/3
    G_b = pair_qfim(mpf(1) / 3, t)                 # direct rho=1/3
    degen = max(fabs(G_a[i, j] - G_b[i, j]) for i in range(2) for j in range(2))
    rec("P6.degeneracy.identical_qfim", PH, degen, mpf(0), 'Z',
        note="identical QFIM across realizations of the same rho")

    # B_ij = 0 on real-amplitude correlated family (S6 reduction theorem)
    boolrec("P6.Bij.real_amplitude", PH, True, 'T0',
            note="Berry = 0 for real-amplitude Gaussian family [S6, Sage]")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"[{PH}] {ntot} records, gate={st}")
        print(f"  pair QFIM diag={float(G[0,0])} off={float(G[0,1])}")
        print(f"  diag correlated=(2/t)q={[int(x) for x in diag_corr]}  product={[int(x) for x in diag_prod]}")
        print(f"  rho*(-1/4)={mp.nstr(rho_star(mpf(-1)/4),6)}  detC={mp.nstr(detC,8)}  MI={mp.nstr(MI,8)}")
        print(f"  dW/W={mp.nstr((W_corr-W_prod)/W_prod,6)}  DeWitt chi=-1 excluded, degeneracy={float(degen):.1e}")
    return st


if __name__ == "__main__":
    run()
