"""
Phase 1 -- edge blocks.  Brute-force 6x6 fidelity Hessian from the master kernel
vs closed forms vs hand anchors, at t=0.1, p_bar=1.

delta = (dp_x, dp_y, dp_z, phi_x, phi_y, phi_z) right-mult displacement from
g_base = boost(p_bar zhat).  G = 4 Re Q.

Index map:  0 dp_x  1 dp_y  2 dp_z  3 phi_x  4 phi_y  5 phi_z
Blocks (verified against the exploratory 6x6):
  longitudinal    dp_z (2)                G_par  = L''
  transverse mom  dp_x (0), dp_y (1)      G_perp = L' coth p_bar
  cross           (dp_x,phi_y)=(0,4)      G_cross= -L'           (|.| = L')
  transverse 2x2  {dp_x, phi_y}           eigs   = L' e^{+-p}/sinh p
"""
from mpmath import mp, mpf, cosh, sinh, sqrt, e, fabs

from ..core.dictionary import T_BENCH
from ..core.kernel import boost, displace, xarg, qgt, G_from_Q, L_derivs
from ..harness import rec, boolrec, phase_status

PH = "Phase1"


def edge_family(pbar, t):
    gbase = boost([0, 0, pbar])

    def g_of(d):
        return displace(gbase, [d[0], d[1], d[2]], [d[3], d[4], d[5]])
    return g_of


def eig2(a, b, c):
    """eigenvalues of [[a,c],[c,b]]."""
    tr = a + b
    disc = sqrt((a - b) ** 2 + 4 * c * c)
    return (tr + disc) / 2, (tr - disc) / 2


def run(pbar=mpf(1), t=T_BENCH, verbose=True):
    _, Lp, Lpp = L_derivs(pbar, t)                    # brute spin-sum L', L''
    Gperp_closed = Lp * cosh(pbar) / sinh(pbar)
    eplus = Lp * e ** pbar / sinh(pbar)
    eminus = Lp * e ** (-pbar) / sinh(pbar)

    g_of = edge_family(pbar, t)
    Q = qgt(g_of, 6, t)
    G = G_from_Q(Q)

    Gpar = G[2, 2]
    Gperp = G[0, 0]
    Gcross = G[0, 4]
    lam_hi, lam_lo = eig2(G[0, 0], G[4, 4], G[0, 4])
    lam_hi, lam_lo = max(lam_hi, lam_lo), min(lam_hi, lam_lo)

    # kernel Hessian vs closed forms
    rec("P1.Gpar.vs_closed", PH, Gpar, Lpp, 'NUM', note="G_par = L''")
    rec("P1.Gperp.vs_closed", PH, Gperp, Gperp_closed, 'NUM', note="G_perp = L' coth p")
    rec("P1.Gcross.vs_closed", PH, fabs(Gcross), Lp, 'NUM', note="|G_cross| = L'")
    rec("P1.Teig.vs_closed", PH, [lam_hi, lam_lo], [eplus, eminus], 'NUM',
        note="transverse eigs L' e^{+-p}/sinh p")

    # hand anchors (T2, log-don't-fail)
    rec("P1.Gpar.anchor", PH, Gpar, mpf('19.724062'), 'T2')
    rec("P1.Gperp.anchor", PH, Gperp, mpf('25.849679'), 'T2')
    rec("P1.Gcross.anchor", PH, fabs(Gcross), mpf('19.686965'), 'T2')
    rec("P1.Teig.anchor", PH, [lam_hi, lam_lo],
        [mpf('45.536644'), mpf('6.162715')], 'T2')

    # exact 1D families (kernel geometry identities), T1
    beta = mpf('0.3')
    g1 = boost([0, 0, pbar])
    g2l = displace(g1, [0, 0, beta])
    rec("P1.long.zeta12", PH, xarg(g1, g2l), cosh(pbar + beta / 2), 'T1',
        note="cosh zeta_12 = cosh(p+beta/2)")
    g2t = displace(g1, [beta, 0, 0])
    rec("P1.trans.zeta12", PH, xarg(g1, g2t), cosh(pbar) * cosh(beta / 2), 'T1',
        note="cosh zeta_12 = cosh p cosh(beta/2)")
    rec("P1.trans.zeta22", PH, xarg(g2t, g2t), cosh(pbar) * cosh(beta), 'T1',
        note="cosh zeta_22 = cosh p cosh beta")

    # structural zeros: everything off the {diag, (0,4),(4,0),(1,3),(3,1)} blocks
    nonzero_offdiag = {(0, 4), (4, 0), (1, 3), (3, 1)}
    zmax = mpf(0)
    for i in range(6):
        for j in range(6):
            if i != j and (i, j) not in nonzero_offdiag:
                zmax = max(zmax, fabs(G[i, j]))
    boolrec("P1.zeros.offblock", PH, zmax < mpf(10) ** (-16), 'Znum',
            note=f"max off-block |G_ij| = {float(zmax):.2e}")
    rec("P1.Gperp.isotropy", PH, G[1, 1], Gperp, 'NUM',
        note="G[dpx]=G[dpy] transverse momentum isotropy")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        errmax = max(float(fabs(Gpar - Lpp)), float(fabs(Gperp - Gperp_closed)),
                     float(fabs(fabs(Gcross) - Lp)))
        print(f"[{PH}] p_bar={float(pbar)}  {ntot} records, gate={st}")
        print(f"  G_par   = {mp.nstr(Gpar,12)}   L''    = {mp.nstr(Lpp,12)}")
        print(f"  G_perp  = {mp.nstr(Gperp,12)}   L'coth = {mp.nstr(Gperp_closed,12)}")
        print(f"  G_cross = {mp.nstr(Gcross,12)}   -L'    = {mp.nstr(-Lp,12)}")
        print(f"  T-eigs  = {mp.nstr(lam_hi,12)} / {mp.nstr(lam_lo,12)}   closed {mp.nstr(eplus,10)}/{mp.nstr(eminus,10)}")
        print(f"  max Hessian-vs-closed error = {errmax:.2e}   (off-block zeros {float(zmax):.2e})")
    return st


if __name__ == "__main__":
    run()
