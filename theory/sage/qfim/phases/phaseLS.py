"""
Phase LS -- finite-j Livine-Speziale coherent-intertwiner QFIM (referee dossier).

Independent finite-spin check of the shape-QFIM weight 16 j_a j_b/(j_a+j_b), which
Phase 5 establishes only from asymptotic reduced-mass eigenvalue algebra. A literal
2-face pair is degenerate (QFIM = 0 / undefined); the well-posed object is the
4-valent antipodal tetrahedron: active faces (+z, -z) with spin j_a and (+b, -b) with
spin j_b, closure Sum_k j_k n_k = 0 automatic. The relative-angle QFIM comes from the
Haar-integrated Livine-Speziale overlap
    N(t,t') = Int_{SU(2)} dg  (u11)^{2 j_a}(u22)^{2 j_a}(o_b)^{2 j_b}(o_{-b})^{2 j_b},
    G_theta = 8[ 1 - |N(t,t')| / sqrt(N(t,t) N(t',t')) ] / dtheta^2 ,
with u = D_{1/2}(g) the fundamental SU(2) matrix and o the spinor overlaps.

DISCRIMINATOR (convention/normalization-free): the ratio  G(10,20)/G(10,10)  is
  1.333  if the weight is the HARMONIC mean  j_a j_b/(j_a+j_b),
  1.414  if the GEOMETRIC mean  sqrt(j_a j_b),
  1.500  if the SUM  j_a+j_b.
Sanity: symmetric cases scale 1:2:4 (linear in j). Anchor: 2-face singlet norm ~ 1/(2j+1).

numpy float64 Gauss-Legendre Haar quadrature (Euler ZYZ).  All records tier T2 (advisory).
"""
import numpy as np
from mpmath import mpf
from ..harness import boolrec, phase_status

PH = "PhaseLS"


def _gl(n, a, b):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def _grids(nA=48, nB=64, nC=48):
    a, wa = _gl(nA, 0, 2 * np.pi)
    b, wb = _gl(nB, 0, np.pi)
    c, wc = _gl(nC, 0, 2 * np.pi)
    A, B, C = np.meshgrid(a, b, c, indexing='ij')
    W = wa[:, None, None] * wb[None, :, None] * wc[None, None, :] * np.sin(B)   # Haar (unnormalized)
    cb, sb = np.cos(B / 2), np.sin(B / 2)
    eacm, eacp = np.exp(-0.5j * (A + C)), np.exp(0.5j * (A + C))
    eamcm, eamcp = np.exp(-0.5j * (A - C)), np.exp(0.5j * (A - C))
    u = (eacm * cb, -eamcm * sb, eamcp * sb, eacp * cb)                          # D_{1/2}(a,b,c)
    return W, u


def _ovl(u, psi, psip):
    """<psi| u |psip> for real 2-spinors psi, psip."""
    u11, u12, u21, u22 = u
    p0, p1 = psi; q0, q1 = psip
    return p0 * (u11 * q0 + u12 * q1) + p1 * (u21 * q0 + u22 * q1)


def _sp(th):      return (np.cos(th / 2), np.sin(th / 2))        # +b-hat(theta), phi=0
def _spneg(th):   return (np.sin(th / 2), -np.cos(th / 2))       # -b-hat(theta)


def _N(W, u, ja, jb, th, thp):
    u11, u12, u21, u22 = u
    fa = (u11 ** (2 * ja)) * (u22 ** (2 * ja))                   # +z, -z faces (spin j_a), fixed
    o3 = _ovl(u, _sp(th), _sp(thp))
    o4 = _ovl(u, _spneg(th), _spneg(thp))
    return np.sum(W * fa * (o3 ** (2 * jb)) * (o4 ** (2 * jb)))


def _Gtheta(W, u, ja, jb, th0=np.pi / 2, dth=1e-3):
    Naa = _N(W, u, ja, jb, th0, th0).real
    Nbb = _N(W, u, ja, jb, th0 + dth, th0 + dth).real
    Nab = abs(_N(W, u, ja, jb, th0, th0 + dth))
    return 8 * (1 - Nab / np.sqrt(Naa * Nbb)) / dth ** 2


def run(verbose=True):
    W, u = _grids(nA=48, nB=80, nC=48)

    # ---- anchor: 2-face singlet norm  ~ 1/(2j+1)  (integrator validation) ----
    u11, u12, u21, u22 = u
    sing = {j: float(np.sum(W * (u11 ** (2 * j)) * (u22 ** (2 * j))).real) for j in (5, 10, 20)}
    prod = {j: sing[j] * (2 * j + 1) for j in sing}
    anc_ok = max(abs(prod[j] - prod[5]) for j in prod) < 1e-6 * abs(prod[5])
    boolrec("PhaseLS.anchor.singlet", PH, anc_ok, 'T2', computed=prod[10],
            note=f"singlet norm*(2j+1) const across j: {[round(prod[j],8) for j in (5,10,20)]}")

    # ---- 4-valent tetrahedron QFIM at the four spin points ----
    cases = [(5, 5), (10, 10), (20, 20), (10, 20)]
    G = {c: float(_Gtheta(W, u, c[0], c[1])) for c in cases}
    for (ja, jb) in cases:
        boolrec(f"PhaseLS.G.{ja}_{jb}", PH, True, 'T2', computed=G[(ja, jb)],
                note=f"G_theta numeric (j_a={ja}, j_b={jb})")

    # symmetric linear-in-j scaling (all candidates agree): expect 1:2:4
    r_sym = [G[(10, 10)] / G[(5, 5)], G[(20, 20)] / G[(10, 10)]]
    sym_ok = all(abs(r - 2.0) < 0.1 for r in r_sym)
    boolrec("PhaseLS.scaling.linear", PH, sym_ok, 'T2', computed=float(r_sym[0]),
            note=f"symmetric ratios (expect 2,2): {[round(r,4) for r in r_sym]}")

    # convergence: asymmetric ratio r(j) = G(j,2j)/G(j,j) -> 4/3 (harmonic) with O(1/j)
    conv = [(j, float(_Gtheta(W, u, j, 2 * j)) / G[(j, j)]) for j in (5, 10, 20)]
    conv_ok = abs(conv[-1][1] - 4.0 / 3.0) < 0.02
    boolrec("PhaseLS.converge.asym", PH, conv_ok, 'T2', computed=conv[-1][1],
            note=f"G(j,2j)/G(j,j) -> 4/3=1.3333: {[(j, round(r, 4)) for j, r in conv]}")

    # DISCRIMINATOR: G(10,20)/G(10,10)  ->  harmonic 1.333 | geometric 1.414 | sum 1.500
    r = G[(10, 20)] / G[(10, 10)]
    preds = {'harmonic': (10 * 20 / 30) / (10 * 10 / 20),
             'geometric': np.sqrt(10 * 20) / np.sqrt(10 * 10),
             'sum': (10 + 20) / (10 + 10)}
    best = min(preds, key=lambda k: abs(r - preds[k]))
    is_harmonic = (best == 'harmonic') and abs(r - preds['harmonic']) < abs(r - preds['geometric'])
    boolrec("PhaseLS.discriminator", PH, is_harmonic, 'T2', computed=float(r),
            note=(f"G(10,20)/G(10,10)={r:.4f}; harmonic={preds['harmonic']:.4f} "
                  f"geometric={preds['geometric']:.4f} sum={preds['sum']:.4f} -> {best}"))

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    anchor singlet*(2j+1): {[round(prod[j],6) for j in (5,10,20)]}  ok={anc_ok}")
        print(f"    G_theta: " + ", ".join(f"{c}={G[c]:.5f}" for c in cases))
        print(f"    symmetric ratios (->2,2): {[round(x,4) for x in r_sym]}")
        print(f"    asym convergence G(j,2j)/G(j,j) (->1.3333): {[(j,round(r,4)) for j,r in conv]}")
        print(f"    DISCRIMINATOR G(10,20)/G(10,10)={r:.4f}  "
              f"[harm {preds['harmonic']:.3f} | geom {preds['geometric']:.3f} | sum {preds['sum']:.3f}]  -> {best.upper()}")
    return st


if __name__ == "__main__":
    run()
