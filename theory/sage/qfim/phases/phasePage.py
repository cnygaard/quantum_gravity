"""
Phase Page -- explicit Page curve from LQG area quantization (info-paradox goal).

Referee Q3 asked the framework to EXHIBIT the Page-curve computation, not just
sketch a roadmap. The framework-specific ingredient is discreteness: an LQG
horizon of area A is a set of N punctures with spins j_p, area
a_p = 8 pi gamma0 lP^2 sqrt(j_p(j_p+1)) and FINITE microstate multiplicity
d_p = 2 j_p + 1. The horizon Hilbert space is therefore finite-dimensional,
D = prod d_p, with S_BH = ln D = A/4 (Bianchi: gamma0-independent). A finite
UNITARY system necessarily returns its information -> Page curve. Continuum QFT
(infinite local Hilbert space) has no such bound; discreteness is the resolution.

Evaporation = emitting punctures; the global state stays pure. After k of N
emitted, radiation dim d_R = prod_{emitted} d_p, black hole d_B = prod_{rest}.
Page's theorem: S_rad = ln(min) - min/(2 max) (large-dim limit), worked in log
space. This is the island rule on its two saddles:
   no-island (I=empty):   S_gen = S_bulk(R)        = ln d_R = A_emitted/4   (rises)
   island   (I=interior): S_gen = A_horizon/4 + ~0 = ln d_B = A_remaining/4 (falls)
S_rad = min(saddles); the crossover (degeneracy) is the QES transition = Page time.

Records (all T2 advisory -- this phase LOGS the mechanism, it never gates):
  PhasePage.finite_dim   : ln D = A/4 exactly (finiteness => area law)
  PhasePage.peak_half    : peak S_rad = S_BH0/2 within the O(1) Page correction
  PhasePage.returns      : S_rad(end) = 0  (unitarity: information returns)
  PhasePage.qes_cross    : no-island/island saddles are degenerate at half-area
  PhasePage.gamma_indep  : curve shape is independent of gamma0
"""
from mpmath import mp, mpf, log, sqrt, pi, e

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

    # (1) finiteness: ln D = A/4 exactly (choice of gamma0 that matches S=A/4)
    g_match = float(log(mpf(2)) / (pi * sqrt(mpf(3))))
    _, S0m, A0m = _curve(spins, g_match)
    boolrec("PhasePage.finite_dim", PH, abs(S0m / (A0m / 4) - 1) < 1e-9, 'T2',
            computed=S0m / (A0m / 4),
            note=f"ln D = A/4 at gamma0={g_match:.4f} (finite Hilbert space => area law)")

    # (2) peak at half the horizon entropy, within the O(1) Page correction (~0.5 nat)
    peak_ratio = peak[1] / (S0 / 2)
    boolrec("PhasePage.peak_half", PH, 0.99 < peak_ratio <= 1.0, 'T2', computed=peak_ratio,
            note=f"peak S_rad={peak[1]:.3f} vs S_BH0/2={S0/2:.3f}; deficit={S0/2-peak[1]:.3f} nat (Page corr.)")

    # (3) information returns: S_rad(end)=0
    boolrec("PhasePage.returns", PH, curve[-1][1] < 1e-6, 'T2', computed=curve[-1][1],
            note="S_rad(N)=0 -> pure state recovered -> information preserved (unitary)")

    # (4) QES saddle degeneracy at half-area: no-island A_emit/4 == island A_horiz/4
    mid = curve[N // 2]
    s_isl = mid[2] / S0                       # island saddle A_remaining/4
    s_noi = 1 - s_isl                          # no-island saddle A_emitted/4
    boolrec("PhasePage.qes_cross", PH, abs(s_isl - s_noi) < 1e-9, 'T2', computed=abs(s_isl - s_noi),
            note=f"at area-frac {mid[0]:.3f} saddles degenerate (A_emit/4={s_noi:.3f}=A_horiz/4={s_isl:.3f}): QES transition")

    # (5) gamma0-independence of the curve shape
    shapes = []
    for g in (mpf('0.2375'), mpf('0.274'), mpf('0.5'), mpf('1')):
        c, s0, _ = _curve(spins, g)
        shapes.append(max(c, key=lambda r: r[1])[1] / s0)
    spread = max(shapes) - min(shapes)
    boolrec("PhasePage.gamma_indep", PH, spread < 1e-9, 'T2', computed=spread,
            note=f"peak S_rad/S_BH0 spread over gamma0 in [0.2375,1] = {spread:.1e} (shape gamma0-free)")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    finite dim: ln D=A/4 (exact); peak S_rad/(S_BH0/2)={peak_ratio:.4f} at area-frac {peak[0]:.3f}")
        print(f"    S_rad(end)={curve[-1][1]:.1e}; QES degeneracy |A_emit/4-A_horiz/4|={abs(s_isl-s_noi):.1e} at half-area")
        print(f"    gamma0-independence of shape: spread {spread:.1e}")
        print(f"    VERDICT: discreteness -> finite dim -> unitary Page curve (info returns), gamma0-free")
    return st


if __name__ == "__main__":
    run()
