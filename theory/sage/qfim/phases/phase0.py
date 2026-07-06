"""
Phase 0 -- Omega engine.  GATES EVERYTHING.

Checks:
  Z0(0.1) = 114.9376                              (brute Omega(0) = closed; SPEC anchor 114.9382 is a slip)
  Z1/Z0   = 3/(2t) - 1/4 = 14.75  at t=0.1        (brute moment vs closed, vs anchor)
  k_t(chi) = (chi/sin chi) e^{-chi^2/t} shape     (compact heat kernel asymptotic)
  F_{1/2}(chi) = cos^2 chi exactly, t-independent (Z)
  flat-limit  L''(0) = 2/t - 1/3 = 19.6667        (brute vs closed)
"""
from mpmath import mp, mpf, exp, sin, cos, sqrt, fabs

from ..core.dictionary import (T_BENCH, nmax, Z0, Zratio_closed, flat_iso,
                               L2_closed)
from ..core.kernel import Omega_x, L_derivs, rot, dag
from ..harness import rec, boolrec, phase_status

PH = "Phase0"


def Z_moments(t):
    """Z0 = sum n^2 e^{-t(n^2-1)/4},  Z1 = sum n^2 (n^2-1)/4 e^{-t(n^2-1)/4}."""
    N = nmax(t)
    z0 = mpf(0)
    z1 = mpf(0)
    for n in range(1, N + 1):
        wc = mpf(n * n) * exp(-t * mpf(n * n - 1) / 4)
        z0 += wc
        z1 += wc * mpf(n * n - 1) / 4
    return z0, z1


def Kc(chi, t):
    """Compact SU(2) heat kernel  sum_n n e^{-t(n^2-1)/4} sin(n chi)/sin(chi)."""
    N = nmax(t)
    s = sin(chi)
    tot = mpf(0)
    for n in range(1, N + 1):
        tot += n * exp(-t * mpf(n * n - 1) / 4) * sin(n * chi) / s
    return tot


def F_half(chi):
    """Fundamental (spin-1/2) fidelity of two rotations separated by 2 chi about z."""
    g1 = rot([0, 0, 0])
    g2 = rot([0, 0, 2 * chi])
    M = dag(g1) * g2
    x = (M[0, 0] + M[1, 1]) / 2
    return fabs(x) ** 2


def run():
    t = T_BENCH

    # ---- Z0 ----
    z0_brute = Omega_x(mpf(1), t)                 # Omega(0): U_{n-1}(1)=n -> sum n^2 w
    z0_closed = Z0(t)
    rec("P0.Z0.sum_vs_closed", PH, z0_brute, z0_closed, 'T1',
        note="Omega(0) brute spin-sum vs 2 sqrt(pi) e^{t/4} t^{-3/2}")
    # SPEC hand value 114.9382 is a rounding slip; true Z0(0.1)=114.93764... (114.9376).
    # This advisory T2 log intentionally flags the 4.8e-6 gap (see report/DISCREPANCIES.md);
    # the authoritative check is the T1 sum_vs_closed above, which passes at error 0.
    rec("P0.Z0.anchor", PH, z0_brute, mpf('114.9382'), 'T2',
        note="SPEC hand anchor 114.9382 -- rounding slip; true value 114.9376")

    # ---- Z1/Z0 ----
    z0m, z1m = Z_moments(t)
    ratio = z1m / z0m
    rec("P0.Zratio.sum_vs_closed", PH, ratio, Zratio_closed(t), 'T1',
        note="<j(j+1)> brute vs 3/(2t)-1/4")
    rec("P0.Zratio.anchor", PH, ratio, mpf('14.75'), 'T2', note="anchor 14.75")

    # ---- k_t heat-kernel shape (compact), residual should shrink with t ----
    grid = [mpf(k) / 10 for k in range(2, 25)]     # chi in (0.2, 2.4)
    devs = {}
    for tt in [mpf('0.5'), mpf('0.1'), mpf('0.02')]:
        z0_tt, _ = Z_moments(tt)
        maxdev = mpf(0)
        for chi in grid:
            khat = Kc(chi, tt) / z0_tt
            shape = (chi / sin(chi)) * exp(-chi * chi / tt)
            maxdev = max(maxdev, fabs(khat - shape))
        devs[str(tt)] = maxdev
    # monotone decrease with t is the physics; record the t=0.02 residual as the gate-ish check
    rec("P0.kt.shape_t0.02", PH, devs['0.02'], mpf(0), 'T1',
        note="max|khat-(chi/sin)e^{-chi^2/t}| over grid at t=0.02")
    # convergence: 0.5->0.1 drops ~28 orders; below t~0.1 the residual is precision-floored
    boolrec("P0.kt.monotone", PH,
            devs['0.5'] > devs['0.1'] and devs['0.02'] < mpf(10) ** (-30), 'Z',
            note=f"residual t=0.5:{float(devs['0.5']):.2e} -> t=0.1:{float(devs['0.1']):.2e} -> t=0.02:{float(devs['0.02']):.2e} (floor)")

    # ---- F_1/2 = cos^2 chi, exact, t-independent ----
    maxdiff = mpf(0)
    for chi in grid:
        maxdiff = max(maxdiff, fabs(F_half(chi) - cos(chi) ** 2))
    rec("P0.Fhalf.cos2", PH, maxdiff, mpf(0), 'Z',
        note="max|F_1/2(chi) - cos^2 chi| over grid (exact, no t)")

    # ---- flat-limit isotropy: L''(0) = 2/t - 1/3 ----
    zsmall = mpf('1e-4')
    _, _, L2_brute = L_derivs(zsmall, t)
    rec("P0.flat.L2_brute_vs_closed", PH, L2_brute, L2_closed(zsmall, t), 'T1',
        note="brute L''(1e-4) vs closed")
    rec("P0.flat.iso_value", PH, L2_closed(zsmall, t), flat_iso(t), 'T2',
        note="L''(0+) -> 2/t-1/3 = 19.6667")

    st, ntot, nfail = phase_status(PH)
    print(f"[{PH}] {ntot} records, gate={st}")
    print(f"  Z0 brute  = {mp.nstr(z0_brute, 12)}  closed = {mp.nstr(z0_closed, 12)}")
    print(f"  Z1/Z0     = {mp.nstr(ratio, 12)}  closed = {mp.nstr(Zratio_closed(t), 12)}")
    print(f"  k_t resid : t=0.5 {float(devs['0.5']):.2e}  t=0.1 {float(devs['0.1']):.2e}  t=0.02 {float(devs['0.02']):.2e}")
    print(f"  F_1/2 max dev from cos^2 = {float(maxdiff):.2e}")
    print(f"  L''(0+)   = {mp.nstr(L2_closed(zsmall,t),12)}  (2/t-1/3 = {mp.nstr(flat_iso(t),12)})")
    return st


if __name__ == "__main__":
    run()
