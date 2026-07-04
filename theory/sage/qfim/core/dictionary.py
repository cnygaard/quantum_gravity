"""
core/dictionary.py -- single source of truth for constants, conventions, and
closed-form expressions for the QFIM -> spatial-metric verification sprint.

ALL conventions live HERE so factor-of-2/4 drift is impossible: every phase
imports from this module and never redefines a convention locally.

Conventions (CANONICAL -- never deviate):
  G    = 4 Re Q                          (QFIM = quantum Fisher information metric)
  B    = 2 Im Q                          (Berry curvature)
  ln F = -1/4 G_AB d^A d^B + O(d^3)      (fidelity expansion)
  Q_ab = d_{x_a} d_{y_b} ln N(x,y)|x=y   (x = bra label, y = ket label)
  N(x,y) = <psi(x)|psi(y)> = Omega( X = 1/2 tr(g(x)^dag g(y)) )
  J : (theta, beta) -> (-beta, theta)    (complex structure on label space)

Generating function (valid mod O(e^{-4 pi^2 / t})):
  Omega(z;t) = sum_{n>=1} n e^{-t(n^2-1)/4} sinh(n z)/sinh(z)
             = sum_{n>=1} n e^{-t(n^2-1)/4} U_{n-1}(cosh z)   (Chebyshev-U form)
  L(z)   = ln Omega
  L'(z)  = 2 z/t + 1/z - coth z
  L''(z) = 2/t - 1/z^2 + 1/sinh^2 z
  L(z)   = z^2/t + ln(z/sinh z) + ln Z0                (integrated; const from Omega(0))
  Z0(t)  = 2 sqrt(pi) e^{t/4} t^{-3/2}                 (= Omega(0))
"""
from mpmath import mp, mpf, sqrt, pi, exp, sinh, cosh, coth, log, ceil

mp.dps = 40

# ---- benchmark point ----
T_BENCH = mpf('0.1')
PBAR = [mpf(1) / 2, mpf(1), mpf(2)]        # mean momenta (rapidities) p_bar_a
Q_DIAG = [mpf(4), mpf(1), mpf(1) / 4]      # target spatial metric diag(4,1,1/4)
EPS2_A2 = mpf(1)                           # calibration eps^2 / a^2


def jbar(pbar=None, t=T_BENCH):
    """spin labels j_a = p_bar_a / t  -> (5, 10, 20) at the benchmark."""
    if pbar is None:
        pbar = PBAR
    return [p / t for p in pbar]


# ---- tolerance tiers ----
TOL = {
    'T0': mpf(0),               # symbolic exact (Sage); recorded as boolean pass
    'T1': mpf(10) ** (-30),     # sum vs closed form (t <= 0.5), analytic
    'T2': mpf(10) ** (-5),      # hand anchors, RELATIVE, log-don't-fail
    'Z':  mpf(10) ** (-33),     # structural zeros, analytic
    'NUM': mpf(10) ** (-18),    # finite-difference kernel Hessian vs closed form
    'Znum': mpf(10) ** (-16),   # structural zeros measured by finite differences
}


def nmax(t):
    """spin cutoff: ceil(sqrt(92/t)) + 15 in j, returned as n_max = 2 j_max + 1."""
    jm = int(ceil(sqrt(mpf(92) / t))) + 15
    return 2 * jm + 1


# ---- closed forms (asymptotic; exact mod O(e^{-4 pi^2 / t})) ----
def Z0(t=T_BENCH):
    return 2 * sqrt(pi) * exp(t / 4) * t ** mpf('-1.5')


def L_closed(z, t=T_BENCH):
    return z * z / t + log(z / sinh(z)) + log(Z0(t))


def L1_closed(z, t=T_BENCH):
    return 2 * z / t + 1 / z - coth(z)


def L2_closed(z, t=T_BENCH):
    return 2 / t - 1 / z ** 2 + 1 / sinh(z) ** 2


def Zratio_closed(t=T_BENCH):
    """Z1/Z0 = <j(j+1)> = 3/(2t) - 1/4."""
    return mpf(3) / (2 * t) - mpf(1) / 4


def flat_iso(t=T_BENCH):
    """L''(0) = 2/t - 1/3  (flat-limit isotropic metric coefficient)."""
    return 2 / t - mpf(1) / 3
