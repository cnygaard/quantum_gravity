"""
core/kernel.py -- the master ground-truth overlap kernel.

A coherent-state label is  g = exp(1/2 p.sigma) . u  in SL(2,C), with u in SU(2)
the "position" (holonomy angle) and p in R^3 the "momentum" (flux / boost rapidity).

  Overlap:   N(g1,g2) = Omega( X ),  X = 1/2 tr(g1^dag g2)   (complex, branch-free)
  Norm:      N(g,g)   = Omega( cosh|p| )                     (real)
  Fidelity:  ln F     = 2 Re ln N12 - ln N11 - ln N22
  QGT:       Q_ab = d_{x_a} d_{y_b} ln N(x,y),   G = 4 Re Q,  B = 2 Im Q

Omega is evaluated branch-free via Chebyshev-U in X = cosh z:
  Omega(X) = sum_{n>=1} n e^{-t(n^2-1)/4} U_{n-1}(X),
  U_0 = 1, U_1 = 2X, U_k = 2X U_{k-1} - U_{k-2}.

This module has NO physics conventions of its own beyond the kernel definition;
all numbers come from core.dictionary.
"""
from mpmath import (mp, mpf, mpc, matrix, sqrt, sinh, cosh, coth, exp, log,
                    cos, sin, asinh, conj, re, im, fabs)

from .dictionary import T_BENCH, nmax

IU = mpc(0, 1)
ID2 = matrix([[1, 0], [0, 1]])


# ---------------------------------------------------------------- group elements
def pauli_dot(v):
    """v.sigma = [[vz, vx - i vy],[vx + i vy, -vz]]."""
    vx, vy, vz = v
    return matrix([[vz, vx - IU * vy], [vx + IU * vy, -vz]])


def boost(p):
    """exp(1/2 p.sigma), a Hermitian positive SL(2,C) element (the momentum part)."""
    P = sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
    if P == 0:
        return matrix(ID2.tolist())
    return cosh(P / 2) * ID2 + (sinh(P / 2) / P) * pauli_dot(p)


def rot(phi):
    """exp(-1/2 i phi.sigma) in SU(2), the position (holonomy angle) part."""
    F = sqrt(phi[0] ** 2 + phi[1] ** 2 + phi[2] ** 2)
    if F == 0:
        return matrix(ID2.tolist())
    return cos(F / 2) * ID2 - IU * (sin(F / 2) / F) * pauli_dot(phi)


def label(p, phi=(0, 0, 0)):
    """g = exp(1/2 p.sigma) . exp(-1/2 i phi.sigma)."""
    return boost([mpf(x) for x in p]) * rot([mpf(x) for x in phi])


def expm_sigma(a):
    """exp(a.sigma) for a possibly-complex 3-vector a = (ax,ay,az)."""
    alpha = sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
    if alpha == 0:
        return matrix(ID2.tolist())
    return cosh(alpha) * ID2 + (sinh(alpha) / alpha) * pauli_dot(a)


def displace(g_base, dp=(0, 0, 0), dphi=(0, 0, 0)):
    """
    Right-multiplication coherent-state displacement:
        g_base . exp( 1/2 (dp - i dphi) . sigma )
    dp     = momentum displacement (boost generator 1/2 sigma)
    dphi   = angle displacement    (rotation generator -1/2 i sigma)
    This is the convention that reproduces the SPEC's exact 1D families
    (e.g. transverse cosh z_22 = cosh p_bar cosh beta).
    """
    a = [(mpf(dp[i]) - IU * mpf(dphi[i])) / 2 for i in range(3)]
    return g_base * expm_sigma(a)


def dag(A):
    return matrix([[conj(A[0, 0]), conj(A[1, 0])],
                   [conj(A[0, 1]), conj(A[1, 1])]])


def xarg(g1, g2):
    """X = 1/2 tr(g1^dag g2)  (Chebyshev argument, generally complex)."""
    M = dag(g1) * g2
    return (M[0, 0] + M[1, 1]) / 2


def flux_of(g):
    """
    Classical flux vector p of the polar decomposition g = exp(1/2 p.sigma) . u.
    (Boost/Hermitian part only; used to test the connector map.)
    """
    M = g * dag(g)                     # = exp(p.sigma) = H^2, Hermitian positive
    a = re((M[0, 0] + M[1, 1]) / 2)
    bz = re((M[0, 0] - M[1, 1]) / 2)
    bx = re((M[0, 1] + M[1, 0]) / 2)
    by = re(IU * (M[0, 1] - M[1, 0]) / 2)
    bmag = sqrt(bx ** 2 + by ** 2 + bz ** 2)
    if bmag == 0:
        return [mpf(0), mpf(0), mpf(0)]
    sp = sqrt(a + bmag)
    sm = sqrt(a - bmag)
    gamma = (sp - sm) / (2 * bmag)     # H = alpha I + gamma b.sigma
    P = 2 * asinh(gamma * bmag)        # |vec(H)| = sinh(P/2)
    return [P * bx / bmag, P * by / bmag, P * bz / bmag]


# ---------------------------------------------------------------- Omega engine
def Omega_x(X, t=T_BENCH, N=None):
    """Omega as a function of X = cosh z, via Chebyshev-U. Works for complex X."""
    if N is None:
        N = nmax(t)
    total = mpf(0) * X            # keep type (mpf or mpc)
    U_pp = None                   # U_{n-3}
    U_p = None                    # U_{n-2}
    for n in range(1, N + 1):
        if n == 1:
            U = mpf(1)            # U_0
        elif n == 2:
            U = 2 * X            # U_1
        else:
            U = 2 * X * U_p - U_pp
        w = n * exp(-t * mpf(n * n - 1) / 4)
        total += w * U
        U_pp = U_p
        U_p = U
    return total


def overlap(g1, g2, t=T_BENCH):
    return Omega_x(xarg(g1, g2), t)


def lognorm(g, t=T_BENCH):
    return log(overlap(g, g, t))


def logfid(g1, g2, t=T_BENCH):
    """ln F = 2 Re ln N12 - ln N11 - ln N22  (real)."""
    n12 = log(overlap(g1, g2, t))
    return 2 * re(n12) - re(lognorm(g1, t)) - re(lognorm(g2, t))


# ------------------------------------------ Omega and z-derivatives (real z, sinh form)
def omega_derivs(z, t=T_BENCH, N=None):
    """
    Return (Omega, Omega', Omega'') as functions of the rapidity z (real).
    Per-term g = sinh(n z)/sinh z obeys  g'' = (n^2-1) g - 2 coth(z) g',
    so all three sums are analytic (no finite differencing) -> full precision.
    """
    if N is None:
        N = nmax(t)
    z = mpf(z)
    s = sinh(z)
    c = cosh(z)
    cothz = c / s
    Om = mpf(0)
    Om1 = mpf(0)
    Om2 = mpf(0)
    for n in range(1, N + 1):
        w = n * exp(-t * mpf(n * n - 1) / 4)
        S = sinh(n * z)
        C = cosh(n * z)
        g = S / s
        g1 = (n * C * s - S * c) / s ** 2
        g2 = (mpf(n * n - 1)) * g - 2 * cothz * g1
        Om += w * g
        Om1 += w * g1
        Om2 += w * g2
    return Om, Om1, Om2


def L_derivs(z, t=T_BENCH, N=None):
    """Return (L, L', L'') from the brute-force spin sum (analytic derivatives)."""
    Om, Om1, Om2 = omega_derivs(z, t, N)
    L = log(Om)
    L1 = Om1 / Om
    L2 = Om2 / Om - L1 ** 2
    return L, L1, L2


# ---------------------------------------------------------------- QGT via kernel
def _pert(vec, i, d):
    w = list(vec)
    w[i] = w[i] + d
    return w


def qgt(g_of, dim, t=T_BENCH, h=None, workdps=60):
    """
    Quantum geometric tensor at the base point (all deltas = 0) by mixed
    central differences of  f(x,y) = ln N(g(x), g(y)),  x = bra, y = ket:
        Q_ab = [f(+a,+b) - f(+a,-b) - f(-a,+b) + f(-a,-b)] / (4 h^2)
    Returns the complex mpmath matrix Q (dim x dim). G = 4 Re Q, B = 2 Im Q.

    workdps raises the working precision so the finite-difference floor sits far
    below any tolerance we test against (h~1e-15 @ dps 60 -> error ~1e-25).
    """
    with mp.workdps(workdps):
        if h is None:
            h = mpf(10) ** (-15)
        z = [mpf(0)] * dim

        def f(xv, yv):
            return log(overlap(g_of(xv), g_of(yv), t))

        Q = matrix(dim, dim)
        for a in range(dim):
            for b in range(dim):
                fpp = f(_pert(z, a, h), _pert(z, b, h))
                fpm = f(_pert(z, a, h), _pert(z, b, -h))
                fmp = f(_pert(z, a, -h), _pert(z, b, h))
                fmm = f(_pert(z, a, -h), _pert(z, b, -h))
                Q[a, b] = (fpp - fpm - fmp + fmm) / (4 * h * h)
    return Q


def berry_plaquette(g_of, dim, a, b, delta=mpf('1e-8'), t=T_BENCH, workdps=50):
    """
    Independent (Wilson-loop / Fukui-Hatsugai-Suzuki) Berry curvature in the
    (a,b) label-plane:  F_ab = -Im ln W / delta^2,
    W = <0|A><A|AB><AB|B><B|0>  around the small parallelogram.
    Sign convention: returns F with F = -B (= -2 Im Q); callers compare |.|.
    """
    with mp.workdps(workdps):
        z = [mpf(0)] * dim
        p00 = z
        p10 = _pert(z, a, delta)
        p01 = _pert(z, b, delta)
        p11 = _pert(_pert(z, a, delta), b, delta)
        g00, g10, g01, g11 = (g_of(p00), g_of(p10), g_of(p01), g_of(p11))
        W = (overlap(g00, g10, t) * overlap(g10, g11, t) *
             overlap(g11, g01, t) * overlap(g01, g00, t))
        return -im(log(W)) / (delta * delta)


def det(A):
    """Determinant of a small mpmath matrix via Gaussian elimination (mpf/mpc)."""
    n = A.rows
    M = [[A[i, j] for j in range(n)] for i in range(n)]
    d = mpf(1)
    for col in range(n):
        piv = next((r for r in range(col, n) if M[r][col] != 0), None)
        if piv is None:
            return mpf(0)
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
            d = -d
        d = d * M[col][col]
        inv = 1 / M[col][col]
        for r in range(col + 1, n):
            f = M[r][col] * inv
            if f != 0:
                for c in range(col, n):
                    M[r][c] = M[r][c] - f * M[col][c]
    return d


def pfaffian(A):
    """Pfaffian of an even-dim antisymmetric mpmath matrix (recursive expansion)."""
    n = A.rows
    if n == 0:
        return mpf(1)
    if n % 2 == 1:
        return mpf(0)
    if n == 2:
        return A[0, 1]
    total = mpf(0)
    for j in range(1, n):
        if A[0, j] == 0:
            continue
        idx = [k for k in range(n) if k not in (0, j)]
        minor = matrix([[A[r, c] for c in idx] for r in idx])
        total += (-1) ** (j - 1) * A[0, j] * pfaffian(minor)
    return total


def re_mat(M):
    n, m = M.rows, M.cols
    return matrix([[re(M[i, j]) for j in range(m)] for i in range(n)])


def im_mat(M):
    n, m = M.rows, M.cols
    return matrix([[im(M[i, j]) for j in range(m)] for i in range(n)])


def G_from_Q(Q):
    return 4 * re_mat(Q)


def B_from_Q(Q):
    return 2 * im_mat(Q)
