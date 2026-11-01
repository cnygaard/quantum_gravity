"""
Does LQG geometry dynamics scramble?  Quantum chaos of the volume operator.
===========================================================================

The Page-curve bookkeeping (theory/page_curve.py) ASSUMES the global state is Haar-typical.
This script asks a narrower, kinematical question: what are the level statistics of the LQG
VOLUME operator on a single node?  Classically, Haggard (2013, arXiv:1211.7311) found evidence of
chaos (with a mixed phase space) for the pentahedral volume flow and conjectured it at higher
valence; Brunnemann-Rideout (2008) computed the spectrum and gap.  The volume is an OBSERVABLE,
not a generator, so its statistics are circumstantial evidence for typicality, not a test of it.

Design (theorist): V = sum_v V_v is a sum of node-local COMMUTING terms, so a multi-node
graph is non-interacting and cannot scramble.  The genuine multi-DOF object is a SINGLE
high-valence node.  On its SU(2)-invariant intertwiner space Inv(tensor_p V_{j_p}) we build
the Ashtekar-Lewandowski volume argument
    Q_hat = sum_{p<q<r} sigma(p,q,r) J_p . (J_q x J_r),     V_hat = sqrt(|Q_hat|),
using the exact operator identity  J_p.(J_q x J_r) = -i [ G_pq , G_pr ],  G_ab = J_a . J_b.

SCALABLE CONSTRUCTION.  G_ab = J_a.J_b preserves total J_z, so we build every operator in
the M=0 SECTOR (dim N0 << D_full = prod(2j+1)), find the singlet (J=0) subspace inside it,
and project.  This reaches intertwiner dimensions D_int ~ hundreds cheaply, where the naive
full-tensor-product projection could not (D_int ~ 11 at D_full ~ 2500).

SYMMETRY CLASS.  In the real M=0/recoupling basis the G_ab are real-symmetric, so
Q_hat = i*A with A real-antisymmetric: its spectrum is +/- symmetric (the "signed-volume"
chirality) and its class is NOT generic GUE.  We therefore analyse the POSITIVE spectrum and
calibrate the chaotic target against a random REAL-ANTISYMMETRIC matrix (the correct class),
alongside GUE (sanity) and Poisson (integrable).

DIAGNOSTIC: level-spacing ratio r~ = <min(r,1/r)> (Atas et al. 2013, unfolding-free) of the
positive spectrum of V = sqrt|Q| (operator and reference alike), full and bulk window, plus the
small-s exponent of unfolded spacings, over ensembles of realizable embeddings, against a
matched-D antisymmetric reference.  Eigenvector statistics are taken in the Clebsch-Gordan
RECOUPLING basis (recoupling_basis); an arbitrary basis of the singlet space would make the
Porter-Thomas test nearly trivial.  numpy float64, exact diagonalization.
"""
import os
import sys
import time
from functools import lru_cache

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recoupling import cg as _cg_exact                    # noqa: E402

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

RMT = {'Poisson': 0.386294, 'GOE': 0.530700, 'GUE': 0.599455, 'GSE': 0.674007}
# Atas et al. (PRL 110, 084101): large-N numerical values (RMT above) vs the 3x3 Wigner-like
# surmise values.  Quote them separately -- they are different numbers.
ATAS_SURMISE = {'Poisson': 2 * np.log(2) - 1, 'GOE': 4 - 2 * np.sqrt(3),
                'GUE': 2 * np.sqrt(3) / np.pi - 0.5}
MAXN0 = 60000                      # cap on the M=0 subspace dimension (sparse build)


# --------------------------------------------------- M=0 sector construction (scalable)
def m0_states(spins):
    """All (m_1,...,m_n), m_p in {j_p..-j_p}, sum m_p = 0; pruned DFS (never builds D_full)."""
    js = [float(j) for j in spins]
    n = len(js)
    mrng = [[js[p] - k for k in range(int(round(2 * js[p] + 1)))] for p in range(n)]
    sufmin = [0.0] * (n + 1)
    sufmax = [0.0] * (n + 1)
    for p in range(n - 1, -1, -1):
        sufmin[p] = sufmin[p + 1] - js[p]
        sufmax[p] = sufmax[p + 1] + js[p]
    out = []

    def dfs(p, cur, acc):
        if p == n:
            if abs(acc) < 1e-9:
                out.append(tuple(cur))
            return
        for m in mrng[p]:
            na = acc + m
            if na + sufmin[p + 1] <= 1e-9 and na + sufmax[p + 1] >= -1e-9:
                cur.append(m)
                dfs(p + 1, cur, na)
                cur.pop()

    dfs(0, [], 0.0)
    return out, {s: i for i, s in enumerate(out)}


def _cp(j, m):
    return np.sqrt(max(0.0, (j - m) * (j + m + 1)))     # J+|j,m> coefficient


def _cm(j, m):
    return np.sqrt(max(0.0, (j + m) * (j - m + 1)))     # J-|j,m> coefficient


def G_matrix(a, b, states, idx, spins):
    """G_ab = J_a . J_b = Jz_a Jz_b + 1/2 (J+_a J-_b + J-_a J+_b), real-symmetric on M=0
    (sparse CSR: the N0 ~ 2e4 builds would need GBs dense)."""
    N = len(states)
    rows, cols, vals = [], [], []
    ja, jb = float(spins[a]), float(spins[b])
    for i, s in enumerate(states):
        ma, mb = s[a], s[b]
        rows.append(i); cols.append(i); vals.append(ma * mb)
        s2 = list(s); s2[a] = ma + 1; s2[b] = mb - 1                 # J+_a J-_b
        k = idx.get(tuple(s2))
        if k is not None:
            rows.append(k); cols.append(i); vals.append(0.5 * _cp(ja, ma) * _cm(jb, mb))
        s2 = list(s); s2[a] = ma - 1; s2[b] = mb + 1                 # J-_a J+_b
        k = idx.get(tuple(s2))
        if k is not None:
            rows.append(k); cols.append(i); vals.append(0.5 * _cm(ja, ma) * _cp(jb, mb))
    G = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    return (G + G.T) / 2


@lru_cache(maxsize=None)
def _cg(j1, m1, j2, m2, J, M):
    return _cg_exact(j1, m1, j2, m2, J, M)


def _mrange(j):
    return [j - k for k in range(int(round(2 * j)) + 1)]           # j, j-1, ..., -j


def _reachable(rest):
    """Total spins attainable by coupling the spins in `rest` (empty -> {0})."""
    if not rest:
        return [0.0]
    S, jm = sum(rest), max(rest)
    kmin = max(2 * jm - S, 0.5 if round(2 * S) % 2 else 0.0)
    return [kmin + k for k in range(int(round(S - kmin)) + 1)]


def recoupling_basis(spins):
    """Orthonormal SEQUENTIAL-COUPLING basis of Inv(tensor_p V_{j_p}): paths
    (((j1 j2) k2 j3) k3 ...) k_n = 0, each vector expanded on the M=0 product states by
    Clebsch-Gordan recursion.  This is the physical recoupling basis (an eigh basis of the
    degenerate J^2 = 0 eigenspace would be an arbitrary rotation of it, which trivializes
    basis-dependent tests such as Porter-Thomas).  Returns (P [N0 x D], states, idx, paths)."""
    js = [float(j) for j in spins]
    n = len(js)
    sect = {m: [(m,)] for m in _mrange(js[0])}                       # mu -> product states
    mult = [((js[0],), js[0], {m: np.array([1.0]) for m in _mrange(js[0])})]
    for i in range(1, n):
        j, rest = js[i], js[i + 1:]
        reach, mumax = _reachable(rest), sum(rest)
        mus = sorted({mu + m for mu in sect for m in _mrange(j) if abs(mu + m) <= mumax + 1e-9})
        newsect, offs = {}, {}
        for mup in mus:                       # sector mu' = concat over m of sect[mu'-m] x {m}
            lst, o = [], {}
            for m in _mrange(j):
                if mup - m in sect:
                    o[m] = (len(lst), len(sect[mup - m]))
                    lst.extend(t + (m,) for t in sect[mup - m])
            newsect[mup], offs[mup] = lst, o
        newmult = []
        for path, k, vecs in mult:
            for K in np.arange(abs(k - j), k + j + 0.5, 1.0):
                K = float(K)
                if not any(abs(K - r) < 1e-9 for r in reach):
                    continue
                nv = {}
                for mup in [mu for mu in _mrange(K) if mu in newsect]:
                    v = np.zeros(len(newsect[mup]))
                    for m, (o0, ln) in offs[mup].items():
                        mu = mup - m
                        if abs(mu) <= k + 1e-9:
                            c = _cg(k, mu, j, m, K, mup)
                            if c:
                                v[o0:o0 + ln] += c * vecs[mu]
                    nv[mup] = v
                newmult.append((path + (K,), K, nv))
        sect, mult = newsect, newmult
    states = sect.get(0.0, [])
    P = (np.column_stack([vecs[0.0] for _, _, vecs in mult]) if mult
         else np.zeros((len(states), 0)))
    return P, states, {s: i for i, s in enumerate(states)}, [p for p, _, _ in mult]


_GCACHE = {}
_J0RES = {}


def intertwiner_G(spins):
    """Return {(a,b): G_ab in the recoupling basis of the singlet space}, D_int, N0.  The basis
    is built by Clebsch-Gordan recursion (recoupling_basis) and every G_ab is sparse, so the
    N0 ~ 2-3e4 seven-valent nodes need no dense N0 x N0 matrix.  Records the J^2 residual and
    the orthonormality defect of the basis (certificates).  Memoized on the spin tuple."""
    key = tuple(float(j) for j in spins)
    if key in _GCACHE:
        return _GCACHE[key]
    P, states, idx, _ = recoupling_basis(spins)
    N0, D_int = len(states), P.shape[1]
    if N0 == 0 or N0 > MAXN0:
        _GCACHE[key] = (None, None, N0)
        return _GCACHE[key]
    n = len(spins)
    J2 = sp.identity(N0, format='csr') * sum(float(j) * (float(j) + 1) for j in spins)
    Gint = {}
    for a in range(n):
        for b in range(a + 1, n):
            G = G_matrix(a, b, states, idx, spins)
            J2 = J2 + 2 * G
            Gint[(a, b)] = P.T @ (G @ P)
    _J0RES[key] = (float(np.abs(J2 @ P).max()) if D_int else None,
                   float(np.abs(P.T @ P - np.eye(D_int)).max()) if D_int else None)
    _GCACHE[key] = (Gint, D_int, N0)
    return _GCACHE[key]


def Q_from_Gint(Gint, n, sign):
    """Q = sum_{p<q<r} sigma(pqr) (-i)[G_pq, G_pr]  (cheap; the sign pattern varies here)."""
    D = next(iter(Gint.values())).shape[0]
    Q = np.zeros((D, D), complex)
    for p in range(n):
        for q in range(p + 1, n):
            for r in range(q + 1, n):
                Gpq, Gpr = Gint[(p, q)], Gint[(p, r)]
                Q += sign(p, q, r) * (-1j) * (Gpq @ Gpr - Gpr @ Gpq)
    return (Q + Q.conj().T) / 2


def _default_sign(p, q, r):
    return (-1) ** (p + q + r)


def random_sign(seed):
    """A fixed random +/-1 orientation pattern (one generic 'embedding'); RMT class is
    embedding-independent, so averaging over these gives error bars on r~."""
    rng = np.random.default_rng(seed)
    tab = {}

    def s(p, q, r):
        if (p, q, r) not in tab:
            tab[(p, q, r)] = 1 if rng.random() < 0.5 else -1
        return tab[(p, q, r)]
    return s


def embedding_sign(seed, n, with_mindet=False):
    """A REALIZABLE orientation pattern: sigma(p,q,r) = sign det(u_p,u_q,u_r) for n random
    tangent directions u_p in R^3 (one generic embedding of the node).  Arbitrary +/-1 tables
    (random_sign) need not be realizable by any embedding.  The default (-1)^(p+q+r) is
    realizable (alternating moment curve u_p = (-1)^p (1, p, p^2)).  with_mindet: also return
    min |det(u_p,u_q,u_r)| over triples of UNIT tangents (closeness to a degenerate embedding)."""
    u = np.random.default_rng(seed).normal(size=(n, 3))
    tab = {(p, q, r): int(np.sign(np.linalg.det(u[[p, q, r]])))
           for p in range(n) for q in range(p + 1, n) for r in range(q + 1, n)}
    sign = lambda p, q, r: tab[(p, q, r)]                          # noqa: E731
    if not with_mindet:
        return sign
    uh = u / np.linalg.norm(u, axis=1, keepdims=True)
    mindet = min(abs(np.linalg.det(uh[[p, q, r]])) for (p, q, r) in tab)
    return sign, float(mindet)


def volume_spectrum(spins, sign=None):
    """Positive eigenvalues of V=sqrt(|Q|) on the intertwiner space, plus (D_int, N0)."""
    Gint, D_int, N0 = intertwiner_G(spins)
    if Gint is None:
        return None, D_int, N0
    if D_int == 0:
        return np.array([]), 0, N0
    Q = Q_from_Gint(Gint, len(spins), sign or _default_sign)
    qe = np.linalg.eigvalsh(Q)
    return np.sqrt(qe[qe > 1e-9]), D_int, N0                          # one per +/- pair


def rbar_ensemble(spins, nsign=8, seed=0):
    """Mean +/- std of r~ over nsign random orientation patterns (intertwiner_G built once)."""
    Gint, D_int, N0 = intertwiner_G(spins)
    if Gint is None or D_int < 8:
        return None, None, D_int, N0
    n = len(spins)
    vals = []
    for k in range(nsign):
        qe = np.linalg.eigvalsh(Q_from_Gint(Gint, n, random_sign(seed + k)))
        vals.append(rbar(np.sqrt(qe[qe > 1e-9])))
    vals = np.array([v for v in vals if v == v])
    return float(vals.mean()), float(vals.std()), D_int, N0


def eth_test(spins, sign=None, pair=(0, 1)):
    """ETH on the dihedral observable O=G_ab=J_a.J_b (varies over intertwiners, unlike the
    fixed single-edge area).  Returns (levels, diag matrix elements <alpha|O|alpha>, eta),
    where eta = std of the fluctuation of <O> about a smooth trend, /range -- small & shrinking
    with D_int signals ETH."""
    Gint, D_int, N0 = intertwiner_G(spins)
    if Gint is None or D_int < 8:
        return None, None, None
    n = len(spins)
    Q = Q_from_Gint(Gint, n, sign or _default_sign)
    w, U = np.linalg.eigh(Q)                                          # Q eigenbasis
    O = Gint[pair]
    diag = np.real(np.einsum('ia,ij,ja->a', U.conj(), O, U))          # <alpha|O|alpha>
    order = np.argsort(w)
    lv, dg = w[order], diag[order]
    trend = np.polyval(np.polyfit(lv, dg, 4), lv)                     # smooth f(E)
    eta = float(np.std(dg - trend) / (dg.max() - dg.min() + 1e-12))
    return lv, dg, eta


# --------------------------------------------------- level-spacing ratio + calibrations
def rbar(levels):
    """Atas et al. mean ratio r~ = <min(r,1/r)>, unfolding-free."""
    e = np.sort(np.asarray(levels, float))
    s = np.diff(e)
    s = s[s > 1e-9 * max(1.0, np.mean(np.abs(e)))]
    if len(s) < 3:
        return np.nan
    r = s[1:] / s[:-1]
    return float(np.mean(np.minimum(r, 1.0 / r)))


def rbar_sem(levels):
    """(mean, SEM) of r~ over individual gap ratios -- referee: always quote error bars."""
    e = np.sort(np.asarray(levels, float))
    s = np.diff(e)
    s = s[s > 1e-9 * max(1.0, np.mean(np.abs(e)))]
    if len(s) < 4:
        return np.nan, np.nan
    r = np.minimum(s[1:] / s[:-1], s[:-1] / s[1:])
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def gap_ratios(levels):
    """Individual ratios r_i = min(s_i/s_{i-1}, s_{i-1}/s_i) of consecutive spacings."""
    e = np.sort(np.asarray(levels, float))
    s = np.diff(e)
    s = s[s > 1e-9 * max(1.0, np.mean(np.abs(e)))]
    return np.minimum(s[1:] / s[:-1], s[:-1] / s[1:])


def rbar_block_bootstrap(levels, block=None, nboot=4000, seed=0):
    """Correlation-robust error on r~ (referee: adjacent ratios share a spacing, so the iid SEM
    understates the error).  Moving-block bootstrap over the ratio sequence, block length
    ~ n^(1/3) unless given.  Returns dict(mean, sem_iid, se_block, block, lag1)."""
    r = gap_ratios(levels)
    n = len(r)
    b = block or max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    nblk = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(nboot, nblk))
    means = np.array([np.concatenate([r[s:s + b] for s in row])[:n].mean() for row in starts])
    lag1 = float(np.corrcoef(r[1:], r[:-1])[0, 1]) if n > 3 else float('nan')
    return {'mean': float(r.mean()), 'sem_iid': float(r.std(ddof=1) / np.sqrt(n)),
            'se_block': float(means.std(ddof=1)), 'block': b, 'lag1': lag1, 'n': n}


def bulk_window(v, lo=0.2, hi=0.9):
    """Positive levels with rank in [lo, hi) of the sorted spectrum: removes the small-volume
    edge (Brunnemann-Rideout accumulation) and the top edge."""
    v = np.sort(np.asarray(v, float))
    return v[int(np.floor(lo * v.size)):int(np.ceil(hi * v.size))]


def unfolded_spacings(v, deg=7, lo=0.2, hi=0.9):
    """Nearest-neighbour spacings of the levels unfolded by a degree-`deg` polynomial fit to the
    staircase N(E), restricted to the rank window [lo, hi) and normalized to unit mean."""
    e = np.sort(np.asarray(v, float))
    x = (e - e.mean()) / (e.std() + 1e-300)
    u = np.polyval(np.polyfit(x, np.arange(e.size, dtype=float), deg), x)
    s = np.diff(u)
    s = s[int(np.floor(lo * s.size)):int(np.ceil(hi * s.size))]
    return s / s.mean()


def beta_mle(s, smax=0.5):
    """Small-s level-repulsion exponent: MLE of P(s) ~ s^beta on (0, smax) (pooled unfolded
    spacings).  Returns (beta, SE, count).  The e^{-c s^2} tail biases it slightly low, so it
    is always compared with the same estimator on the matched reference."""
    x = np.asarray(s, float)
    x = x[(x > 0) & (x < smax)]
    b1 = -x.size / np.sum(np.log(x / smax))
    return float(b1 - 1), float(b1 / np.sqrt(x.size)), int(x.size)


def antisym_reference(D, nrep=200, seed=11):
    """Finite-size class reference at MATCHED intertwiner dimension D: r~ of the positive
    spectrum of i*A, A real-antisymmetric DxD (r~ on V = sqrt|q|, as for the volume), mean and
    spread over nrep draws, the per-draw values (full spectrum and bulk window), the near-zero
    fraction (see near_zero_fraction) and the small-s exponent of the pooled unfolded bulk
    spacings (beta_mle)."""
    rng = np.random.default_rng(seed)
    rs, rb, nz, sp_pool = [], [], [], []
    for _ in range(nrep):
        A = rng.normal(size=(D, D))
        q = np.linalg.eigvalsh(1j * (A - A.T) / 2)
        v = np.sqrt(q[q > 1e-9])
        rs.append(rbar(v))
        rb.append(rbar(bulk_window(v)))
        nz.append(near_zero_fraction(v))
        sp_pool.append(unfolded_spacings(v))
    rs, rb = np.array(rs), np.array(rb)
    beta, beta_se, nb = beta_mle(np.concatenate(sp_pool))
    return {'mean': float(rs.mean()), 'std': float(rs.std(ddof=1)), 'nrep': nrep,
            'sem': float(rs.std(ddof=1) / np.sqrt(nrep)), 'near_zero': float(np.mean(nz)),
            'mean_bulk': float(rb.mean()), 'std_bulk': float(rb.std(ddof=1)),
            'values': rs.tolist(), 'values_bulk': rb.tolist(),
            'beta': beta, 'beta_se': beta_se, 'beta_n': nb}


def goe_beta_reference(D, nrep=200, seed=11):
    """Calibration of beta_mle on the GOE (beta=1): the same pipeline as antisym_reference
    (positive levels of a real-symmetric DxD matrix, square root, unfolded bulk spacings).
    Returns (beta, SE, mean r~)."""
    rng = np.random.default_rng(seed)
    pool, rs = [], []
    for _ in range(nrep):
        A = rng.normal(size=(D, D))
        q = np.linalg.eigvalsh((A + A.T) / 2)
        v = np.sqrt(q[q > 1e-9])
        pool.append(unfolded_spacings(v))
        rs.append(rbar(v))
    b, se, _ = beta_mle(np.concatenate(pool))
    return b, se, float(np.mean(rs))


def near_zero_fraction(v, frac=0.1):
    """Fraction of positive volume levels below frac * max (the Brunnemann-Rideout question of
    accumulation at zero volume), to be compared with the matched-D class reference."""
    v = np.asarray(v, float)
    return float(np.mean(v < frac * v.max())) if v.size else float('nan')


def anchor_analysis(spins, nsign=64, seed=5000):
    """Referee-grade statistics on one node: the default embedding (sign (-1)^(p+q+r)) with
    block-bootstrap error, and the ensemble over nsign random REALIZABLE embeddings
    (embedding_sign).  Embeddings are classified as DEGENERATE (zero modes of Q beyond the
    D-parity one, or exactly degenerate positive levels: an unresolved symmetry / zero-volume
    structure) or GENERIC; r~ is reported for both and for all.  Porter-Thomas KS, spacing
    degeneracy and near-zero fraction per embedding, against the matched-D antisymmetric
    reference.  One intertwiner build."""
    Gint, D_int, N0 = intertwiner_G(spins)
    n = len(spins)
    out = {'spins': [float(j) for j in spins], 'D_int': D_int, 'N0': N0,
           'distinct': len(set(spins)) == len(spins)}

    def stats(sign):
        w, U = np.linalg.eigh(Q_from_Gint(Gint, n, sign))
        pos = w > 1e-9
        v = np.sqrt(w[pos])
        s = np.diff(np.sort(v))
        y = np.sort((np.abs(U[:, pos]) ** 2).ravel() * D_int)
        ks = float(np.max(np.abs(np.arange(1, len(y) + 1) / len(y) - (1.0 - np.exp(-y)))))
        zero_modes = int(np.sum(np.abs(w) < 1e-9))
        degenerate = zero_modes > D_int % 2 or bool(np.any(s < 1e-6 * s.mean()))
        return v, ks, float(s.min() / s.mean()), near_zero_fraction(v), zero_modes, degenerate

    v0, ks0, deg0, nz0, zm0, dg0 = stats(_default_sign)
    out['default'] = dict(rbar_block_bootstrap(v0), ks=ks0, min_over_mean_spacing=deg0,
                          near_zero=nz0, levels=int(v0.size), zero_modes=zm0, degenerate=dg0,
                          mean_bulk=rbar(bulk_window(v0)))
    rows, pool = [], []
    for k in range(nsign):
        sign, mindet = embedding_sign(seed + k, n, with_mindet=True)
        v, ks, deg, nz, zm, dg = stats(sign)
        rows.append((rbar(v), ks, deg, nz, zm, dg, rbar(bulk_window(v)), mindet))
        if not dg:
            pool.append(unfolded_spacings(v))

    def summ(sel, col=0):
        r = np.array([x[col] for x in sel])
        m = len(r)
        return {'count': m, 'mean': float(r.mean()) if m else None,
                'std': float(r.std(ddof=1)) if m > 1 else None,
                'sem': float(r.std(ddof=1) / np.sqrt(m)) if m > 1 else None,
                'ks_mean': float(np.mean([x[1] for x in sel])) if m else None,
                'ks_max': float(max(x[1] for x in sel)) if m else None,
                'near_zero_mean': float(np.mean([x[3] for x in sel])) if m else None}
    generic = [x for x in rows if not x[5]]
    ref = antisym_reference(D_int)
    out['ensemble'] = {'nsign': nsign, 'all': summ(rows), 'generic': summ(generic),
                       'degenerate': summ([x for x in rows if x[5]]),
                       'generic_bulk': summ(generic, col=6),
                       'zero_mode_counts': sorted({x[4] for x in rows}),
                       'min_over_mean_spacing_generic': float(min(x[2] for x in generic)) if generic else None,
                       'values': [x[0] for x in rows], 'values_bulk': [x[6] for x in rows],
                       'mindet': [x[7] for x in rows], 'degenerate_flags': [x[5] for x in rows]}
    if len(generic) > 2:
        from scipy.stats import ks_2samp, levene, spearmanr
        g, gb = [x[0] for x in generic], [x[6] for x in generic]
        out['tests'] = {
            # per-embedding r~ vs per-draw reference r~: location+shape (KS) and spread (Levene)
            'ks_ref': list(map(float, ks_2samp(g, ref['values']))),
            'ks_ref_bulk': list(map(float, ks_2samp(gb, ref['values_bulk']))),
            'levene': list(map(float, levene(g, ref['values']))),
            'levene_bulk': list(map(float, levene(gb, ref['values_bulk']))),
            'spread_ratio': float(np.std(g, ddof=1) / ref['std']),
            # does r~ track how close an embedding is to degenerate?
            'det_corr': list(map(float, spearmanr([x[7] for x in generic], g))),
            'det_corr_all': list(map(float, spearmanr([x[7] for x in rows], [x[0] for x in rows])))}
        beta, beta_se, nb = beta_mle(np.concatenate(pool))
        out['tests']['beta'] = {'beta': beta, 'se': beta_se, 'n': nb,
                                'ref': ref['beta'], 'ref_se': ref['beta_se']}
    out['reference'] = ref
    out['atas_large_N'] = {k: RMT[k] for k in ('Poisson', 'GOE', 'GUE')}
    out['atas_surmise'] = {k: float(v) for k, v in ATAS_SURMISE.items()}
    return out


def pt_ks(spins, sign=None):
    """Porter-Thomas test at the DISTRIBUTION level (referee: not just the IPR 2nd moment).
    KS distance of y = D|c|^2 (eigenvector components of Q in the orthonormal singlet basis,
    positive-eigenvalue vectors only, avoiding the +/- conjugate double-count) from the
    beta=2 Porter-Thomas law P(y)=e^{-y}.  Returns (ks, nsamples, D_int)."""
    Gint, D_int, _ = intertwiner_G(spins)
    if Gint is None or D_int < 8:
        return None
    Q = Q_from_Gint(Gint, len(spins), sign or _default_sign)
    w, U = np.linalg.eigh(Q)
    cols = U[:, w > 1e-9]
    y = np.sort((np.abs(cols) ** 2).ravel() * D_int)
    cdf = np.arange(1, len(y) + 1) / len(y)
    ks = float(np.max(np.abs(cdf - (1.0 - np.exp(-y)))))
    return ks, len(y), D_int


def j0_residual(spins):
    """Certificate that the working basis is the total-J=0 block (not merely M=0):
    max |J^2_total P| over the recoupling-basis vectors (and P^T P = 1 to the returned
    orthonormality defect, see basis_defect)."""
    intertwiner_G(spins)
    return _J0RES.get(tuple(float(j) for j in spins), (None, None))[0]


def basis_defect(spins):
    """max |P^T P - 1| of the Clebsch-Gordan recoupling basis (orthonormality certificate)."""
    intertwiner_G(spins)
    return _J0RES.get(tuple(float(j) for j in spins), (None, None))[1]


def classify(rb, chaotic_ref):
    refs = {'Poisson': RMT['Poisson'], 'chaotic': chaotic_ref}
    return min(refs, key=lambda k: abs(refs[k] - rb))


def _gue_rbar(N=1500, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    return rbar(np.linalg.eigvalsh((A + A.conj().T) / 2))


def _antisym_rbar(N=1500, seed=3):
    """Chaotic reference in the volume's class: positive spectrum of i*A, A real-antisymmetric."""
    A = np.random.default_rng(seed).normal(size=(N, N))
    A = (A - A.T) / 2
    ev = np.linalg.eigvalsh(1j * A)
    return rbar(ev[ev > 1e-9])


DISTINCT = [[1, 1.5, 2, 2.5, 3, 4], [1, 2, 2.5, 3, 3.5, 4], [1.5, 2, 2.5, 3, 3.5, 4.5]]
MONO = [[2, 2, 2, 2, 2, 2], [2.5, 2.5, 2.5, 2.5, 2.5, 2.5]]
NSIGN = 16


def eigvec_ipr(spins, sign=None):
    """Tier 2 (in D_int): eigenvector-level typicality.  Mean IPR*D_int of V-hat's eigenstates
    in the intertwiner basis: ~2 => Porter-Thomas (Haar-random components, ETH/2-design-adjacent);
    >>2 => localized (integrable).  This is the eigenvector randomness the Page curve needs, on
    the volume operator's OWN states -- NOT yet the physical evaporating state (the honest gap)."""
    Gint, D_int, _ = intertwiner_G(spins)
    if Gint is None or D_int < 8:
        return None, D_int
    Q = Q_from_Gint(Gint, len(spins), sign or _default_sign)
    _, U = np.linalg.eigh(Q)
    ipr = np.sum(np.abs(U) ** 4, axis=0)
    haar = 2.0                                    # complex Porter-Thomas: <IPR>*D -> 2
    return float(np.mean(ipr) * D_int), haar


def thermalize(spins, sign=None, pair=(0, 2), nt=500):
    """Tier 3 (in D_int): dynamical scrambling in 'geometry time'.  Evolve a basis state under
    e^{-i Q t} (Q = signed volume-squared) and track <O>(t), O=J_a.J_b.  Relaxation of <O>(t) to
    the diagonal-ensemble mean with small late-time fluctuation = the dynamics thermalizes the
    observable (scrambling), reusing the SAME operator whose spectrum is chaotic (Tier 1)."""
    Gint, D_int, _ = intertwiner_G(spins)
    if Gint is None or D_int < 8:
        return None
    lam, U = np.linalg.eigh(Q_from_Gint(Gint, len(spins), sign or _default_sign))
    Oe = U.conj().T @ Gint[pair] @ U
    c = U.conj().T @ np.eye(D_int)[0]                              # amplitudes of basis state |0>
    tmax = 30.0 / (np.std(lam) + 1e-12)
    ts = np.linspace(0, tmax, nt)
    Ot = np.array([np.real((np.exp(-1j * lam * t) * c).conj() @ Oe @ (np.exp(-1j * lam * t) * c))
                   for t in ts])
    Omicro = float(np.real(np.sum(np.abs(c) ** 2 * np.diag(Oe))))  # diagonal-ensemble mean
    late = Ot[nt // 2:]
    return float(Ot[0]), Omicro, float(np.std(late) / (abs(Omicro) + 1e-9)), D_int


def run():
    print("=" * 74)
    print("QUANTUM CHAOS OF THE LQG VOLUME OPERATOR (single high-valence node, M=0 build)")
    print("=" * 74)
    gue, anti = _gue_rbar(), _antisym_rbar()
    print(f"  calibration:  Poisson {RMT['Poisson']:.3f}  |  GUE {gue:.3f} (expect .600)"
          f"  |  antisym-random {anti:.3f}  <- chaotic target (the volume's class)", flush=True)
    print(f"  interpretation: r~ ~ {anti:.2f} = chaotic ; ~0.39 = integrable ; between = MIXED"
          f" (Haggard's pentahedron has mixed phase space)", flush=True)

    lv, Di, _ = volume_spectrum([1, 1, 1, 1])
    print(f"\n[control] 4-valent quantum tetrahedron: vol={lv[0]:.3f} (cross-check 2.632), D_int={Di}", flush=True)

    print(f"\n[TIER 1] distinct spins (no S_n symmetry), r~ averaged over {NSIGN} orientation patterns:", flush=True)
    for spins in DISTINCT:
        t = time.time()
        mean, std, Di, N0 = rbar_ensemble(spins, nsign=NSIGN)
        if mean is None:
            print(f"    {spins}  D_int={Di} too small / N0={N0} skipped", flush=True)
            continue
        tag = 'chaotic' if mean > 0.5 else ('MIXED' if mean > 0.45 else 'Poisson')
        print(f"    {spins}  D_int={Di}  r~ = {mean:.3f} +/- {std:.3f}  -> {tag}"
              f"   [{time.time()-t:.0f}s]", flush=True)

    print(f"    large nodes (valence 7-8, D_int up to 937): embedding ensembles in scrambling_big.py", flush=True)

    print(f"\n[TIER 2] eigenvector-level typicality (IPR*D_int of V eigenstates; ~2=Porter-Thomas):", flush=True)
    for spins in DISTINCT:
        val, haar = eigvec_ipr(spins)
        if val is not None:
            print(f"    {spins}  IPR*D_int = {val:.2f}  (Haar/Porter-Thomas = {haar:.0f};"
                  f" {'random-like' if val < 4 else 'localized'})", flush=True)

    print(f"\n[TIER 3] dynamical thermalization under e^(-iQt) (geometry flow), O=J_a.J_b:", flush=True)
    for spins in DISTINCT:
        res = thermalize(spins, pair=(0, 2))
        if res is not None:
            O0, Omic, fluc, Di = res
            print(f"    {spins}  <O>: t=0 {O0:+.2f} -> micro {Omic:+.2f}, late fluct/|mean|={fluc:.2f}"
                  f"  ({'thermalizes' if fluc < 0.3 else 'partial'})", flush=True)

    print(f"\n[ETH] dihedral observable O=J_a.J_b in the volume eigenbasis (eta = fluctuation/range):", flush=True)
    for spins in DISTINCT[:2]:
        lvl, dg, eta = eth_test(spins, pair=(0, 2))
        if eta is not None:
            print(f"    {spins}  eta = {eta:.3f}  (smaller => smoother f(E) => ETH-like)", flush=True)

    print(f"\n[sector demo] FIXED symmetric sign (-1)^(p+q+r): distinct vs monochromatic.", flush=True)
    print(f"    (random signs would themselves break S_n; a fixed sign exposes the pitfall)", flush=True)
    for spins in ([1, 1.5, 2, 2.5, 3, 4], [2, 2, 2, 2, 2, 2]):
        lv, Di, _ = volume_spectrum(spins)                           # default (-1)^(p+q+r)
        rb = rbar(lv)
        kind = 'distinct' if len(set(spins)) == len(spins) else 'monochromatic (S_n)'
        tag = 'chaotic' if rb > 0.5 else ('Poisson (sectors unresolved)' if rb < 0.45 else 'mixed')
        print(f"    {kind:22s} {spins}  D_int={Di} n={lv.size}  r~={rb:.3f} -> {tag}", flush=True)

    print("\nVERDICT (single node, kinematical): strong level repulsion, intermediate between Poisson and", flush=True)
    print("  the beta=2 class at valence 6 and rising with valence (scrambling_big.py: 6 -> 7 -> 8); the", flush=True)
    print("  default embedding is one operator, the embedding ensemble is the statistic.  Eigenvectors in", flush=True)
    print("  the recoupling basis are delocalized but measurably non-Porter-Thomas.  Circumstantial support", flush=True)
    print("  for typicality; the volume is an observable, not the evaporation dynamics.", flush=True)


if __name__ == "__main__":
    run()
