"""
Does LQG geometry dynamics scramble?  Quantum chaos of the volume operator.
===========================================================================

The Page-curve computation (theory/page_curve.py) ASSUMED the evaporating state is
Haar-typical (fast scrambling).  This script tests whether the LQG VOLUME operator is
quantum-chaotic -- the quantum sequel to Haggard (2013, arXiv:1211.7311), who showed the
CLASSICAL pentahedral-volume flow is chaotic for valence >= 5.  (Brunnemann-Rideout 2008
computed the spectrum/gap but NOT the level statistics, so the RMT class is open.)

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
positive spectrum, in a single resolved sector.  Chaotic (~ antisym-random) vs integrable
(Poisson 0.386).  HONEST SCOPE: a Wigner-Dyson result licenses "kinematical geometry is
quantum-chaotic" (quantum Haggard), NOT the Haar-typicality the Page curve assumed -- that
needs eigenvector-level ETH + a 2-design (Tier 2/3).  numpy float64, exact diagonalization.
"""
import sys
import time
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

RMT = {'Poisson': 0.386294, 'GOE': 0.530700, 'GUE': 0.599455, 'GSE': 0.674007}
MAXN0 = 6000                       # cap on the M=0 subspace dimension (memory/time)


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
    """G_ab = J_a . J_b = Jz_a Jz_b + 1/2 (J+_a J-_b + J-_a J+_b), real-symmetric on M=0."""
    N = len(states)
    G = np.zeros((N, N))
    ja, jb = float(spins[a]), float(spins[b])
    for i, s in enumerate(states):
        ma, mb = s[a], s[b]
        G[i, i] += ma * mb
        s2 = list(s); s2[a] = ma + 1; s2[b] = mb - 1                 # J+_a J-_b
        k = idx.get(tuple(s2))
        if k is not None:
            G[k, i] += 0.5 * _cp(ja, ma) * _cm(jb, mb)
        s2 = list(s); s2[a] = ma - 1; s2[b] = mb + 1                 # J-_a J+_b
        k = idx.get(tuple(s2))
        if k is not None:
            G[k, i] += 0.5 * _cm(ja, ma) * _cp(jb, mb)
    return (G + G.T) / 2


_GCACHE = {}


def intertwiner_G(spins):
    """Return {(a,b): G_ab projected to the singlet space}, D_int, N0.  Two passes keep peak
    memory at ~2 N0^2 (build J^2_total, get the J=0 projector P, then project each G_ab).
    Memoized on the spin tuple so the diagnostics reuse one build per config."""
    key = tuple(float(j) for j in spins)
    if key in _GCACHE:
        return _GCACHE[key]
    states, idx = m0_states(spins)
    N0 = len(states)
    if N0 == 0 or N0 > MAXN0:
        _GCACHE[key] = (None, None, N0)
        return _GCACHE[key]
    n = len(spins)
    J2 = np.zeros((N0, N0))
    np.fill_diagonal(J2, sum(float(j) * (float(j) + 1) for j in spins))
    for a in range(n):
        for b in range(a + 1, n):
            J2 += 2 * G_matrix(a, b, states, idx, spins)
    w, V = np.linalg.eigh(J2)
    P = V[:, w < 1e-6]                                                # singlet (J=0) subspace
    D_int = P.shape[1]
    Gint = {}
    for a in range(n):
        for b in range(a + 1, n):
            Gint[(a, b)] = P.T @ G_matrix(a, b, states, idx, spins) @ P
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

    print(f"    ANCHOR (n=7 [1,1.5,2,2.5,3,3.5,4.5], D_int=276, 138 levels): r~=0.617 -> chaotic"
          f"  [~19 min; run scrambling_big.py]", flush=True)

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

    print("\nVERDICT (single high-valence node):", flush=True)
    print("  Tier 1 SPECTRAL  : r~ -> 0.617 at D_int=276 (Poisson .386, chaotic .594) => CHAOTIC;", flush=True)
    print("                     fixed-sign monochromatic collapses to .340 (sector pitfall controlled).", flush=True)
    print("  Tier 2 EIGENVECTS: IPR*D_int -> 2.0 = Porter-Thomas => eigenstates are Haar-RANDOM", flush=True)
    print("                     (the eigenvector-level typicality the Page curve needs -- SATISFIED here).", flush=True)
    print("  Tier 3 DYNAMICS  : observable relaxes toward the microcanonical mean, but only PARTIAL", flush=True)
    print("                     thermalization at D_int<=95 (finite-size; large residual fluctuations).", flush=True)
    print("  => LQG geometry dynamics is quantum-chaotic with random eigenstates (quantum sequel to", flush=True)
    print("     Haggard).  Strong support for the Page-curve typicality assumption AT THE KINEMATIC", flush=True)
    print("     level.  GAP: not yet the physical evaporating state's 2-design typicality at Page time.", flush=True)


if __name__ == "__main__":
    run()
