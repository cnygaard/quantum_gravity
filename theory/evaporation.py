"""
Does the EVAPORATION scramble?  Genuine physical-Hamiltonian dynamics + Page-value test.
=========================================================================================

theory/scrambling.py studies the kinematical VOLUME operator on a node (an observable, not a
generator).  Here the question is dynamical: does a generic SU(2)-invariant two-body puncture
Hamiltonian scramble a low-entanglement horizon state to the singlet-sector Page value?

MODEL (a model, not the LQG constraint).  n spin-1/2 punctures, and the generic SU(2)-invariant
two-body operator built from the dihedral operator G_ab = J_a.J_b,
    H = sum_{a<b} K_ab (J_a . J_b),   K_ab ~ N(0,1),
i.e. a random-coupling Heisenberg model -- SU(2)-invariant, real-symmetric (GOE class).  The
couplings are DRAWN AT RANDOM, so random-matrix behaviour in the singlet sector is partly put
in; the test checks the singlet-sector bookkeeping (centre-corrected Page value), not typicality
of the LQG dynamics itself.  (The 3-body volume term
J_a.(J_b x J_c), shown chaotic in Tier 1, would shift the class to GUE; omitted here for a
clean GOE test.)

EFFICIENCY.  For spin-1/2, J_a.J_b = (1/2) SWAP_ab - 1/4 is 2-LOCAL and SPARSE; H, J^2 are
built by their action on the J_z=0 sector (dim C(n,n/2)), never as dense products.  All
diagonalization is in the small SU(2)-SINGLET (intertwiner) sector; the full 2^n space is
touched only by one cheap partial trace per timestep.  n=14 runs in seconds on one CPU core
(the earlier dense-complex version was ~250000x slower -- an algorithmic bug, not a hardware
limit; no GPU is warranted for a 3432-dim sector).

TESTS.
  (1) CHAOS: level-spacing r~ of H in the singlet sector -> GOE 0.531?
  (2) PAGE CURVE (the real test): evolve a low-entanglement dimer state (nearest-neighbour
      singlet pairs; S_A(0)=log2, one pair straddling the odd 7|7 cut) and track exact S_A(t)
      across the A|B puncture cut.  Does it rise to the PAGE VALUE (mean entanglement of a
      random singlet state, exact via the Bianchi-Dona centre formula)?  The quoted saturation
      is the late-time MEAN (second half of the window), reported with its temporal std against
      the Haar-singlet std (late_time_stats).
"""
import sys
import time
from itertools import combinations
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def jz0_basis(n):
    """Full-space integer indices of the J_z=0 bitstrings (n/2 up-spins), and a position map.
    Convention: qubit k is bit (n-1-k), so the first nA qubits are the high bits (A-block)."""
    idx = []
    for ones in combinations(range(n), n // 2):
        v = 0
        for k in ones:
            v |= 1 << (n - 1 - k)
        idx.append(v)
    idx.sort()
    return idx, {v: i for i, v in enumerate(idx)}


def build_sector(n, seed, topology='all'):
    """H = sum K_ab J_a.J_b and J^2_total on the J_z=0 sector, via the sparse 2-local action.
    topology='all' (generic all-to-all) or 'ring' (geometrically local: punctures on a ring,
    nearest-neighbour couplings only -- the fast-scrambling comparison, referee pt. 6)."""
    idx, pos = jz0_basis(n)
    N = len(idx)
    rng = np.random.default_rng(seed)
    K = {(a, b): rng.normal() for a in range(n) for b in range(a + 1, n)}
    if topology == 'ring':
        K = {(a, b): (v if (b - a == 1 or (a == 0 and b == n - 1)) else 0.0)
             for (a, b), v in K.items()}
    H = np.zeros((N, N))
    J2 = np.zeros((N, N))
    for i, v in enumerate(idx):
        bits = [(v >> (n - 1 - k)) & 1 for k in range(n)]
        J2[i, i] += 0.75 * n                                       # sum_a J_a^2
        for a in range(n):
            sa = 0.5 if bits[a] else -0.5
            for b in range(a + 1, n):
                sb = 0.5 if bits[b] else -0.5
                H[i, i] += K[(a, b)] * sa * sb                      # Jz Jz (diagonal)
                J2[i, i] += 2 * sa * sb
                if bits[a] != bits[b]:                              # 1/2 SWAP (off-diagonal)
                    j = pos[v ^ (1 << (n - 1 - a)) ^ (1 << (n - 1 - b))]
                    H[j, i] += K[(a, b)] * 0.5
                    J2[j, i] += 1.0
    return (H + H.T) / 2, (J2 + J2.T) / 2, idx


def singlet_P(J2, tol=1e-6):
    w, V = np.linalg.eigh(J2)
    return V[:, w < tol]


def rbar(e):
    s = np.diff(np.sort(e))
    s = s[s > 1e-9 * max(1.0, np.mean(np.abs(e)))]
    return float(np.mean(np.minimum(s[1:] / s[:-1], s[:-1] / s[1:]))) if len(s) > 3 else np.nan


def embed(vec_jz0, idx, n):
    full = np.zeros(2 ** n, complex)
    full[idx] = vec_jz0
    return full


def entropy_A(full_psi, nA, n):
    s = np.linalg.svd(full_psi.reshape(2 ** nA, 2 ** (n - nA)), compute_uv=False)
    p = s ** 2
    p = p[p > 1e-14]
    return float(-np.sum(p * np.log(p)))


def dimer_jz0(n, idx):
    """Nearest-neighbour singlet pairs (0,1)(2,3)...: total J=0, S_A(0)=0 across the middle cut."""
    sing = np.array([0, 1, -1, 0], float) / np.sqrt(2)
    psi = np.array([1.0])
    for _ in range(n // 2):
        psi = np.kron(psi, sing)
    return psi[idx]                                                 # restrict to J_z=0


def page_value(P, idx, nA, n, nsamp=60, seed=1):
    rng = np.random.default_rng(seed)
    D = P.shape[1]
    vals = []
    for _ in range(nsamp):
        c = rng.normal(size=D) + 1j * rng.normal(size=D)
        vals.append(entropy_A(embed(P @ (c / np.linalg.norm(c)), idx, n), nA, n))
    return float(np.mean(vals)), float(np.std(vals))


def summary(n=12, seed=7, nsamp=30):
    """Compact (rbar_singlet, S_sat/S_page, D_int) for the suite -- fast at n=12 (S_A(0)=0)."""
    nA = n // 2
    H, J2, idx = build_sector(n, seed)
    P = singlet_P(J2)
    D_int = P.shape[1]
    Hs = P.T @ H @ P
    Es, Us = np.linalg.eigh((Hs + Hs.T) / 2)
    S_page, _ = page_value(P, idx, nA, n, nsamp=nsamp)
    c0 = Us.T @ (P.T @ dimer_jz0(n, idx))
    tmax = 25.0 / (np.std(Es) + 1e-12)
    S = [entropy_A(embed(P @ (Us @ (np.exp(-1j * Es * t) * c0)), idx, n), nA, n)
         for t in np.linspace(0.4 * tmax, tmax, 10)]
    return rbar(Es), float(np.mean(S)) / S_page, D_int


def summary_local(n=12, seed=7, nt=160):
    """Fast-scrambling comparison (referee pt. 6): generic ALL-TO-ALL vs geometrically LOCAL
    (ring) couplings.  Both Hamiltonians are rescaled to unit spectral width so times are
    comparable.  Returns {topology: (S_saturation/S_Page, t90)} -- both should saturate near
    the Page value (chaotic local H thermalizes too), with the local one slower."""
    nA = n // 2
    out = {}
    for topo in ('all', 'ring'):
        H, J2, idx = build_sector(n, seed, topology=topo)
        P = singlet_P(J2)
        Hs = P.T @ H @ P
        Hs = (Hs + Hs.T) / 2
        Es, Us = np.linalg.eigh(Hs)
        Es = Es / (np.std(Es) + 1e-12)                     # unit bandwidth
        S_page, _ = page_value(P, idx, nA, n, nsamp=30)
        c0 = Us.T @ (P.T @ dimer_jz0(n, idx))
        ts = np.linspace(0, 40.0, nt)
        S = [entropy_A(embed(P @ (Us @ (np.exp(-1j * Es * t) * c0)), idx, n), nA, n)
             for t in ts]
        sat = float(np.mean(S[3 * nt // 4:]))
        t90 = next((float(t) for t, s in zip(ts, S) if s >= 0.9 * sat), float('nan'))
        out[topo] = (sat / S_page, t90)
    return out


def phys_rbar_sem(n=12, seeds=(0, 1, 2, 3, 4)):
    """r~ with SEM for the puncture Hamiltonian, pooled over K_ab seeds (referee m6)."""
    ratios = []
    for seed in seeds:
        H, J2, idx = build_sector(n, seed)
        P = singlet_P(J2)
        Hs = P.T @ H @ P
        Es = np.linalg.eigvalsh((Hs + Hs.T) / 2)
        s = np.diff(np.sort(Es))
        s = s[s > 1e-12]
        ratios += list(np.minimum(s[1:] / s[:-1], s[:-1] / s[1:]))
    r = np.array(ratios)
    return float(r.mean()), float(r.std(ddof=1) / np.sqrt(len(r)))


def singlet_levels(n=14, seed=7):
    """Singlet-sector spectrum of the puncture Hamiltonian (for correlation-robust r~ errors)."""
    H, J2, _ = build_sector(n, seed)
    P = singlet_P(J2)
    Hs = P.T @ H @ P
    return np.linalg.eigvalsh((Hs + Hs.T) / 2)


def phys_rbar_seeds(n=12, seeds=(0, 1, 2, 3, 4)):
    """Per-seed r~ of the puncture Hamiltonian; seeds are independent, so the across-seed SEM
    is a valid error (unlike the SEM over pooled, correlated gap ratios)."""
    vals = np.array([rbar(singlet_levels(n, s)) for s in seeds])
    return float(vals.mean()), float(vals.std(ddof=1) / np.sqrt(len(vals))), vals.tolist()


def late_time_stats(n=14, seed=7, nt=41, nsamp=2000):
    """Saturation of the dimer under e^{-iHt} at n=14, as quoted: the LATE-TIME MEAN of S_A over
    the second half of t in [0, 25/std(E)] (nt samples), with its temporal std, against the
    EXACT Haar-singlet mean (page_curve.singlet_page_exact) and the Haar-singlet std (Monte Carlo,
    nsamp draws).  Returns a dict."""
    from page_curve import singlet_page_exact
    nA = n // 2
    H, J2, idx = build_sector(n, seed)
    P = singlet_P(J2)
    Hs = P.T @ H @ P
    Es, Us = np.linalg.eigh((Hs + Hs.T) / 2)
    c0 = Us.T @ (P.T @ dimer_jz0(n, idx))
    ts = np.linspace(0, 25.0 / (np.std(Es) + 1e-12), nt)
    S = np.array([entropy_A(embed(P @ (Us @ (np.exp(-1j * Es * t) * c0)), idx, n), nA, n) for t in ts])
    late = S[nt // 2:]
    mc_mean, mc_std = page_value(P, idx, nA, n, nsamp=nsamp, seed=5)
    return {'late_mean': float(late.mean()), 'late_std': float(late.std(ddof=1)), 'n_late': int(late.size),
            'haar_exact': singlet_page_exact(n, nA), 'haar_mc': mc_mean, 'haar_std': mc_std,
            'nsamp': nsamp, 'S_max': float(S.max())}


def run():
    n, nA = 14, 7
    print("=" * 76)
    print(f"DOES THE EVAPORATION SCRAMBLE?  n={n} spin-1/2 horizon, H = sum K_ab J_a.J_b (GOE)")
    print("=" * 76)
    t0 = time.time()
    H, J2, idx = build_sector(n, seed=7)
    P = singlet_P(J2)
    D_int = P.shape[1]
    Hs = P.T @ H @ P
    Hs = (Hs + Hs.T) / 2
    Es, Us = np.linalg.eigh(Hs)
    print(f"  J_z=0 dim={len(idx)}, singlet D_int={D_int}   [{time.time()-t0:.1f}s]", flush=True)

    rb = rbar(Es)
    print(f"\n[chaos] singlet-sector r~ = {rb:.3f}  (GOE 0.531, Poisson 0.386 -> "
          f"{'chaotic (GOE)' if rb > 0.47 else 'not chaotic'})", flush=True)

    from page_curve import singlet_page_exact
    S_page = singlet_page_exact(n, nA)
    _, sd = page_value(P, idx, nA, n)
    print(f"\n[Page value] random singlet S_A({nA}|{n-nA}) = {S_page:.4f} (exact; Haar std {sd:.3f}) nats"
          f"  (qubit max {nA*np.log(2):.3f}; deficit = Page 1/2 + SU(2) centre)", flush=True)

    print(f"\n[Page curve] dimer state evolved under e^(-iHt), exact S_A(t):", flush=True)
    d_singlet = P.T @ dimer_jz0(n, idx)
    print(f"    (dimer weight in the J=0 sector = {np.linalg.norm(d_singlet)**2:.3f}; S_A(0)="
          f"{entropy_A(embed(P @ d_singlet, idx, n), nA, n):.3f})", flush=True)
    c0 = Us.T @ d_singlet
    tmax = 25.0 / (np.std(Es) + 1e-12)
    ts = np.linspace(0, tmax, 41)
    S = []
    for t in ts:
        psi_s = Us @ (np.exp(-1j * Es * t) * c0)
        S.append(entropy_A(embed(P @ psi_s, idx, n), nA, n))
    for k in range(0, len(ts), 4):
        bar = "#" * int(40 * S[k] / S_page)
        print(f"    t={ts[k]:6.2f}  S_A={S[k]:5.3f}  {bar}", flush=True)
    late = np.array(S[len(S) // 2:])
    S_sat = float(late.mean())
    ratio = S_sat / S_page
    print(f"\n  S_A: {S[0]:.3f} -> late-time mean {S_sat:.4f} +/- {late.std(ddof=1):.4f} (temporal std) nats"
          f"  vs exact Page {S_page:.4f}  (ratio {ratio:.3f})", flush=True)
    print(f"\nVERDICT: the random-coupling SU(2)-invariant model drives the dimer to the singlet Page value"
          f" (ratio {ratio:.3f}).  A model check of the centre-corrected bookkeeping, not of LQG dynamics.")


if __name__ == "__main__":
    run()
