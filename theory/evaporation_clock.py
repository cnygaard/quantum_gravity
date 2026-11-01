"""
Evaporation models: SU(2)-singlet puncture emission, and a U(1) occupation-encoded collision model.
==================================================================================================

theory/evaporation.py showed that a random-coupling SU(2)-invariant two-body Hamiltonian (a model,
not the LQG constraint) scrambles a fixed-spin horizon to the singlet Page value.  Two emission models follow.  In both, the time is
the discrete step count (the place a deparametrized clock field would enter; no clock field is
modelled), and in both the turnover S_rad -> 0 is guaranteed by purity of a finite register --
what the dynamics decides is how close the curve comes to the typical (Page) curve.

(A) SINGLET PUNCTURE EMISSION (page_curve_singlet; the quoted model).  n spin-1/2 punctures in
  a GLOBAL SU(2) SINGLET (gauge invariance of horizon + radiation), starting from the
  low-entanglement dimer state.  Each step the remaining horizon B is scrambled by
  H_B = sum_{a<b in B} K_ab J_a.J_b, then one whole puncture (its spin quantum and its area
  a_p) leaves B for the radiation R and is never touched again.  Every operation commutes with
  total J, so the state stays an exact singlet (<J^2> is certified at every step).  Compared
  cut-by-cut with the EXACT Haar-random-singlet Page curve (Bianchi-Dona centre formula,
  page_curve.singlet_page_exact; a Monte Carlo estimate is kept as a cross-check); the
  no-scrambling control (tau=0) gives the dimer sawtooth (peak ln 2), so the curve's height is
  supplied by the dynamics.

(B) OCCUPATION-ENCODED COLLISION MODEL (page_curve; legacy).  Fresh radiation modes are prepared
  in the fixed state |down>, which breaks SU(2) to U(1) even when the coupling J.sigma is
  SU(2)-invariant; J.sigma generates partial SWAPs (SWAP = 1/2 + J_a.sigma_q = 1/2 + 2 J_a.J_q), so this is a
  swap-out (Hayden-Preskill-type) qubit model whose turnover is built in by the finite register.

MODEL (B) -- repeated-interaction (collision) evaporation, the standard open-system picture of
Hawking radiation (LQG punctures-as-qubits; Bianchi: area = #punctures x area gap):
  * Black hole = n_BH spin-1/2 punctures, initially all occupied (full horizon, k=n_BH quanta).
  * Each clock step: (i) the horizon SCRAMBLES internally,  rho -> U_scr rho U_scr^dag,
    U_scr = e^{-i H_scr dt},  H_scr = sum_{a<b} K_ab J_a.J_b (SU(2)-invariant, chaotic);
    (ii) it EMITS: a FRESH empty radiation mode q interacts via the HOLONOMY hopping
    H_emit = g sum_a J_a . sigma_q  (Heisenberg exchange: an SU(2)-invariant COUPLING, but the
    |down> reference state of the fresh mode breaks SU(2) to U(1); the XX flip-flop variant
    exchange='xx' conserves only total M, with the radiation carrying the M), then the mode
    ESCAPES (is traced out, carrying its entanglement).
  OCCUPATION ENCODING (referee m4): spin-up = occupied area quantum; emission = de-excitation
  |up> -> |down> with the quantum carried by the escaping mode -- the puncture label remains
  but its area quantum is gone; a j=0 puncture is no puncture (KL color-0 convention).  This
  labels a U(1)/S_z quantity, not an SU(2) doublet.
  Because emitted quanta never return, the horizon steadily EMPTIES; no finite-bath plateau.

TIME.  The collision count is a discrete step label.  A physical relational clock would come
from deparametrization (a scalar clock phi turns p_phi + H_grav = 0 into
i d_phi|psi> = H_grav|psi>; Domagala-Giesel-Kaminski-Lewandowski 2010), but no clock field is
modelled here.  The global BH+radiation evolution is unitary, so S_rad = S(rho_BH).

DIAGNOSTIC: S(rho_BH) vs clock step.  The horizon starts pure (full) -> S=0; emission mixes it
-> S rises; as it EMPTIES (accessible Hilbert space shrinks with each lost quantum, C(n_BH,j))
it re-purifies -> S FALLS to 0.  The rise-then-fall is guaranteed by the finite register; the
scrambling sets how close the peak comes to half the capacity.  Only the small 2^n_BH horizon
density matrix is ever stored.
"""
import os
import sys
from itertools import combinations
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaporation as ev                                    # noqa: E402  (singlet-sector tools)
from page_curve import singlet_page_exact                   # noqa: E402  (exact Haar-singlet curve)

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex) / 2
SY = np.array([[0, -1j], [1j, 0]], complex) / 2
SZ = np.array([[1, 0], [0, -1]], complex) / 2
SP = SX + 1j * SY                                          # sigma^+/... raising (spin-1/2)
SM = SX - 1j * SY


def _site(op, a, n):
    M = np.array([[1]], complex)
    for k in range(n):
        M = np.kron(M, op if k == a else I2)
    return M


def _expm_h(H, t):
    w, V = np.linalg.eigh(H)
    return (V * np.exp(-1j * w * t)) @ V.conj().T


def entropy(rho):
    w = np.linalg.eigvalsh(rho)
    w = w[w > 1e-13]
    return float(-np.sum(w * np.log(w)))


def build_unitaries(n_BH, seed, dt_scr=2.0, g=1.0, dt_emit=0.12, exchange='xx'):
    """U_scr (scramble the horizon) and U_emit (horizon<->fresh mode coupling).
    exchange='xx'        : g sum_a (J_a^+ sigma^- + h.c.)  -- flip-flop only; breaks SU(2)
                           to U(1) (total M conserved; the radiation carries the M).
    exchange='heisenberg': g sum_a J_a . sigma_q           -- full SU(2)-invariant exchange
                           (referee m4); adds the Ising piece to the same flip-flop."""
    rng = np.random.default_rng(seed)
    Jx = [_site(SX, a, n_BH) for a in range(n_BH)]
    Jy = [_site(SY, a, n_BH) for a in range(n_BH)]
    Jz = [_site(SZ, a, n_BH) for a in range(n_BH)]
    Hs = np.zeros((2 ** n_BH,) * 2, complex)
    for a in range(n_BH):
        for b in range(a + 1, n_BH):
            Hs += rng.normal() * (Jx[a] @ Jx[b] + Jy[a] @ Jy[b] + Jz[a] @ Jz[b])
    U_scr = _expm_h(Hs, dt_scr)
    d = 2 ** n_BH
    He = np.zeros((2 * d, 2 * d), complex)
    Jp = [_site(SP, a, n_BH) for a in range(n_BH)]
    Jm = [_site(SM, a, n_BH) for a in range(n_BH)]
    for a in range(n_BH):
        if exchange == 'heisenberg':
            He += g * (0.5 * (np.kron(Jp[a], SM) + np.kron(Jm[a], SP))
                       + np.kron(Jz[a], SZ))
        else:
            He += g * (np.kron(Jp[a], SM) + np.kron(Jm[a], SP))
    U_emit = _expm_h(He, dt_emit)
    return U_scr, U_emit


def emit(rho, U_emit, d):
    """One collision with a fresh empty mode |0>, then trace it out (it escapes to infinity)."""
    q0 = np.zeros((2, 2), complex)
    q0[0, 0] = 1.0                                            # fresh mode empty (down)
    rho_full = U_emit @ np.kron(rho, q0) @ U_emit.conj().T
    return rho_full.reshape(d, 2, d, 2).trace(axis1=1, axis2=3)


def page_curve(n_BH, seeds, nsteps=75, dt_scr=2.0, g=1.0, dt_emit=0.12, exchange='xx'):
    d = 2 ** n_BH
    full = np.zeros(d, complex)
    full[-1] = 1.0                                           # all-up = full horizon (pure)
    rho0 = np.outer(full, full.conj())
    curves = []
    for seed in seeds:
        U_scr, U_emit = build_unitaries(n_BH, seed, dt_scr, g, dt_emit, exchange)
        rho = rho0.copy()
        S = [entropy(rho)]
        for _ in range(nsteps):
            rho = U_scr @ rho @ U_scr.conj().T
            rho = emit(rho, U_emit, d)
            rho = (rho + rho.conj().T) / 2
            S.append(entropy(rho))
        curves.append(S)
    return np.mean(curves, axis=0)


def summary(n_BH=6, seeds=(0, 1, 2, 3), nsteps=120, dt_emit=0.12, exchange='xx'):
    """Compact (peak, end, step_peak, nsteps) of the Page curve, for the suite."""
    S = page_curve(n_BH, seeds, nsteps=nsteps, dt_emit=dt_emit, exchange=exchange)
    return float(S.max()), float(np.mean(S[-4:])), int(np.argmax(S)), len(S)


# ------------------------------------------------------------------ (A) singlet puncture emission
def sector_H(n, idx, pos, K, members):
    """H_B = sum_{a<b in members} K[a,b] J_a.J_b on the J_z=0 sector (sparse CSR).
    J_a.J_b = Jz Jz + (1/2)(flip-flop): diagonal Ising part plus half-SWAP off-diagonals."""
    rows, cols, vals = [], [], []
    mem = sorted(members)
    for i, v in enumerate(idx):
        bits = [(v >> (n - 1 - k)) & 1 for k in range(n)]
        diag = 0.0
        for x, a in enumerate(mem):
            sa = 0.5 if bits[a] else -0.5
            for b in mem[x + 1:]:
                sb = 0.5 if bits[b] else -0.5
                diag += K[a, b] * sa * sb
                if bits[a] != bits[b]:
                    rows.append(pos[v ^ (1 << (n - 1 - a)) ^ (1 << (n - 1 - b))])
                    cols.append(i)
                    vals.append(0.5 * K[a, b])
        rows.append(i)
        cols.append(i)
        vals.append(diag)
    N = len(idx)
    return sp.csr_matrix((vals, (rows, cols)), shape=(N, N))


_SINGLET_CACHE = {}


def page_curve_singlet(n=14, seeds=(0, 1, 2, 3), tau=3.0, nsamp=30, nquench=1):
    """Singlet-preserving puncture emission (model A).  Puncture k leaves the horizon at step k
    (k = 0..n-1) after the remaining horizon B_k = {k..n-1} has been scrambled for tau inverse
    bandwidths of H_B.  Returns dict: S (radiation entropy after k emissions, k = 0..n, seed
    mean), page (EXACT Haar-random-singlet S of the first k punctures), page_mc (Monte Carlo
    estimate with nsamp draws, cross-check), mid_ratio = S/page at k=n/2, max_rel_dev over
    interior cuts, below = True if S < page at every interior cut, j2_max = max <J_tot^2>.
    nquench > 1: each step applies nquench evolutions of tau each under FRESH coupling draws
    (removes the single-Hamiltonian energy envelope; control for energy conservation)."""
    if (n, nsamp) not in _SINGLET_CACHE:
        idx, pos = ev.jz0_basis(n)
        _, J2, _ = ev.build_sector(n, 0)                   # J^2 does not depend on the seed
        P = ev.singlet_P(J2)
        page = np.array([singlet_page_exact(n, k) for k in range(n + 1)])
        page_mc = np.array([0.0] + [ev.page_value(P, idx, k, n, nsamp=nsamp)[0] for k in range(1, n)]
                           + [0.0])
        _SINGLET_CACHE[(n, nsamp)] = (idx, pos, P, sp.csr_matrix(J2), page, page_mc)
    idx, pos, P, J2s, page, page_mc = _SINGLET_CACHE[(n, nsamp)]
    psi0 = ev.dimer_jz0(n, idx).astype(complex)
    psi0 /= np.linalg.norm(psi0)
    curves, j2_max = [], 0.0
    for seed in seeds:
        rng = np.random.default_rng(100 + seed)
        K = rng.normal(size=(n, n))
        psi = psi0.copy()
        S = [0.0]
        for k in range(n):
            for q in range(nquench if tau > 0 and n - k > 1 else 0):
                if nquench > 1:
                    K = rng.normal(size=(n, n))
                H = sector_H(n, idx, pos, K, range(k, n))
                D = H.shape[0]
                width = np.sqrt(max(H.multiply(H).sum() / D - (H.diagonal().sum() / D) ** 2, 1e-30))
                psi = expm_multiply(-1j * (tau / width) * H, psi)
            j2_max = max(j2_max, abs(np.vdot(psi, J2s @ psi)))
            S.append(ev.entropy_A(ev.embed(psi, idx, n), k + 1, n))
        curves.append(S)
    S = np.mean(curves, axis=0)
    interior = slice(1, n)
    rel = (S[interior] - page[interior]) / page[interior]
    return {'S': S, 'page': page, 'page_mc': page_mc, 'mid_ratio': float(S[n // 2] / page[n // 2]),
            'max_rel_dev': float(np.max(np.abs(rel))), 'argmax_dev': int(np.argmax(np.abs(rel)) + 1),
            'below': bool(np.all(rel[1:-1] < 0)), 'j2_max': float(j2_max), 'D_int': P.shape[1]}


def run():
    print("=" * 80)
    print("(A) SINGLET PUNCTURE EMISSION: global SU(2) singlet, whole punctures leave the horizon")
    print("=" * 80)
    n = 14
    res = {}
    for tau in (0.0, 1.0, 3.0, 6.0, 10.0):
        res[tau] = r = page_curve_singlet(n=n, tau=tau)
        tag = 'no-scrambling control' if tau == 0 else f'tau={tau:g} inverse bandwidths per step'
        print(f"  [{tag}]  S(R_{n//2}) = {r['S'][n//2]:.3f} vs Haar-singlet {r['page'][n//2]:.4f}"
              f"  (ratio {r['mid_ratio']:.3f}; max rel. dev. {r['max_rel_dev']:.3f} at k={r['argmax_dev']}; "
              f"below Haar at every interior cut: {r['below']}; max <J^2> {r['j2_max']:.1e})", flush=True)
    r = res[3.0]
    for k in range(n + 1):
        bar = "#" * int(40 * r['S'][k] / r['page'][n // 2])
        print(f"    k={k:2d}  S_rad={r['S'][k]:5.3f}  Haar-singlet={r['page'][k]:6.4f}  {bar}", flush=True)
    print("\n  Endpoints are fixed by symmetry and purity (S(R_1)=ln2, S(R_n)=0); the height of the")
    print("  curve is supplied by the SU(2)-invariant scrambling (control: ln2 sawtooth).\n")

    n_BH = 6
    seeds = tuple(range(12))
    print("=" * 80)
    print("(B) U(1) OCCUPATION-ENCODED COLLISION MODEL (fresh |down> modes; partial-SWAP exchange)")
    print(f"  horizon = {n_BH} spin-1/2 punctures (full); S(rho_BH) averaged over {len(seeds)} realizations")
    print("=" * 80)
    # Heisenberg exchange J_a.sigma_q (dt compensates the halved flip-flop coefficient); XX
    # variant retained as an option.
    S = page_curve(n_BH, seeds, nsteps=130, dt_emit=0.24, exchange='heisenberg')
    Speak, kpeak = float(S.max()), int(np.argmax(S))
    Send = float(np.mean(S[-4:]))
    turn = (Speak - Send) / (Speak + 1e-9)
    print(f"  peak S_rad={Speak:.3f} at step {kpeak} (of {len(S)-1}); S_rad(end)={Send:.3f}\n", flush=True)
    for k in range(0, len(S), 5):
        bar = "#" * int(50 * S[k] / (Speak + 1e-9))
        print(f"    step {k:3d}  S_rad={S[k]:5.3f}  {bar}"
              f"{'  <- Page time' if abs(k - kpeak) < 2 else ''}", flush=True)
    print(f"\n  S_rad: 0 -> peak {Speak:.3f} -> {Send:.3f}   turnover = {turn:.1%}", flush=True)
    print("  (turnover is guaranteed by the finite register; scrambling sets how close the peak")
    print("   comes to the half-capacity bound)")


if __name__ == "__main__":
    run()
