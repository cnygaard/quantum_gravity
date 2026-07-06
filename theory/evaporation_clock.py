"""
The genuine evaporation: spin-changing emission + a relational clock -> dynamical Page curve.
=============================================================================================

Tier 3+ (theory/evaporation.py) showed a physical SU(2)-invariant Hamiltonian scrambles the
horizon to the Page value AT FIXED SPINS.  The remaining step is the genuine evaporation:
spin-CHANGING dynamics (area quanta LEAVE the horizon, irreversibly, to infinity), in a
well-defined relational TIME, with the radiation entropy tracing the full Page curve.

MODEL -- repeated-interaction (collision) evaporation, the standard open-system picture of
Hawking radiation (LQG punctures-as-qubits; Bianchi: area = #punctures x area gap):
  * Black hole = n_BH spin-1/2 punctures, initially all occupied (full horizon, k=n_BH quanta).
  * Each clock step: (i) the horizon SCRAMBLES internally,  rho -> U_scr rho U_scr^dag,
    U_scr = e^{-i H_scr dt},  H_scr = sum_{a<b} K_ab J_a.J_b (SU(2)-invariant, chaotic);
    (ii) it EMITS: a FRESH empty radiation mode q interacts via the HOLONOMY hopping
    H_emit = g sum_a (J_a^+ sigma_q^- + h.c.), then ESCAPES to infinity (is traced out,
    carrying its entanglement) -- a genuine spin-changing emission that removes area quanta.
  Because emitted quanta never return, the horizon steadily EMPTIES; no finite-bath plateau.

THE CLOCK (deparametrization; Rovelli; Domagala-Giesel-Kaminski-Lewandowski 2010).  A scalar
clock phi turns p_phi + H_grav = 0 into i d_phi|psi> = H_grav|psi>; the collision count IS the
relational time phi.  The global BH+radiation evolution is unitary, so S_rad = S(rho_BH).

DIAGNOSTIC: S(rho_BH) vs clock step.  The horizon starts pure (full) -> S=0; emission mixes it
-> S rises; as it EMPTIES (accessible Hilbert space shrinks with each lost quantum, C(n_BH,j))
it re-purifies -> S FALLS to 0.  Rise-then-fall = the PAGE CURVE, from genuine spin-changing
emission in relational time.  Only the small 2^n_BH horizon density matrix is ever stored.
"""
import sys
from itertools import combinations
import numpy as np

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


def build_unitaries(n_BH, seed, dt_scr=2.0, g=1.0, dt_emit=0.12):
    """U_scr (scramble the horizon) and U_emit (horizon<->fresh mode holonomy hop)."""
    rng = np.random.default_rng(seed)
    Jx = [_site(SX, a, n_BH) for a in range(n_BH)]
    Jy = [_site(SY, a, n_BH) for a in range(n_BH)]
    Jz = [_site(SZ, a, n_BH) for a in range(n_BH)]
    Hs = np.zeros((2 ** n_BH,) * 2, complex)
    for a in range(n_BH):
        for b in range(a + 1, n_BH):
            Hs += rng.normal() * (Jx[a] @ Jx[b] + Jy[a] @ Jy[b] + Jz[a] @ Jz[b])
    U_scr = _expm_h(Hs, dt_scr)
    # emission on horizon (x) fresh mode q : sum_a (J_a^+ sigma^- + J_a^- sigma^+)
    d = 2 ** n_BH
    He = np.zeros((2 * d, 2 * d), complex)
    Jp = [_site(SP, a, n_BH) for a in range(n_BH)]
    Jm = [_site(SM, a, n_BH) for a in range(n_BH)]
    for a in range(n_BH):
        He += g * (np.kron(Jp[a], SM) + np.kron(Jm[a], SP))
    U_emit = _expm_h(He, dt_emit)
    return U_scr, U_emit


def emit(rho, U_emit, d):
    """One collision with a fresh empty mode |0>, then trace it out (it escapes to infinity)."""
    q0 = np.zeros((2, 2), complex)
    q0[0, 0] = 1.0                                            # fresh mode empty (down)
    rho_full = U_emit @ np.kron(rho, q0) @ U_emit.conj().T
    return rho_full.reshape(d, 2, d, 2).trace(axis1=1, axis2=3)


def page_curve(n_BH, seeds, nsteps=75, dt_scr=2.0, g=1.0, dt_emit=0.12):
    d = 2 ** n_BH
    full = np.zeros(d, complex)
    full[-1] = 1.0                                           # all-up = full horizon (pure)
    rho0 = np.outer(full, full.conj())
    curves = []
    for seed in seeds:
        U_scr, U_emit = build_unitaries(n_BH, seed, dt_scr, g, dt_emit)
        rho = rho0.copy()
        S = [entropy(rho)]
        for _ in range(nsteps):
            rho = U_scr @ rho @ U_scr.conj().T
            rho = emit(rho, U_emit, d)
            rho = (rho + rho.conj().T) / 2
            S.append(entropy(rho))
        curves.append(S)
    return np.mean(curves, axis=0)


def summary(n_BH=6, seeds=(0, 1, 2, 3), nsteps=120):
    """Compact (peak, end, step_peak, nsteps) of the Page curve, for the suite."""
    S = page_curve(n_BH, seeds, nsteps=nsteps)
    return float(S.max()), float(np.mean(S[-4:])), int(np.argmax(S)), len(S)


def run():
    n_BH = 6
    seeds = tuple(range(12))
    print("=" * 80)
    print("GENUINE EVAPORATION: spin-changing emission + relational clock -> Page curve")
    print(f"  horizon = {n_BH} spin-1/2 punctures (full), radiation escapes irreversibly (collisions)")
    print(f"  S(rho_BH) averaged over {len(seeds)} scrambling realizations")
    print("=" * 80)
    S = page_curve(n_BH, seeds, nsteps=130)
    Speak, kpeak = float(S.max()), int(np.argmax(S))
    Send = float(np.mean(S[-4:]))
    turn = (Speak - Send) / (Speak + 1e-9)
    print(f"  peak S_rad={Speak:.3f} at step {kpeak} (of {len(S)-1}); S_rad(end)={Send:.3f}\n", flush=True)
    for k in range(0, len(S), 5):
        bar = "#" * int(50 * S[k] / (Speak + 1e-9))
        print(f"    step {k:3d}  S_rad={S[k]:5.3f}  {bar}"
              f"{'  <- Page time' if abs(k - kpeak) < 2 else ''}", flush=True)
    print(f"\n  S_rad: 0 -> peak {Speak:.3f} (~half-evaporation) -> {Send:.3f}   turnover = {turn:.0%}",
          flush=True)
    if turn > 0.6 and 3 < kpeak < len(S) - 4:
        print("\nVERDICT: THE PAGE CURVE.  S_rad rises from 0, peaks at ~half-evaporation, and RETURNS")
        print("  toward 0 as the horizon empties -- from genuine spin-changing emission dynamics in")
        print("  relational (deparametrized-clock) time.  Information is returned to the radiation;")
        print("  the finite LQG horizon re-purifies.  This is the dynamical Page curve, derived.")
    else:
        print(f"\nVERDICT: partial ({turn:.0%} turnover); tune dt_emit/dt_scr for a cleaner arc.")


if __name__ == "__main__":
    run()
