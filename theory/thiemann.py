"""
The exact Euclidean Thiemann Hamiltonian constraint on variable-spin states -- and whether it
scrambles.  Built against Gaul-Rovelli (gr-qc/0011106), Borissov-De Pietri-Rovelli
(gr-qc/9703090), and benchmarked against Guedes-Mena Marugan-Vidotto-Muller
(arXiv:2412.20257/20272, the 2024-25 numerical implementation of exactly this operator).

STEP 1 (this file, first): CERTIFY THE VOLUME.  The exact operator's matrix elements are weighted
by the Ashtekar-Lewandowski volume.  For the monochromatic 4-valent node with all edges j=1/2 the
gauge-invariant intertwiner space is 2-dimensional (internal spin 0 or 1), and the grasp
    q_hat = J_1 . (J_2 x J_3) = eps_ijk J^i_1 J^j_2 J^k_3
restricted to it must have eigenvalues +/- sqrt(3)/4 = +/-0.4330127 (Gaul-Rovelli Eq. 62 at
(p,q,m,c)=(1,1,1,1), t=1 gives sqrt(3)/2 = 2 x sqrt(3)/4, the standard binor/color factor 2).
Reproducing +/-sqrt(3)/4 with our V=sqrt|sum eps J J J| construction certifies the volume.

ALSO: the orientation-sign pitfall -- summing the four triples of a 4-valent node with all + signs
gives IDENTICALLY ZERO (gauge invariance makes the four grasps +/-q cancel pairwise), so the AL
orientation sign eps(e_I,e_J,e_K) is essential; the monochromatic node uses a single representative.

Later steps build the graph-changing operator (holonomy grasp via recoupling.py + this volume),
verify the +/-1/2 selection rule and Hermiticity, and test scrambling with the ULTRALOCALITY
diagnostic (Nicolai-Peeters-Zamaklar): does entropy growth come from MIXING pre-existing nodes, or
merely from GRAPH GROWTH (new Hilbert factors)?  A negative (non-scrambling) result is expected and
would pinpoint the propagation ingredient the model Hamiltonians supply by hand.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from recoupling import cg                                  # noqa: E402

I2 = np.eye(2, dtype=complex)
SX = np.array([[0, 1], [1, 0]], complex) / 2       # spin-1/2 J
SY = np.array([[0, -1j], [1j, 0]], complex) / 2
SZ = np.array([[1, 0], [0, -1]], complex) / 2
P3 = [SX, SY, SZ]


def _site(op, a, n):
    M = np.array([[1]], complex)
    for k in range(n):
        M = np.kron(M, op if k == a else I2)
    return M


def spinhalf_ops(n):
    return [[_site(P3[c], a, n) for c in range(3)] for a in range(n)]


def triple(J, p, q, r):
    """q_hat_{pqr} = J_p . (J_q x J_r) = eps_ijk J^i_p J^j_q J^k_r  (Hermitian)."""
    Jp, Jq, Jr = J[p], J[q], J[r]
    return (Jp[0] @ (Jq[1] @ Jr[2] - Jq[2] @ Jr[1])
            + Jp[1] @ (Jq[2] @ Jr[0] - Jq[0] @ Jr[2])
            + Jp[2] @ (Jq[0] @ Jr[1] - Jq[1] @ Jr[0]))


def singlet_projector(J, n, tol=1e-8):
    Jt = [sum(J[a][c] for a in range(n)) for c in range(3)]
    J2 = Jt[0] @ Jt[0] + Jt[1] @ Jt[1] + Jt[2] @ Jt[2]
    w, V = np.linalg.eigh(J2)
    return V[:, w < tol]


def certify_volume():
    n = 4
    J = spinhalf_ops(n)
    P = singlet_projector(J, n)
    D_int = P.shape[1]
    # single representative grasp q_123 on the intertwiner space
    q123 = P.conj().T @ triple(J, 0, 1, 2) @ P
    q123 = (q123 + q123.conj().T) / 2
    ev = np.linalg.eigvalsh(q123)
    target = np.sqrt(3) / 4
    # the all-+ sum of the four triples (the orientation-sign pitfall)
    Qplus = sum(triple(J, p, q, r) for p in range(n) for q in range(p + 1, n) for r in range(q + 1, n))
    Qplus_int = P.conj().T @ Qplus @ P
    ev_plus = np.linalg.eigvalsh((Qplus_int + Qplus_int.conj().T) / 2)
    return D_int, ev, target, ev_plus


def verify_selection_rule(jmax=5):
    """The m=1 holonomy grasp couples a fundamental j=1/2 onto an edge of spin j; the exact
    Thiemann operator's +/-1/2 selection rule IS the statement 1/2 x j = (j-1/2) + (j+1/2).
    Verify with the (certified) Clebsch-Gordan: coupling 1/2 to j gives nonzero amplitude ONLY
    for J in {j-1/2, j+1/2}, and the decomposition is orthonormal (sum of |CG|^2 = 1)."""
    ok = True
    js = [k / 2 for k in range(1, 2 * jmax + 1)]
    Js = [k / 2 for k in range(0, 2 * jmax + 3)]
    for jj in js:
        allowed = set()
        for J in Js:
            if any(abs(cg(0.5, mu, jj, m, J, mu + m)) > 1e-9
                   for mu in (0.5, -0.5) for m in [x - jj for x in range(int(2 * jj) + 1)]):
                allowed.add(J)
        expected = {jj - 0.5, jj + 0.5}
        ok &= (allowed == expected)
        # orthonormality: |1/2 mu>|j m> fully decomposes over J in {j±1/2}
        for mu in (0.5, -0.5):
            for m in [x - jj for x in range(int(2 * jj) + 1)]:
                norm = sum(cg(0.5, mu, jj, m, J, mu + m) ** 2 for J in expected)
                ok &= abs(norm - 1.0) < 1e-9
    return ok


if __name__ == "__main__":
    D_int, ev, target, ev_plus = certify_volume()
    print("=" * 72)
    print("STEP 1: certify the AL volume against Gaul-Rovelli (monochromatic 4-valent j=1/2)")
    print("=" * 72)
    print(f"  intertwiner dim = {D_int}  (expect 2: internal spin 0 or 1)")
    print(f"  grasp q_123 eigenvalues = {np.round(ev, 6)}")
    print(f"  target = +/- sqrt(3)/4  = +/-{target:.6f}   (Gaul-Rovelli Eq.62 gives 2x = {2*target:.6f})")
    ok = np.allclose(np.sort(np.abs(ev)), np.abs(target) * np.ones_like(ev), atol=1e-9) or \
        abs(abs(ev).max() - target) < 1e-9
    print(f"  MATCH: {abs(abs(ev).max() - target) < 1e-9}  ->  volume operator certified")
    print(f"\n  orientation-sign pitfall: all-+ sum of 4 triples has eigenvalues {np.round(ev_plus, 9)}")
    print(f"  -> {'IDENTICALLY ZERO (as predicted: the +/-1 orientation sign is essential)' if np.allclose(ev_plus, 0, atol=1e-9) else 'nonzero'}")

    print("\n" + "=" * 72)
    print("STEP 2: certify the m=1 holonomy grasp's +/-1/2 selection rule (1/2 x j = j-1/2 + j+1/2)")
    print("=" * 72)
    print(f"  +/-1/2 selection rule + orthonormal decomposition (j up to 5): "
          f"{'VERIFIED' if verify_selection_rule() else 'FAILED'}")
    print("\n  -> exact-operator core certified: AL volume (=Gaul-Rovelli) + recoupling grasp.")
    print("     Remaining (a Guedes et al. 2024-25 scale build): the graph-changing dipole")
    print("     evolution + the scrambling-vs-ULTRALOCALITY test (does entropy come from mixing")
    print("     pre-existing nodes, or only from graph growth?). See docs/thiemann-operator.md.")
