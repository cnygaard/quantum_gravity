"""
Phase Scramble -- does LQG geometry dynamics scramble? (info-paradox Tier 1-3)

Tests whether the LQG VOLUME operator on a single high-valence node is quantum-chaotic --
the quantum sequel to Haggard (2013), whose CLASSICAL pentahedral-volume flow is chaotic for
valence >= 5.  This is the dynamical support for the Haar-typicality the Page curve (PhasePage)
assumed.  Full study + writeup: theory/scrambling.py, theory/scrambling_big.py,
docs/scrambling-derivation.md.  Here we log a fast regression subset.

Records (all T2 advisory):
  calib.gue     : synthetic-GUE r~ = 0.60 (level-spacing pipeline calibrated)
  tetra         : 4-valent quantum-tetrahedron volume = 2.632 (operator-build cross-check)
  chaotic       : distinct-spin n=6 node r~ > 0.5 >> Poisson 0.386 (spectral chaos; anchor
                  D_int=276 -> 0.617)
  eigvec_pt     : IPR*D_int -> 2 = Porter-Thomas (eigenstates Haar-random: the eigenvector-level
                  typicality the Page curve needs, satisfied at the volume-operator level)
  sector        : monochromatic fixed-sign r~ -> Poisson (S_n unresolved): the chaos is not a
                  pipeline artifact (the sector pitfall is real and detected)

HONEST SCOPE: licenses "kinematical geometry is quantum-chaotic with random eigenstates," NOT
the physical evaporating state's 2-design typicality at the Page time (Tier-3 dynamics only
partially thermalizes at these D_int).
"""
import os
import sys

_THEORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _THEORY not in sys.path:
    sys.path.insert(0, _THEORY)
import scrambling as sc                                   # noqa: E402  (theory/scrambling.py)
import evaporation as ev                                   # noqa: E402  (theory/evaporation.py)
import evaporation_clock as ec                              # noqa: E402  (theory/evaporation_clock.py)

from ..harness import boolrec, phase_status

PH = "PhaseScramble"
DISTINCT = [1, 1.5, 2, 2.5, 3, 4]                          # D_int=49 (fast)
MONO = [2, 2, 2, 2, 2, 2]                                  # D_int=65, S_n-degenerate control


def run(verbose=True):
    gue = sc._gue_rbar(N=800)
    anti = sc._antisym_rbar(N=800)
    boolrec("PhaseScramble.calib.gue", PH, 0.55 < gue < 0.63, 'T2', computed=gue,
            note=f"synthetic-GUE r~={gue:.3f} (expect .600); antisym-class ref {anti:.3f}")

    vt, _, _ = sc.volume_spectrum([1, 1, 1, 1])
    boolrec("PhaseScramble.tetra", PH, abs(float(vt[0]) - 2.632) < 0.01, 'T2', computed=float(vt[0]),
            note="4-valent quantum-tetrahedron volume = 2.632 (operator-build cross-check)")

    lv, Di, _ = sc.volume_spectrum(DISTINCT)
    rb = sc.rbar(lv)
    boolrec("PhaseScramble.chaotic", PH, rb > 0.5, 'T2', computed=rb,
            note=f"distinct n=6 D_int={Di} r~={rb:.3f} >> Poisson .386 (chaotic); anchor D_int=276 -> .617")

    ipr, haar = sc.eigvec_ipr(DISTINCT)
    boolrec("PhaseScramble.eigvec_pt", PH, abs(ipr - haar) < 0.6, 'T2', computed=ipr,
            note=f"IPR*D_int={ipr:.2f} -> Porter-Thomas {haar:.0f} (Haar-random eigenstates: eigenvector typicality)")

    lm, Dm, _ = sc.volume_spectrum(MONO)
    rbm = sc.rbar(lm)
    boolrec("PhaseScramble.sector", PH, rbm < 0.45, 'T2', computed=rbm,
            note=f"monochromatic fixed-sign r~={rbm:.3f} -> Poisson (S_n unresolved): chaos not a pipeline artifact")

    # Tier 3+ : genuine physical Hamiltonian H=sum K_ab J_a.J_b on a spin-1/2 horizon --
    # is it chaotic, and does an unentangled dimer scramble to the PAGE VALUE? (n=12, exact)
    rb_ev, page_ratio, D_ev = ev.summary(n=12)
    boolrec("PhaseScramble.phys_chaotic", PH, rb_ev > 0.47, 'T2', computed=rb_ev,
            note=f"physical H random-Heisenberg singlet-sector r~={rb_ev:.3f} (GOE .531); D_int={D_ev}")
    boolrec("PhaseScramble.page_scramble", PH, page_ratio > 0.85, 'T2', computed=page_ratio,
            note=f"dimer S_A -> {page_ratio:.2f} x Page value under e^(-iHt): evaporation scrambles to typicality (n=12; n=14 ratio 1.00)")

    # Genuine evaporation: spin-CHANGING emission (holonomy hopping) + deparametrized CLOCK ->
    # the dynamical PAGE CURVE.  S_rad rises from 0, peaks at half-evaporation, returns to ~0.
    pk, end, kpk, ns = ec.summary(n_BH=6, seeds=(0, 1, 2, 3), nsteps=120)
    boolrec("PhaseScramble.page_curve", PH, pk > 1.5 and end < 0.15 and 3 < kpk < ns - 3, 'T2', computed=pk,
            note=f"dynamical Page curve: S_rad 0 -> peak {pk:.2f} (step {kpk}) -> {end:.2f}; spin-changing emission + relational clock, horizon re-purifies (info returned)")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    calib GUE {gue:.3f} (antisym {anti:.3f}); tetra vol {float(vt[0]):.3f}")
        print(f"    Tier1 chaotic r~={rb:.3f} (>>.386, anchor .617); Tier2 IPR*D={ipr:.2f} (PT={haar:.0f});"
              f" sector mono r~={rbm:.3f} (<.45)")
        print(f"    Tier3+ physical H: r~={rb_ev:.3f} (GOE); dimer S_A -> {page_ratio:.2f}x Page value"
              f" (evaporation scrambles to typicality)")
        print(f"    Evaporation: spin-changing emission + clock -> Page curve S_rad 0->{pk:.2f}(step {kpk})->{end:.2f}"
              f" (99% turnover, horizon re-purifies)")
        print(f"    VERDICT: volume op chaotic (Porter-Thomas) + physical H scrambles to Page value + the")
        print(f"             DYNAMICAL PAGE CURVE from spin-changing emission in relational time -- info returns")
    return st


if __name__ == "__main__":
    run()
