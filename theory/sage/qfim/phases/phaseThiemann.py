"""
Phase Thiemann -- the exact Euclidean Thiemann Hamiltonian constraint: certified core.

The final step of the information-paradox programme is to test whether the EXACT canonical LQG
dynamics (Thiemann's Hamiltonian constraint) scrambles, as the model Hamiltonians of PhaseScramble
do.  This phase logs the verified CORE of that operator, built against Gaul-Rovelli (gr-qc/0011106)
and Borissov-De Pietri-Rovelli (gr-qc/9703090) and benchmarked to Guedes et al. (arXiv:2412.20257,
the 2024-25 numerical implementation).  Full study: theory/thiemann.py, theory/recoupling.py,
docs/thiemann-operator.md.

Records (all T2 advisory):
  volume_gr    : the AL volume grasp on the monochromatic 4-valent j=1/2 node = sqrt(3)/4
                 (Gaul-Rovelli Eq. 62, up to the color factor 2) -- the volume operator is exact
  orient_sign  : the all-+ sum of the four triples is IDENTICALLY ZERO -- the orientation sign
                 eps(e_I,e_J,e_K) is essential (the pitfall, independently confirmed)
  selection    : the m=1 holonomy grasp obeys 1/2 x j = (j-1/2)+(j+1/2) -- the +/-1/2 selection
                 rule + orthonormal decomposition (certified Clebsch-Gordan)

HONEST SCOPE: this certifies the exact operator's CORE (volume + recoupling grasp).  The full
graph-changing dipole evolution + the scrambling test is a Guedes-et-al.-scale build; its expected
outcome (Nicolai-Peeters-Zamaklar ultralocality) is that the exact operator does NOT scramble by
mixing -- which would pinpoint the propagation ingredient the model Hamiltonians supply by hand.
"""
import os
import sys

_THEORY = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if _THEORY not in sys.path:
    sys.path.insert(0, _THEORY)
import thiemann as th                                       # noqa: E402  (theory/thiemann.py)
import dipole as dp                                         # noqa: E402  (theory/dipole.py)
import thiemann_bdr as tb                                   # noqa: E402  (verbatim BDR amplitudes)
import graph_growing as gg                                  # noqa: E402  (graph-growing verdict)

from ..harness import boolrec, phase_status

PH = "PhaseThiemann"


def run(verbose=True):
    import numpy as np
    D_int, ev, target, ev_plus = th.certify_volume()
    vmax = float(np.max(np.abs(ev)))
    boolrec("PhaseThiemann.volume_gr", PH, abs(vmax - target) < 1e-9, 'T2', computed=vmax,
            note=f"4-valent j=1/2 grasp eigenvalue {vmax:.6f} = sqrt(3)/4 (Gaul-Rovelli Eq.62 x1/2); AL volume exact; D_int={D_int}")
    pmax = float(np.max(np.abs(ev_plus)))
    boolrec("PhaseThiemann.orient_sign", PH, pmax < 1e-9, 'T2', computed=pmax,
            note="all-+ 4-triple sum = 0 identically; orientation sign eps(IJK) essential (GR Eq.53 / the pitfall)")
    sel = th.verify_selection_rule()
    boolrec("PhaseThiemann.selection", PH, sel, 'T2', computed=1.0 if sel else 0.0,
            note="m=1 grasp: 1/2 x j = (j-1/2)+(j+1/2), +/-1/2 selection rule + orthonormal (certified CG)")

    # DIPOLE three-arm scrambling test (structural, truncated arena)
    arms, ncfg = dp.summary()
    S_gue, spr_gue = arms['gue']
    S_th, spr_th = arms['thiemann']
    S_mo, spr_mo = arms['model']
    S_fx, spr_fx = arms['fixed']
    boolrec("PhaseThiemann.dipole_arena", PH, spr_gue > 0.5 * ncfg, 'T2', computed=spr_gue,
            note=f"GUE baseline explores {spr_gue:.0f}/{ncfg} configs (arena+measure scramble); dim ok")
    boolrec("PhaseThiemann.dipole_propagation", PH, spr_th > 1.5 * spr_fx, 'T2', computed=spr_th / max(spr_fx, 1e-9),
            note=f"graph-changing arms propagate (Thiemann {spr_th:.1f} cfg, model {spr_mo:.1f}) vs fixed-graph {spr_fx:.1f} (localized)")
    boolrec("PhaseThiemann.dipole_thiemann_vs_model", PH, abs(S_th - S_mo) < 0.5, 'T2', computed=S_th - S_mo,
            note=f"Thiemann S_A={S_th:.2f} ~ model {S_mo:.2f}: volume-gating ~neutral; PROXY arena (moves reach the far vertex) -- contrast grow.ultralocal")

    # VERBATIM BDR + GRAPH-GROWING (the frontier, settled at theta-graph level) -----------
    # (i) BDR verbatim: GR gr-qc/0011106 Eq.(87) general recoupling formula == App.A closed
    #     forms i)-viii) (= corrected Borissov-De Pietri-Rovelli gr-qc/9703090) on the full
    #     admissible grid; volume element certified vs first-principles J-operator grasp.
    worst, namp = tb._check_closed_vs_general(cmax=8)
    boolrec("PhaseThiemann.bdr_verbatim", PH, worst < 1e-12, 'T2', computed=worst,
            note=f"Eq.87 == BDR closed forms over {namp} amplitudes (colors<=8): max diff {worst:.1e}; nets pass (89)-(94)")
    okv, tabv = tb._check_volume_first_principles()
    boolrec("PhaseThiemann.bdr_volume_fp", PH, okv, 'T2', computed=float(len(tabv)),
            note="Eq.62 volume == 2|eig(J-grasp)| at 5 color configs incl. non-monochromatic (binor factor 2 uniform)")
    # (ii) graph-growing verdict: new vertices inert (eps=0 exactly), and the exact dynamics
    #      generates ZERO A:B entanglement for all times (ultralocality CONFIRMED here).
    inert_ok, orig_nonzero = gg.inertness()
    boolrec("PhaseThiemann.grow_inert", PH, inert_ok and orig_nonzero, 'T2',
            computed=1.0 if inert_ok else 0.0,
            note="all grasped triples at degenerate new vertices have eps=0 (exact int dets); generic vertex nonzero (GR Eq.53)")
    Smax, _, obs, lay, NA, NB = gg.joint_evolution()
    boolrec("PhaseThiemann.grow_ultralocal", PH, Smax < 1e-12, 'T2', computed=Smax,
            note=f"max_t S_A:B = {Smax:.1e} on theta-graph towers {NA}x{NB} (verbatim amplitudes): exact H_E is a NON-SCRAMBLER; local dressing churns (<layers> to {max(lay):.2f})")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    AL volume grasp = {vmax:.6f} = sqrt(3)/4 (Gaul-Rovelli); orient-sign pitfall = {pmax:.1e};"
              f" +/-1/2 selection rule {'OK' if sel else 'FAIL'}")
        print(f"    dipole 3-arm (proxy): GUE {S_gue:.2f}/{spr_gue:.0f}cfg, Thiemann {S_th:.2f}/{spr_th:.1f},"
              f" model {S_mo:.2f}/{spr_mo:.1f}, fixed {S_fx:.2f}/{spr_fx:.1f}")
        print(f"    BDR verbatim: {namp} amplitudes max diff {worst:.1e}; volume=first-principles at 5 configs")
        print(f"    GRAPH-GROWING VERDICT: new vertices inert; max_t S_A:B = {Smax:.1e} -> exact H_E is")
        print(f"    ULTRALOCAL (non-scrambler); propagation = the models' hand-supplied ingredient.")
    return st


if __name__ == "__main__":
    run()
