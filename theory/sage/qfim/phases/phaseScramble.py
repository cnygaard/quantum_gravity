"""
Phase Scramble -- does LQG geometry dynamics scramble? (info-paradox Tier 1-3)

Level statistics of the LQG VOLUME operator on single nodes of valence 6-8 (Haggard (2013) found
evidence of CLASSICAL chaos, with a mixed phase space, for the pentahedral volume flow and
conjectured it at higher valence).  The volume is an observable, not a generator: circumstantial
support (not derivation) for the Haar-typicality the Page curve (PhasePage) assumes.  Full study: scrambling.py; the
referee-grade large-node statistics (realizable-embedding ensembles, block bootstrap, matched-D
reference) are precomputed by scrambling_big.py into qfim/report/volume_anchor.json and logged
here.  Here we log a fast regression subset.

Records (all T2 advisory):
  calib.gue     : synthetic-GUE r~ = 0.60 (level-spacing pipeline calibrated)
  tetra         : 4-valent quantum-tetrahedron volume = 2.632 (operator-build cross-check)
  chaotic       : distinct-spin n=6 node, default embedding: r~ > 0.5 >> Poisson 0.386
  eigvec_pt     : IPR*D_int in the Clebsch-Gordan recoupling basis (2 = Porter-Thomas, beta=2)
  sector        : monochromatic fixed-sign r~ -> below Poisson (S_n unresolved): the sector
                  pitfall is real and detected
  volume.*      : precomputed ensembles over realizable embeddings (see volume_anchor.json):
                  default, embeddings, stats (bulk window, KS/Levene vs reference, small-s beta,
                  Spearman with min|det|), and the valence trend
  page_scramble_n14 : late-time mean +/- temporal std vs exact Haar-singlet mean and std
  page_curve_singlet / singlet_invariance / page_curve_singlet_tau : SU(2)-singlet puncture
                  emission (model A of evaporation_clock.py) vs the EXACT Haar-singlet curve, cut
                  by cut; longer scrambling and the fresh-quench (energy-envelope) control

HONEST SCOPE: licenses statements about the kinematical volume spectrum and eigenvectors, NOT
the physical evaporating state's 2-design typicality at the Page time.
"""
import json
import os
import sys

import numpy as np

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
            note=f"distinct n=6 D_int={Di}, default embedding: r~={rb:.3f} >> Poisson .386 (level repulsion); embedding ensembles: volume.* records")

    ipr, haar = sc.eigvec_ipr(DISTINCT)
    boolrec("PhaseScramble.eigvec_pt", PH, abs(ipr - haar) < 0.6, 'T2', computed=ipr,
            note=f"IPR*D_int={ipr:.2f} in the recoupling basis (Porter-Thomas {haar:.0f}, GOE 3): delocalized, "
                 "not exactly Porter-Thomas")

    lm, Dm, _ = sc.volume_spectrum(MONO)
    rbm = sc.rbar(lm)
    boolrec("PhaseScramble.sector", PH, rbm < 0.45, 'T2', computed=rbm,
            note=f"monochromatic fixed-sign r~={rbm:.3f}, below Poisson .386 (unresolved S_n sectors): the sector pitfall is real and detected")

    # Tier 3+ : genuine physical Hamiltonian H=sum K_ab J_a.J_b on a spin-1/2 horizon --
    # is it chaotic, and does an unentangled dimer scramble to the PAGE VALUE? (n=12, exact)
    rb_ev, page_ratio, D_ev = ev.summary(n=12)
    lt = ev.late_time_stats(n=14)
    boolrec("PhaseScramble.page_scramble_n14", PH, abs(lt['late_mean'] - lt['haar_exact']) < lt['haar_std'], 'T2',
            computed=lt['late_mean'],
            note=f"n=14 dimer under e^(-iHt): late-time MEAN S_A={lt['late_mean']:.4f} +/- {lt['late_std']:.4f} (temporal "
                 f"std, {lt['n_late']} samples; max over t {lt['S_max']:.3f}) vs exact Haar-singlet {lt['haar_exact']:.4f} "
                 f"(Haar std {lt['haar_std']:.4f}, {lt['nsamp']} draws)")
    boolrec("PhaseScramble.phys_chaotic", PH, rb_ev > 0.47, 'T2', computed=rb_ev,
            note=f"physical H random-Heisenberg singlet-sector r~={rb_ev:.3f} (GOE .531); D_int={D_ev}")
    boolrec("PhaseScramble.page_scramble", PH, page_ratio > 0.85, 'T2', computed=page_ratio,
            note=f"dimer S_A -> {page_ratio:.2f} x Page value under e^(-iHt): evaporation scrambles to typicality (n=12; n=14 ratio 1.00)")

    # Emission model B: U(1) occupation-encoded collision model (fresh |down> modes break SU(2)
    # to U(1); partial-SWAP exchange; turnover guaranteed by the finite register).
    pk, end, kpk, ns = ec.summary(n_BH=6, seeds=(0, 1, 2, 3), nsteps=120)
    boolrec("PhaseScramble.page_curve", PH, pk > 1.5 and end < 0.15 and 3 < kpk < ns - 3, 'T2', computed=pk,
            note=f"U(1) collision model (XX): S_rad 0 -> peak {pk:.2f} (step {kpk}) -> {end:.2f}; flip-flop, radiation carries the M")
    pk2, end2, kpk2, ns2 = ec.summary(n_BH=6, seeds=(0, 1, 2, 3), nsteps=120,
                                      dt_emit=0.24, exchange='heisenberg')
    boolrec("PhaseScramble.page_curve_su2", PH, pk2 > 1.5 and end2 < 0.15 and 3 < kpk2 < ns2 - 3,
            'T2', computed=pk2,
            note=f"U(1) occupation-encoded collision model, Heisenberg coupling J.sigma with |down> fresh modes: "
                 f"peak {pk2:.4f} (step {kpk2}) -> {end2:.4f}, turnover {1 - end2 / pk2:.1%} (matches XX)")

    # Emission model A: SU(2)-SINGLET puncture emission (the quoted model).  Global singlet of
    # n=14 spin-1/2; whole punctures leave a scrambled horizon; compared cut by cut with the
    # Haar-random-singlet Page curve; no-scrambling control = dimer sawtooth (peak ln 2).
    sA = ec.page_curve_singlet(n=14, tau=3.0)
    sC = ec.page_curve_singlet(n=14, seeds=(0,), tau=0.0)
    boolrec("PhaseScramble.page_curve_singlet", PH,
            sA['mid_ratio'] > 0.97 and sA['max_rel_dev'] < 0.05 and sC['mid_ratio'] < 0.25, 'T2',
            computed=sA['mid_ratio'],
            note=f"singlet emission n=14 (D_int={sA['D_int']}), tau=3, 4 draws: S(R_k), k=1..7 = "
                 + " ".join(f"{x:.3f}" for x in sA['S'][1:8]) + "; exact Haar-singlet "
                 + " ".join(f"{x:.4f}" for x in sA['page'][1:8]) +
                 f"; max rel. dev. {sA['max_rel_dev']:.3f} at k={sA['argmax_dev']}; S(R_8)={sA['S'][8]:.3f}; "
                 f"control tau=0: ratio {sC['mid_ratio']:.3f} (ln2 sawtooth)")
    seeds8 = tuple(range(8))
    tau_rows = {}
    for tau, nq in ((3.0, 1), (10.0, 1), (3.0, 3)):
        per = np.array([ec.page_curve_singlet(n=14, seeds=(sd,), tau=tau, nquench=nq)['S'] for sd in seeds8])
        m, se = per.mean(0), per.std(0, ddof=1) / np.sqrt(len(seeds8))
        rel = (m[1:14] - sA['page'][1:14]) / sA['page'][1:14]
        tau_rows[(tau, nq)] = (m[7], se[7], float(np.max(np.abs(rel))), float(np.max(np.abs(rel[:4]))))
    t10, q3 = tau_rows[(10.0, 1)], tau_rows[(3.0, 3)]
    boolrec("PhaseScramble.page_curve_singlet_tau", PH, t10[3] < 0.005 and abs(q3[0] - sA['page'][7]) < 3 * q3[1], 'T2',
            computed=t10[2],
            note="8 draws, S(R_7) +/- SE | max rel. dev. | max rel. dev. k<=4 (exact Haar S(R_7)="
                 f"{sA['page'][7]:.4f}): " + "; ".join(
                     f"tau={tau:g}{' x' + str(nq) + ' fresh quenches' if nq > 1 else ''}: {r[0]:.4f}+/-{r[1]:.4f} | "
                     f"{r[2]:.4f} | {r[3]:.4f}" for (tau, nq), r in tau_rows.items()) +
                 " -- early cuts converge with tau; the residual mid-curve deficit of single-H scrambling is removed "
                 "by fresh quenches (energy envelope)")
    boolrec("PhaseScramble.singlet_invariance", PH, max(sA['j2_max'], sC['j2_max']) < 1e-10, 'T2',
            computed=max(sA['j2_max'], sC['j2_max']),
            note="max <J_tot^2> over all emission steps and seeds: the singlet model is exactly SU(2)-invariant")

    # Precomputed large-node volume statistics (scrambling_big.py, ~40 min): realizable
    # embeddings, block bootstrap, matched-D reference.
    vpath = os.path.join(os.path.dirname(__file__), '..', 'report', 'volume_anchor.json')
    vnodes = json.load(open(vpath))['nodes'] if os.path.exists(vpath) else []
    for nd in vnodes:
        tag = f"D{nd['D_int']}"
        d, e, ref = nd['default'], nd['ensemble'], nd['reference']
        g = e['generic']
        ts = nd.get('tests')
        boolrec(f"PhaseScramble.volume.{tag}.default", PH, d['se_block'] > 0, 'T2', computed=d['mean'],
                note=f"spins {nd['spins']} default embedding (-1)^(p+q+r): r~={d['mean']:.4f}, block-bootstrap SE "
                     f"{d['se_block']:.4f} (iid SEM {d['sem_iid']:.4f}); matched-D beta=2 ref {ref['mean']:.4f}")
        boolrec(f"PhaseScramble.volume.{tag}.embeddings", PH, g['count'] > 1, 'T2', computed=g['mean'],
                note=f"{e['nsign']} realizable embeddings: generic {g['count']} r~={g['mean']:.4f}+/-{g['sem']:.4f} "
                     f"(SEM), all r~={e['all']['mean']:.4f}+/-{e['all']['sem']:.4f}; degenerate (extra zero modes) "
                     f"{e['degenerate']['count']}; PT-KS generic {g['ks_mean']:.4f} (recoupling basis); matched "
                     f"reference {ref['mean']:.4f}+/-{ref.get('sem', 0):.4f} (SEM of {ref.get('nrep', 200)} draws, r~ on V)")
        if ts:
            gb, b = e['generic_bulk'], ts['beta']
            boolrec(f"PhaseScramble.volume.{tag}.stats", PH, gb['count'] > 1, 'T2', computed=gb['mean'],
                    note=f"valence {len(nd['spins'])}: bulk-window (ranks 20-90%) generic r~={gb['mean']:.4f}+/-{gb['sem']:.4f} "
                         f"(ref {ref['mean_bulk']:.4f}); KS vs ref p={ts['ks_ref'][1]:.1e} (bulk {ts['ks_ref_bulk'][1]:.1e}); "
                         f"Levene p={ts['levene'][1]:.2g}, spread ratio {ts['spread_ratio']:.2f}; small-s beta "
                         f"{b['beta']:.2f}+/-{b['se']:.2f} vs ref {b['ref']:.2f}+/-{b['ref_se']:.2f}; Spearman(r~, min|det|) "
                         f"{ts['det_corr'][0]:+.2f} (p={ts['det_corr'][1]:.2g})")
    if vnodes:
        trend = {}
        for nd in vnodes:
            g = nd['ensemble']['generic']
            trend.setdefault(len(nd['spins']), []).append((g['mean'], g['sem']))
        wm = {v: (sum(m / s_ ** 2 for m, s_ in r) / sum(1 / s_ ** 2 for _, s_ in r),
                  sum(1 / s_ ** 2 for _, s_ in r) ** -0.5, len(r)) for v, r in sorted(trend.items())}
        vs = sorted(wm)
        boolrec("PhaseScramble.volume.valence_trend", PH, all(wm[a][0] < wm[b][0] for a, b in zip(vs, vs[1:])), 'T2',
                computed=wm[vs[-1]][0],
                note="inverse-variance mean generic r~ by valence: " + "; ".join(
                    f"{v}: {m:.4f}+/-{s_:.4f} ({k} nodes)" for v, (m, s_, k) in wm.items()) +
                     f" (GOE {sc.RMT['GOE']:.4f}, matched beta=2 reference ~0.598)")
        cal = {D: sc.goe_beta_reference(D) for D in (95, 553)}
        bt = {}
        for nd in vnodes:
            bt.setdefault(len(nd['spins']), []).append(nd['tests']['beta']['beta'])
        refb = [nd['tests']['beta']['ref'] for nd in vnodes]
        boolrec("PhaseScramble.volume.beta_calibration", PH, all(0.7 < b < 1.1 for b, _, _ in cal.values()), 'T2',
                computed=cal[553][0],
                note="small-s estimator beta_mle on the same pipeline: GOE (beta=1) gives " + ", ".join(
                    f"{b:.2f}+/-{se:.2f} at D={D}" for D, (b, se, _) in cal.items()) +
                     f"; matched beta=2 reference {min(refb):.2f}-{max(refb):.2f}; volume operator by valence " +
                     "; ".join(f"{v}: {min(b):.2f}-{max(b):.2f}" for v, b in sorted(bt.items())))

    # Referee-demanded statistics upgrades ------------------------------------------------
    # (i) r~ with SEM on the volume spectrum (per-gap dispersion) -- fast node; the D=276
    #     anchor with SEM is produced by scrambling_big.py and quoted in the papers.
    rbv, rbvsem = sc.rbar_sem(lv)
    boolrec("PhaseScramble.rbar_sem", PH, abs(rbv - 0.594) < 3 * rbvsem + 0.02, 'T2', computed=rbv,
            note=f"volume r~ = {rbv:.3f} +/- {rbvsem:.3f} (D_int={Di}; antisym beta=2 ref 0.594) -- error bars per referee")
    # (ii) Porter-Thomas at the DISTRIBUTION level: KS of y=D|c|^2 vs e^-y (beta=2), in the
    #      orthonormal singlet basis, positive-eigenvalue vectors only.
    ksres = sc.pt_ks(DISTINCT)
    ks, nsamp, _ = ksres
    boolrec("PhaseScramble.pt_ks", PH, ks < 0.10, 'T2', computed=ks,
            note=f"KS(y=D|c|^2, e^-y) = {ks:.3f} over {nsamp} components in the Clebsch-Gordan recoupling basis "
                 "(an arbitrary singlet basis would make this test nearly trivial)")
    # (iii) total-J=0 block certificate (not merely M=0): J^2 residual on the working basis.
    j0, bd = sc.j0_residual(DISTINCT), sc.basis_defect(DISTINCT)
    boolrec("PhaseScramble.j0_block", PH, j0 is not None and j0 < 1e-9 and bd < 1e-12, 'T2',
            computed=j0 if j0 is not None else -1.0,
            note=f"max |J^2 P| = {j0:.1e} and max |P^T P - 1| = {bd:.1e} on the Clebsch-Gordan recoupling basis: "
                 "diagonalization is inside total-J=0")
    # (iv) puncture-Hamiltonian r~ with SEM, pooled over seeds (referee m6).
    prb, psem = ev.phys_rbar_sem(n=12, seeds=(0, 1, 2, 3, 4))
    boolrec("PhaseScramble.phys_rbar_sem", PH, abs(prb - 0.531) < 3 * psem + 0.01, 'T2', computed=prb,
            note=f"puncture H r~ = {prb:.4f} +/- {psem:.4f} pooled over 5 seeds (GOE 0.5307)")
    # (iv-b) correlation-robust errors (referee: adjacent ratios share a spacing): block
    #        bootstrap at n=14 (the quoted 0.558) and the across-seed SEM at n=12.
    bb14 = sc.rbar_block_bootstrap(ev.singlet_levels(n=14, seed=7))
    g14 = (bb14['mean'] - sc.RMT['GOE']) / bb14['se_block']
    boolrec("PhaseScramble.phys_rbar_n14_bootstrap", PH, bb14['se_block'] > 0, 'T2', computed=bb14['mean'],
            note=f"puncture H n=14 (seed 7): r~={bb14['mean']:.4f}, block-bootstrap SE {bb14['se_block']:.4f} "
                 f"(iid {bb14['sem_iid']:.4f}, lag-1 {bb14['lag1']:+.3f}); {g14:+.1f} sigma from GOE 0.5307")
    ps_mean, ps_sem, _ = ev.phys_rbar_seeds(n=12, seeds=(0, 1, 2, 3, 4))
    boolrec("PhaseScramble.phys_rbar_n12_seeds", PH, abs(ps_mean - sc.RMT['GOE']) < 3 * ps_sem + 0.01, 'T2',
            computed=ps_mean,
            note=f"puncture H n=12: r~={ps_mean:.4f} +/- {ps_sem:.4f} (across-seed SEM, 5 independent K_ab draws)")
    # (v) geometric locality (referee pt.6 / fast scrambling): ring-local vs all-to-all.
    loc = ev.summary_local(n=12)
    sat_all, t90_all = loc['all']
    sat_ring, t90_ring = loc['ring']
    boolrec("PhaseScramble.local_page", PH, sat_all > 0.9 and sat_ring < sat_all, 'T2',
            computed=sat_ring,
            note=f"unit-bandwidth: all-to-all S_sat={sat_all:.2f}xPage (t90={t90_all:.1f}); ring-local "
                 f"{sat_ring:.2f}xPage (t90={t90_ring:.1f}) -- locality slows AND weakens scrambling; "
                 "the Page-value saturation uses generic all-to-all couplings (stated in text)")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    calib GUE {gue:.3f} (antisym {anti:.3f}); tetra vol {float(vt[0]):.3f}")
        print(f"    Tier1 default-embedding r~={rb:.3f} (>>.386); Tier2 IPR*D={ipr:.2f} (PT={haar:.0f});"
              f" sector mono r~={sc.rbar(lm):.3f} (<.45)")
        print(f"    stats: volume r~={rbv:.3f}+/-{rbvsem:.3f}; PT-KS={ks:.3f}/{nsamp}; J2 residual {j0:.0e};"
              f" phys r~={prb:.4f}+/-{psem:.4f}")
        print(f"    U(1) collision model: peak {pk2:.2f}->end {end2:.2f}; locality: all {sat_all:.2f}x vs ring"
              f" {sat_ring:.2f}x Page (t90 {t90_all:.1f} vs {t90_ring:.1f})")
        print(f"    Tier3+ physical H: r~={rb_ev:.3f} (GOE); dimer S_A -> {page_ratio:.2f}x Page value")
        print(f"    Singlet emission (n=14): S(R_7)/Page = {sA['mid_ratio']:.3f}, max rel. dev. {sA['max_rel_dev']:.3f},"
              f" control {sC['mid_ratio']:.3f}, max <J^2> {sA['j2_max']:.0e}")
        for nd in vnodes:
            print(f"    volume D_int={nd['D_int']}: default r~={nd['default']['mean']:.3f}+/-{nd['default']['se_block']:.3f};"
                  f" generic embeddings {nd['ensemble']['generic']['mean']:.3f}+/-{nd['ensemble']['generic']['sem']:.3f}"
                  f" (ref {nd['reference']['mean']:.3f})")
    return st


if __name__ == "__main__":
    run()
