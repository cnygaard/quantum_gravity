"""
qfim/run_all.py -- run every phase in the strict sequence, honor the Phase-0 gate,
then emit results.json, DISCREPANCIES.md and the capstone residual plot.
"""
import os
from .phases import (phase0, phase1, phase2, phase3, phase4, phase5, phase6,
                     phase7, capstone, proofs)
from .harness import RECORDS, summarize, dump, HARD_TIERS
from .report import write_discrepancies, write_capstone_plot, t2_failures

# Phase 0 gates all; 1->2->3->4 in order; 5-7 parallelizable; capstone last.
PHASES = [phase0, phase1, phase2, phase3, phase4, phase5, phase6, phase7, proofs, capstone]
REPORT = os.path.join(os.path.dirname(__file__), 'report', 'results.json')


def main():
    print("=" * 70)
    for mod in PHASES:
        st = mod.run()
        if st == 'GATE-FAIL' and mod is phase0:
            print("\n!!! Phase 0 GATE FAILURE -- stopping (Phase 0 gates all).")
            break
    print("=" * 70)
    print("SUMMARY (hard gates: T0/T1/Z/NUM/Znum; T2 advisory):")
    print(summarize())

    dump(REPORT)
    dpath = write_discrepancies()
    ppath = write_capstone_plot()

    # highlight reel
    hard_fail = [r for r in RECORDS if r['tier'] in HARD_TIERS and r['status'] == 'FAIL']
    fails = t2_failures()
    caps = [r for r in RECORDS if r['phase'] == 'Capstone' and r['id'].startswith('Cap.intercept')]
    print("\n" + "=" * 70)
    print(f"TOTAL {len(RECORDS)} records | hard-gate failures: {len(hard_fail)} | T2 anchor LOG-FAILs: {len(fails)}")
    if hard_fail:
        for r in hard_fail:
            print(f"   HARD FAIL: {r['id']} ({r['tier']}) err={r['error']}")
    print("\nT2 anchor LOG-FAILs (each a hand-arithmetic slip to fix in the write-up):")
    for r in fails:
        print(f"   {r['id']}: rel err {r['error']:.2e} -- {r['note']}")
    print("\nCapstone channel intercepts r_k^inf (PASS iff |.| < 0.5%):")
    for r in caps:
        print(f"   {r['id']}: r_inf = {r['computed']:.2e}")
    bij = [r for r in RECORDS if r['id'] == 'Cap.Bij.on_class']
    if bij:
        print(f"\nB_ij on the class: max = {bij[0]['computed']:.2e}  (0 => Lagrangian holds, no twist-branch)")
    print(f"\nwrote:\n  {REPORT}\n  {dpath}\n  {ppath}")


if __name__ == "__main__":
    main()
