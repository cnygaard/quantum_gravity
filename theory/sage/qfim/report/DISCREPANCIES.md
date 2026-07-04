# DISCREPANCIES.md -- QFIM verification ledger

Design principle: brute-force spin sum is ground truth; closed forms are first-tier
derived truth; the SPEC's hand decimals are anchors only. Where they disagree, the
code wins and the discrepancy is logged here (never force-fit).

## Adjudicated (code decides)

- **Z0(0.1) anchor slip**: SPEC anchor 114.9382; true (sum = closed) 114.93764.
  Relative 4.8e-6 (passes T2). Hand rounding slip -> use 114.9376.
- **|Pf B| anchor slip**: SPEC 691.895; true 691.8938 (= (1/8)sqrt(det G) = closed).
  Relative 1.7e-6. Use 691.8938.
- **D(0.2) adjudication**: exact = 0.000882157; 2-term series p^2/45 - 4p^4/945 = 0.000882116.
  The earlier-quoted 0.0008852 is a slip; series and exact agree to ~4e-8. RESOLVED.
- **connector 8.506 vs 8.509**: the connector coefficient is p_bar/sinh(p_bar) = 0.850918
  at p_bar=1. "8.509" is that value (decimal slip); "8.506" is a finite-beta explicit
  estimate. They differ by the O(beta^2) truncation (measured slope = 2.000). RESOLVED:
  both converge to 0.850918 as beta -> 0.
- **geometric-mean claim is ASYMPTOTIC** (downgrade): sqrt(w_L w_T) t/(2 sqrt2 p^{3/2}) -> 1
  only as p_bar -> infinity (0.99987 at p_bar=200); it is NOT an exact identity.
- **kappa_12 audit**: kappa_12 = 4 j_1 = 20 confirmed; lambda = 1/sqrt(kappa_12 M_1)
  = 1/sqrt(1200) = 0.028868. RESOLVED.
- **superseded anchors**: w_T(1)=18.195368 (supersedes 18.19539); G_perp=25.849679 and the
  transverse eigenvalues 45.536644/6.162715 all reproduced to <1e-28 by the kernel.
- **rho-normalization**: Phase-4 displacement rho's are convention-x2; only the RATIOS
  (rho1/rho3 etc.) are convention-proof.

## Convention notes

- **shear prefactor 16 = 4 x 4**: the closure-projected tilt QFIM yields shear weight
  4 j_a j_b/(j_a+j_b); the SPEC's 16 j_a j_b/(j_a+j_b) absorbs a further convention factor
  of 4 (eta normalization). The reduction factors (8/9,8/9,0.64) are convention-proof.
- **W = 240 vs 420 (conformal)**: the absolute values are compression-sector and
  model-dependent; the convention-proof content Delta W/W = -3/7 is verified exactly.

## SPEC items under-specified (NOT independently reconstructed)

- **R_shape = 1.756718**, **chi_T -> 2/5**, **displacement ratios (rho1/rho3, spread)**:
  no explicit generating formula in the SPEC; not reproduced from the anchors alone.
- **shear-plane triple (23.013,78.304,64.927)**: closest reconstruction w_L(a)+w_T(b)
  (~0.5%); the exact anisotropic-probe construction is not in the SPEC.
- **compression weight (Phase 5)**: not derivable from the SPEC; a placeholder positive
  weight was used for the rank/PSD structural check only.

## Headline findings

- **B_ij on the capstone class = 0** (max 1.5e-30): the Lagrangian property holds; there
  is NO twist-branch anomaly. (S1 proves the Kahler relation structural for arbitrary L.)
- **Capstone PASSES** the falsification test: channel intercepts r_k^inf are O(1e-6),
  three orders below the 0.5% threshold; the master equation ell_P^2 G_ij = q_ij
  reconstructs diag(4,1,1/4) with residuals scaling to zero as t -> 0.

## T2 anchor LOG-FAILs this run (advisory; each a hand-arithmetic slip)

- `P3.D02.exact_vs_series`: computed 0.0008821568192311855 vs anchor 0.0008821164021164021 (rel err 4.58e-05) -- D(0.2) exact vs 2-term series p^2/45-4p^4/945
- `P4.shear.triple_recon`: computed [23.11604193209413, 79.8331626349973, 65.02977456823878] vs anchor [23.013432, 78.304476, 64.927172] (rel err 1.95e-02) -- shear triple ~ w_L(a)+w_T(b) (recon ~0.5%; exact probe not in SPEC -- LEDGER)
- `P5.iso.dV_shear`: computed 0.030618621784789725 vs anchor 0.030619 (rel err 1.24e-05) -- (dV/V)_shear=sqrt6/80
- `P7.conditional.lambda`: computed 0.028867513459481287 vs anchor 0.028868 (rel err 1.69e-05) -- lambda = 1/sqrt(kappa_12 M_1) = 1/sqrt(1200)
