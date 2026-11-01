# DISCREPANCIES.md -- QFIM verification ledger

Design principle: brute-force spin sum is ground truth; closed forms are first-tier
derived truth; the SPEC's hand decimals are anchors only. Where they disagree, the
code wins and the discrepancy is logged here (never force-fit).

## Adjudicated (code decides)

- **Z0(0.1) anchor slip**: SPEC anchor 114.9382; true value **114.937644891588**
  (arbitrary precision, dps=50). The closed form `2 sqrt(pi) e^{t/4} t^{-3/2}`, the exact
  spin-sum `Omega(0)=sum n^2 e^{-t(n^2-1)/4}`, and `dictionary.Z0` all agree to **48 digits**
  (correction is `O(e^{-4 pi^2/t}) ~ e^{-395}`). Evaluating the exact sum itself yields
  114.9376 -- so 114.9382 is a hand-rounding slip (rel 4.8e-6, passes T2), NOT a competing
  value. Use 114.9376.
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

## Phase 4 addendum -- all closed forms now derived (SPEC 4a-4d)

The four previously under-specified quantities are RESOLVED with exact closed forms,
verified to their anchors plus three new symbolic tests (S4a-c) and two structural zeros:

- **R_shape = 1.756718** and **R(Q) = sqrt(Q) L''(Q)/L''(1)** are two slices of one
  functional S[q] = sum_k w_L(p_bar_k)/sqrt(det q); implemented as a single S_of_q routine
  (a dictionary-bug tripwire -- the two agree to 1e-30).
- **shear-plane triple = w_T(p_a) + w_T(p_b)** = (23.013432, 78.304476, 64.927172) exactly
  (supersedes the earlier ~0.5% w_L+w_T reconstruction).
- **displacement rho_i = 2[2 W_tot - w(p_bar_i)]**, additive structure (S4b); the ratios
  rho1/rho3 = 1.552442 and spread = 0.415909 are convention-proof and exact.
- **chi_T(p) = p w'(p)/(5 w(p))**; NEW confirmed claims: chi_T -> 2/5 at BOTH p->0 and
  p->infinity, with an interior dip (min ~0.348 near p~5). chi_T(1) = 0.3863935.
- New symbolic tests: **S4a** Gaussian Wick identity (resid 0), **S4b** rho additive
  collapse under q_a p_bar_a^2 = 1, **S4c** equal-footing L''=w_L/p^2, G^phys_perp=w_T/p^2.
- New structural zeros: off-diagonal gamma_ij = 0 (averaged probe on diagonal q);
  dW_tot/dh = 0 under traceless variation at isotropy.
- Anchor refinements: w_L(2) = 79.3040872; chi_T = 0.3863935; the rho absolutes are
  convention-bound (kappa) and carry ~1e-7 hand-rounding -- only the ratios are exact.

## Still under-specified

- **compression weight (Phase 5)**: not derivable from the SPEC; a placeholder positive
  weight was used for the rank/PSD structural check only.

## Headline findings

- **B_ij on the capstone class = 0** (max 1.5e-30): the Lagrangian property holds; there
  is NO twist-branch anomaly. (S1 proves the Kahler relation structural for arbitrary L.)
- **Capstone PASSES** the internal self-consistency reconstruction: channel intercepts
  r_k^inf are O(1e-6), three orders below the 0.5% threshold; the master equation
  ell_P^2 G_ij = q_ij reconstructs diag(4,1,1/4) against the brute-force ground truth,
  with residuals scaling to zero as t -> 0. (Internal check, not an empirical falsification.)

## T2 anchor LOG-FAILs this run (advisory; each a hand-arithmetic slip)

- `P3.D02.exact_vs_series`: computed 0.0008821568192311855 vs anchor 0.0008821164021164021 (rel err 4.58e-05) -- D(0.2) exact vs 2-term series p^2/45-4p^4/945
- `P5.iso.dV_shear`: computed 0.030618621784789725 vs anchor 0.030619 (rel err 1.24e-05) -- (dV/V)_shear=sqrt6/80
- `P7.conditional.lambda`: computed 0.028867513459481287 vs anchor 0.028868 (rel err 1.69e-05) -- lambda = 1/sqrt(kappa_12 M_1) = 1/sqrt(1200)
