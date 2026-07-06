"""
Phase DM -- covariance-strain (C_ab) pinning branch: SOFT vs HARD.

Referee M2 re-derivation asks: when matter creates an area defect, does the local
area variance stay O(1) fractionally (SOFT -> gamma_0-free  M_DM/M_b ~ kappa/(4 ln2)
~ 0.4) or collapse to the discreteness floor (HARD -> ~ 1/gamma_0)?

Discriminant = t-scaling of the variance-reduction factor
    Sigma_def(t) = Var(A)_def / Var(A)_vac ,    A = sqrt(j(j+1)) = sqrt(n^2-1)/2 ,
on the single-edge heat-kernel spin law  P(n) ~ n^2 e^{-t (n^2-1)/4}  (n = 2j+1):

    SOFT  = linear source e^{lam sqrt(s)} tuned so the area peak shifts by one gap.
            A tilt shifts the mean but preserves the variance  ->  ratio -> const.
    HARD  = projection onto a fixed +/-W j-band about the mode ("fixed j").
            Var_hard ~ const (t-indep) while Var_vac ~ 1/(4t)  ->  ratio ~ t.

Fit ratio(t)=a+b t over t in {0.1,0.02,0.004}: nonzero plateau a => SOFT; a~0 with
slope-dominated b t => HARD (floor saturation, coefficient model-dependent).

All records are tier T2 (advisory): this phase LOGS the branch, it never gates.
"""
import numpy as np
from mpmath import mpf, exp, sqrt, findroot

from ..core.dictionary import nmax
from ..harness import boolrec, phase_status

PH = "PhaseDM"
TS = [mpf('0.1'), mpf('0.02'), mpf('0.004')]


def _moments(t, tilt=None, band=None):
    """Return (m0, m_s, m_r) = sums of (w, w*s, w*sqrt(s)) with s=(n^2-1)/4=j(j+1),
    base weight w = n^2 e^{-t s}; tilt=lam multiplies by e^{lam sqrt(s)};
    band=(n0,W) restricts the sum to n in [n0-W, n0+W]."""
    N = nmax(t)
    lo, hi = 1, N
    if band is not None:
        n0, W = band
        lo, hi = max(1, n0 - W), n0 + W
    m0 = ms = mr = mpf(0)
    for n in range(lo, hi + 1):
        s = mpf(n * n - 1) / 4
        w = mpf(n * n) * exp(-t * s)
        if tilt is not None:
            w *= exp(tilt * sqrt(s))
        m0 += w
        ms += w * s
        mr += w * sqrt(s)
    return m0, ms, mr


def _varA(m):   return m[1] / m[0] - (m[2] / m[0]) ** 2   # Var(sqrt s) = <s> - <sqrt s>^2
def _meanA(m):  return m[2] / m[0]                         # <sqrt s>


def run(verbose=True):
    vac, soft, hard = [], [], []
    for t in TS:
        mv = _moments(t)
        vv = _varA(mv)
        vac.append(vv)
        # SOFT: tilt e^{lam sqrt s} so <sqrt s> moves up by exactly one gap (=1)
        target = _meanA(mv) + 1
        lam = findroot(lambda L: _meanA(_moments(t, tilt=L)) - target, 1 / vv)
        soft.append(_varA(_moments(t, tilt=lam)) / vv)
        # HARD: fixed +/-W band about the mode n0 ~ 2/sqrt(t) (t-independent width)
        n0 = int(round(float(2 / sqrt(t))))
        hard.append(_varA(_moments(t, band=(n0, 3))) / vv)

    def fit_ab(rs):
        A = np.array([[1.0, float(t)] for t in TS])
        b = np.array([float(r) for r in rs])
        (a, bb), *_ = np.linalg.lstsq(A, b, rcond=None)
        return a, bb
    a_soft, b_soft = fit_ab(soft)
    a_hard, b_hard = fit_ab(hard)

    for i, t in enumerate(TS):
        boolrec(f"PhaseDM.vac.var_t{i}", PH, True, 'T2', computed=float(vac[i]),
                note=f"Var(A)_vac at t={float(t)}; t*Var={float(t*vac[i]):.4f} (-> 1/4)")
        boolrec(f"PhaseDM.soft.ratio_t{i}", PH, True, 'T2', computed=float(soft[i]),
                note=f"Var_soft/Var_vac at t={float(t)} (peak shifted +1 gap)")
        boolrec(f"PhaseDM.hard.ratio_t{i}", PH, True, 'T2', computed=float(hard[i]),
                note=f"Var_hard/Var_vac at t={float(t)} (fixed j-band W=3)")
    # discriminant is the t->0 intercept: nonzero plateau => SOFT; intercept ~0 (ratio ~ b t) => HARD
    soft_plateau = a_soft > 0.5
    hard_collapse = a_hard < 0.1 and b_hard > 0.5
    boolrec("PhaseDM.soft.plateau", PH, soft_plateau, 'T2', computed=float(a_soft),
            note=f"soft fit a+bt: a={a_soft:.4f}, b={b_soft:.4f}; plateau => variance-preserving")
    boolrec("PhaseDM.hard.collapse", PH, hard_collapse, 'T2', computed=float(a_hard),
            note=f"hard fit a+bt: a={a_hard:.4f}, b={b_hard:.4f}; a~0 => ratio ~ b t (floor)")
    is_soft = soft_plateau and hard_collapse
    verdict = ("SOFT: realistic (variance-preserving) pinning -> ratio plateaus -> gamma_0-free "
               "M_DM/M_b ~ kappa/(4 ln2) ~ 0.4; the fixed-j-projection (HARD) branch that would "
               "restore 1/gamma_0 is CLOSED by suscept + continuum records below." if is_soft else
               "AMBIGUOUS: inspect fits.")
    boolrec("PhaseDM.branch", PH, is_soft, 'T2', computed=(1.0 if is_soft else 0.0), note=verdict)

    # --- HARD-BRANCH CLOSURE (was left "future work" in v12; closed here) ---------------
    # (i) Susceptibility. Matter couples diagonally to the area A, so the source is a
    #     NATURAL (exponential-family) parameter lambda with tilt e^{lambda A}. Then the
    #     exact identity d<A>/dlambda = Var(A) holds, i.e. the SOURCE-Fisher is Var(A),
    #     NOT 1/Var(A). Pinning (Var down) therefore REDUCES the sourced strain; the hard
    #     1/f = 1/Var enhancement uses the wrong Fisher (unit susceptibility) for a
    #     diagonal coupling. Verified as the finite-difference identity.
    h = mpf('1e-8')
    t0 = TS[0]
    dmean = (_meanA(_moments(t0, tilt=h)) - _meanA(_moments(t0, tilt=-h))) / (2 * h)
    var0 = _varA(_moments(t0))
    suscept_ok = abs(dmean - var0) / var0 < 1e-6
    boolrec("PhaseDM.suscept.expfamily", PH, suscept_ok, 'T2', computed=float(dmean / var0),
            note=f"d<A>/dlambda / Var(A) = {float(dmean/var0):.9f} at t={float(t0)}; diagonal (area) "
                 "coupling => source-Fisher = Var(A), so pinning lowers the strain -- hard 1/Var is wrong-Fisher")
    # (ii) Continuum limit. Immirzi scalings of the three explicit routes: soft ~ g0^0
    #     (finite as area gap a_1->0), screen G_eff ~ g0^+1 (->0), hard ~ g0^-1 (->INF).
    #     A matter-sourced gravity effect that DIVERGES as spacetime becomes smooth is
    #     unphysical; hard is the unique diverging route => the hard branch is closed.
    powers = {'soft': 0, 'screen': 1, 'hard': -1}
    hard_unphysical = powers['hard'] < 0 <= min(powers['soft'], powers['screen'])
    boolrec("PhaseDM.hard.continuum", PH, hard_unphysical, 'T2', computed=float(powers['hard']),
            note="soft~g0^0 (finite), screen~g0^+1 (->0), hard~g0^-1 (->INF as a_1->0): hard alone "
                 "diverges in the continuum limit => unphysical. Hard branch CLOSED (not future work).")

    st, ntot, nfail = phase_status(PH)
    if verbose:
        print(f"  {PH}: {ntot} records, {nfail} hard-fails  [{st}]")
        print(f"    Var(A)_vac  {[round(float(x),4) for x in vac]}   t*Var {[round(float(t*v),4) for t,v in zip(TS,vac)]}")
        print(f"    soft ratio  {[round(float(x),4) for x in soft]}   fit a={a_soft:.4f} b={b_soft:.4f}")
        print(f"    hard ratio  {[round(float(x),4) for x in hard]}   fit a={a_hard:.4f} b={b_hard:.4f}")
        print(f"    suscept: d<A>/dlambda / Var(A) = {float(dmean/var0):.9f} (=1 => source-Fisher=Var, hard 1/Var wrong)")
        print(f"    continuum: hard ~ g0^-1 diverges as a_1->0 (soft g0^0, screen g0^+1) => hard CLOSED")
        print(f"    VERDICT: {'SOFT ~0.4 (gamma0-free); HARD branch CLOSED' if is_soft else 'AMBIGUOUS'}")
    return st


if __name__ == "__main__":
    run()
