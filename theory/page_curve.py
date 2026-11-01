"""
Page-curve bookkeeping for an LQG horizon (finite capacity, conditional on typicality).
=====================================================================================

Setting:  a horizon of area A is pierced by N punctures; puncture p carries area
  a_p = 8 pi gamma0 lP^2 sqrt(j_p(j_p+1)) and a FINITE multiplicity d_p = 2 j_p + 1, so the
  horizon Hilbert space is finite-dimensional.  Finiteness gives the ceiling S_rad <= ln m and
  the exact return S_rad(end) = 0; the Page CURVE additionally needs typicality of the global
  state (Page's theorem), which is an assumption, not a consequence of discreteness.

CAPACITY vs AREA:  ln d_p is not proportional to a_p across mixed spins.  The dictionary
  ln D = A/4 holds either (i) for monochromatic punctures at the matched gamma0
  (ln(2j+1) = 2 pi gamma0 sqrt(j(j+1)); j=1/2 -> ABCK 0.1273), one fixed configuration, or
  (ii) for the DIRECT SUM over spin configurations, whose Ghosh-Perez ensemble (PRL 107, 241301)
  has S = A/4 + N sigma(gamma) and sigma(gamma*) = 0 at gamma* = 0.274067.  Route (ii) needs the
  direct sum: one fixed configuration carries only <ln d> (42% of A/4 at gamma*), the rest is
  the label entropy (gp_split).

Model:  evaporation emits punctures one at a time and the global state stays pure.
  * page_curve: fixed-configuration registers, d_R = prod_emitted d_p, d_B = prod_rest d_p, and
    Page's average S = psi(mn+1) - psi(n+1) - (m-1)/(2n) ~ ln m - m/(2n) (unconstrained).
  * page_centre: the exact average with a CENTRE (Bianchi-Dona), for the SU(2) singlet
    constraint (centre J_R) and for the area split of the direct sum (centre n1_R).  The height
    is ln m - 1/2 - O(1) for SU(2) (centre_deficit -> 0.289) and acquires -alpha sqrt(N) from the
    abelian area split (centre_area_split, area_split_alpha); the closure log does not enter it.
  * gp_page_order: the Page point is at half the AREA for exchangeable or composition-preserving
    emission (law of large numbers) and moves for spin-selective orders.

Pure Python + mpmath (arbitrary precision for the digamma corrections).
"""
import numpy as np
from mpmath import mp, mpf, log, sqrt, pi

mp.dps = 40


def area_quantum(j, gamma0):
    """LQG area eigenvalue per puncture, in units of lP^2:  8 pi gamma0 sqrt(j(j+1))."""
    return 8 * pi * gamma0 * sqrt(mpf(j) * (j + 1))


def page_entropy(ln_m, ln_n):
    """Average subsystem entropy of a Haar-random pure state, dim(A)=m<=n=dim(B),
    given the LOGS ln_m<=ln_n (dims are astronomically large, so work in log space).
    S = ln m - m/(2n) + O((m/n)^2) is the large-dim limit of the exact Page formula;
    the correction m/(2n)=exp(ln_m-ln_n)/2 is exp-small except near the Page time."""
    return ln_m - mp.e ** (ln_m - ln_n) / 2


def page_curve(spins, gamma0, order='sorted', seed=0):
    """Return (frac_area_evaporated, S_rad, S_BH_remaining) as punctures are emitted.
    order='sorted': largest-area-first (legacy; deterministic).  order='random': thermal
    (physical) emission order -- REQUIRED for the mixed-spin area-fraction statement, since
    sorted order biases emitted area ahead of emitted capacity.  Entropies in nats."""
    if order == 'random':
        rng = np.random.default_rng(seed)
        spins = [spins[i] for i in rng.permutation(len(spins))]
    elif order != 'given':                                              # 'given': as passed
        spins = sorted(spins, key=lambda j: -float(sqrt(mpf(j) * (j + 1))))  # emit big first
    lnd = [log(mpf(2 * j + 1)) for j in spins]                          # ln d_p
    area = [area_quantum(j, gamma0) for j in spins]
    N = len(spins)
    lnD_tot = sum(lnd)            # ln of the fixed-configuration microstate dim
    A_tot = sum(area)
    out = []
    lnR, A_evap = mpf(0), mpf(0)
    for k in range(N + 1):
        if k:
            lnR += lnd[k - 1]     # ln dim of radiation (k emitted)
            A_evap += area[k - 1]
        lnB = lnD_tot - lnR       # ln dim of remaining BH
        m, n = (lnR, lnB) if lnR <= lnB else (lnB, lnR)
        S_rad = page_entropy(m, n) if k not in (0, N) else mpf(0)
        out.append((float(A_evap / A_tot), float(S_rad), float(lnB)))
    return out, float(lnD_tot), float(A_tot)


# ---------------------------------------------------------------- Ghosh-Perez ensemble
def gp_weights(gamma, jmax=120):
    """GP equilibrium spin weights at Unruh temperature: w_j = (2j+1) e^{-2 pi gamma sqrt(j(j+1))}
    for j = 1/2, 1, 3/2, ...  (Ghosh-Perez PRL 107, 241301, Eqs. 12-15)."""
    js = [mpf(k) / 2 for k in range(1, 2 * jmax + 1)]
    ws = [(2 * j + 1) * mp.e ** (-2 * pi * gamma * sqrt(j * (j + 1))) for j in js]
    Z = sum(ws)
    return js, [w / Z for w in ws], Z


def gp_sigma(gamma, jmax=120):
    """sigma(gamma) = ln sum_j (2j+1) e^{-2 pi gamma sqrt(j(j+1))}  (GP chemical potential
    is mu = -T_U sigma).  sigma(gamma*)=0 recovers the old counting condition."""
    _, _, Z = gp_weights(gamma, jmax)
    return log(Z)


def gp_area_law(gamma):
    """Verify the GP identity S_config/N = A/(4 lP^2 N) + sigma(gamma) exactly:
    per puncture, <ln((2j+1)/p_j)> = 2 pi gamma <sqrt(j(j+1))> + sigma(gamma), where
    A/(4N) = 2 pi gamma <sqrt(j(j+1))> lP^2.  Returns the identity residual (=0 to fp)
    and the universal area coefficient 1/4 it certifies."""
    js, ps, Z = gp_weights(gamma)
    S_per = sum(p * log((2 * j + 1) / p) for j, p in zip(js, ps))
    area_per_over4 = 2 * pi * gamma * sum(p * sqrt(j * (j + 1)) for j, p in zip(js, ps))
    return float(S_per - area_per_over4 - log(Z)), float(S_per), float(area_per_over4)


def gp_gamma_star(lo=0.1, hi=0.5):
    """Solve sigma(gamma)=0: the zero-chemical-potential point (= the old counting value)."""
    from mpmath import findroot
    return float(findroot(lambda g: gp_sigma(g), mpf('0.27')))


def gp_page(gamma, N=400, seed=0):
    """Page curve with GP-sampled spins at Immirzi gamma: returns (area-frac at peak,
    capacity-frac at peak, peak S_rad / (S_tot/2)).  By the law of large numbers the
    emitted-area and emitted-capacity fractions coincide as N->inf, so the peak sits at
    half the area up to O(N^-1/2) for EVERY gamma."""
    js, ps, _ = gp_weights(gamma)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(js), size=N, p=np.array([float(p) for p in ps]) /
                     sum(float(p) for p in ps))
    spins = [js[i] for i in idx]
    # thermal (random) emission order: the physical choice, and the one for which the
    # emitted-area and emitted-capacity fractions coincide by the LLN
    curve, S0, A0 = page_curve(spins, mpf(str(gamma)), order='random', seed=seed + 7)
    rng2 = np.random.default_rng(seed + 7)
    spins_ord = [spins[i] for i in rng2.permutation(N)]      # page_curve's internal order
    caps = np.cumsum([float(log(mpf(2 * j + 1))) for j in spins_ord])
    caps = caps / caps[-1]
    k_pk = int(np.argmax([s for _, s, _ in curve]))
    cap_frac = float(caps[k_pk - 1]) if k_pk > 0 else 0.0
    return curve[k_pk][0], cap_frac, curve[k_pk][1] / (S0 / 2)


def gp_split(gamma):
    """Per-puncture split of the Ghosh-Perez entropy at Immirzi gamma:
    <a>/(4 lP^2) = <ln(2j+1)> + H(label) - sigma(gamma).  Returns (<a>/4, <ln d>, H, <ln d>/(<a>/4)).
    <ln d> is the capacity of ONE fixed spin configuration; the label entropy H is carried only by
    the direct sum over configurations (referee: at gamma* the fixed configuration holds ~42% of A/4)."""
    js, ps, _ = gp_weights(gamma)
    a4 = 2 * pi * gamma * sum(p * sqrt(j * (j + 1)) for j, p in zip(js, ps))
    lnd = sum(p * log(2 * j + 1) for j, p in zip(js, ps))
    H = -sum(p * log(p) for p in ps if p > 0)
    return float(a4), float(lnd), float(H), float(lnd / a4)


def gp_order(spins, order, seed=0):
    """Emission order of a sampled horizon.  'random': exchangeable; 'small' / 'large': spin-
    selective (smallest / largest spins first); 'composition': composition-preserving
    (quasi-static) -- each spin class is spread evenly over the emission sequence, so the hole
    keeps its Ghosh-Perez composition at every stage."""
    rng = np.random.default_rng(seed)
    if order == 'random':
        return [spins[i] for i in rng.permutation(len(spins))]
    if order in ('small', 'large'):
        return sorted(spins, key=float, reverse=(order == 'large'))
    if order == 'composition':
        keyed = []
        for j in sorted(set(spins), key=float):
            cls = [s for s in spins if s == j]
            keyed += [((i + rng.random()) / len(cls), s) for i, s in enumerate(cls)]
        return [s for _, s in sorted(keyed, key=lambda x: x[0])]
    raise ValueError(order)


def gp_page_order(gamma, order, N=400, seed=0):
    """Page-point area fraction (peak of the Page-average curve, fixed-configuration capacities)
    for GP-sampled spins emitted in the given order."""
    js, ps, _ = gp_weights(gamma)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(js), size=N, p=np.array([float(p) for p in ps]) /
                     sum(float(p) for p in ps))
    spins = gp_order([js[i] for i in idx], order, seed=seed + 7)
    curve, _, _ = page_curve(spins, mpf(str(gamma)), order='given')
    return curve[int(np.argmax([s for _, s, _ in curve]))][0]


# ------------------------------------------------------------ Page average with a centre
def spin_multiplicities(spins):
    """{J: multiplicity of total spin J in tensor_p V_{j_p}} (exact integers; character
    convolution over M)."""
    c = np.array([1], dtype=object)
    for j in spins:
        c = np.convolve(c, np.ones(int(round(2 * float(j))) + 1, dtype=object))
    top = len(c) - 1                                    # index of M = sum j
    out = {}
    for i in range(top, (top - 1) // 2, -1):            # M = J >= 0
        m = c[i] - (c[i + 1] if i + 1 <= top else 0)
        if m:
            out[(2 * i - top) / 2] = int(m)
    return out


def page_centre(multR, multB, mag=lambda key: 2 * key + 1):
    """Average entanglement entropy of a Haar-random pure state with a CENTRE (Bianchi-Dona,
    PRD 100, 105010):  H = (+)_z  (V_z^R (x) M_z^R) (x) (V_z^B (x) M_z^B), one invariant per z,
    with multiplicities m_z^R, m_z^B and a maximally mixed 'magnetic' factor of dimension mag(z)
    (2J+1 for SU(2)).  With p_z ~ Dirichlet(d_z), d_z = m_z^R m_z^B, D = sum d_z, and sector
    states Haar-random,  S_R = H(p) + sum_z p_z [S(psi_z) + ln mag(z)], whose exact mean is
        <S_R> = sum_z (d_z/D) [psi(D+1) - psi(max_z+1) - (min_z-1)/(2 max_z) + ln mag(z)].
    Keys must index matching sectors on both sides.  Returns (<S_R>, D)."""
    dps0 = mp.dps
    mp.dps = max(dps0, 30)
    try:
        keys = [z for z in multR if multB.get(z)]
        D = sum(multR[z] * multB[z] for z in keys)
        psiD = mp.digamma(mpf(D) + 1)
        S = mpf(0)
        for z in keys:
            a, b = multR[z], multB[z]
            lo, hi = min(a, b), max(a, b)
            S += mpf(a * b) / D * (psiD - mp.digamma(mpf(hi) + 1) - mpf(lo - 1) / (2 * hi)
                                   + log(mpf(mag(z))))
        return float(S), D
    finally:
        mp.dps = dps0


def singlet_page_exact(n, k):
    """Exact Haar-random-SINGLET radiation entropy for n spin-1/2 punctures cut k | n-k."""
    half = mpf(1) / 2
    return page_centre(spin_multiplicities([half] * k), spin_multiplicities([half] * (n - k)))[0]


def centre_deficit(n):
    """Midpoint (k = n/2) deficit of the singlet Page value below the UNCONSTRAINED Page value
    ln m - 1/2 of the radiation register (m = 2^{n/2}); referee: an O(1) amount that saturates.
    Returns (S_singlet_mid, deficit vs ln m - 1/2, deficit vs ln m)."""
    k = n // 2
    S = singlet_page_exact(n, k)
    lnm = k * float(log(2))
    return S, lnm - 0.5 - S, lnm - S


def centre_area_split(N, n1):
    """Option-(a) bookkeeping check: a microcanonical two-species horizon (N punctures, exactly
    n1 of spin 1 and N-n1 of spin 1/2, i.e. fixed total area), with the hole space the DIRECT SUM
    over spin configurations of Inv(tensor V_{j_p}).  Cutting k | N-k at the midpoint, the centre
    is z = (n1 in R, J_R); the R multiplicity of z is C(k, n1_R) x mult_J.  Returns
    (S_mid, ln D_tot, check) with check = D_tot == C(N, n1) mult_0 (exact)."""
    from math import comb
    half, one = mpf(1) / 2, mpf(1)
    k = N // 2
    cache = {}

    def mult(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = spin_multiplicities([half] * a + [one] * b)
        return cache[(a, b)]
    multR, multB = {}, {}
    for r in range(max(0, n1 - (N - k)), min(k, n1) + 1):
        mR, mB = mult(k - r, r), mult(N - k - (n1 - r), n1 - r)
        for J, m in mR.items():
            if J in mB:
                multR[(r, J)] = comb(k, r) * m
                multB[(r, J)] = comb(N - k, n1 - r) * mB[J]
    S, D = page_centre(multR, multB, mag=lambda z: 2 * z[1] + 1)
    check = D == comb(N, n1) * mult(N - n1, n1).get(0.0, 0)
    return S, float(log(mpf(D))), check


def area_split_alpha(f):
    """Predicted coefficient of the -alpha sqrt(N) midpoint correction from the abelian (area-
    split) centre of centre_area_split with n1 = f N: the split r = n1_R is hypergeometric with
    sigma^2 = N f(1-f)/4, the typical state's entanglement in sector r is capped by the smaller
    side, and ln(m_R/m_B) has slope 2[ln((1-f)/f) + ln(3/2)] in r (labels + spins), so
    alpha = sigma/sqrt(N) * sqrt(2/pi) * [ln((1-f)/f) + ln(3/2)]."""
    import math
    return math.sqrt(f * (1 - f) / 4) * math.sqrt(2 / math.pi) * (math.log((1 - f) / f) + math.log(1.5))


def area_split_fit(f=0.25, Ns=(32, 64, 128, 256, 512)):
    """Fit S_mid - (1/2) ln D_tot = -alpha sqrt(N) + b ln N + c over the two-species model.
    Returns (alpha_fit, b, c, max residual, alpha_predicted, all exact-dimension checks)."""
    rows = [(N,) + centre_area_split(N, int(round(f * N))) for N in Ns]
    Nv = np.array([r[0] for r in rows], float)
    y = np.array([r[1] - r[2] / 2 for r in rows])
    A = np.column_stack([-np.sqrt(Nv), np.log(Nv), np.ones_like(Nv)])
    (a, b, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(a), float(b), float(c), float(np.abs(A @ [a, b, c] - y).max()), \
        area_split_alpha(f), all(r[3] for r in rows)


def closure_log_fit(Ns=(20, 40, 80, 160, 320, 640)):
    """Gauge-invariant counting: fit ln dim Inv((C^2)^N) = ln Catalan(N/2) against
    N ln2 + c1 ln N + c0 -> c1 must be -3/2 (the Kaul-Majumdar/Meissner closure log)."""
    from math import comb, log as mlog
    xs, ys = [], []
    for N in Ns:
        n = N // 2
        xs.append((N, mlog(N)))
        ys.append(mlog(comb(N, n) // (n + 1)) - N * mlog(2))
    A = np.array([[x[1], 1.0] for x in xs])
    (c1, c0), *_ = np.linalg.lstsq(A, np.array(ys), rcond=None)
    return float(c1), float(c0)


def run():
    N = 400                                   # horizon punctures
    print("=" * 72)
    print("PAGE-CURVE BOOKKEEPING FOR A FINITE LQG HORIZON (conditional on typicality)")
    print("=" * 72)

    # ---- (A) fixed configuration, unconstrained registers (H2(i): j=1/2 at the ABCK gamma0) ----
    j_half = [mpf(1) / 2] * N
    g_match = float(log(mpf(2)) / (pi * sqrt(mpf(3))))     # ln 2 = 2 pi gamma0 sqrt(3)/2
    curve, S0, A0 = page_curve(j_half, g_match)
    peak = max(curve, key=lambda r: r[1])
    print(f"\n[A] j=1/2 horizon, N={N}, gamma0={g_match:.4f} (ABCK matching): ln D/(A/4) = {S0/(A0/4):.6f}")
    print(f"    Page point at area fraction {peak[0]:.4f}; peak {peak[1]:.3f} vs ln m = {S0/2:.3f}"
          f" (deficit {S0/2-peak[1]:.3f} = Page's 1/2 nat); S_rad(end) = {curve[-1][1]:.1e}")

    # ---- (B) SU(2) centre: exact Haar-singlet curve (Bianchi-Dona) ----
    print("\n[B] SU(2) singlet constraint as a centre (exact):")
    print("    n=14 Haar-singlet S(R_k):", " ".join(f"{singlet_page_exact(14, k):.4f}" for k in range(1, 8)))
    for n in (14, 60, 240, 960):
        S, d_page, d_lnm = centre_deficit(n)
        print(f"    n={n:4d}: S_mid={S:9.4f}; below ln m - 1/2 by {d_page:.4f}; below ln m by {d_lnm:.4f}")
    print("    -> an O(1) cost that saturates; the -(3/2) ln N closure log does NOT enter the height")

    # ---- (C) option (a): direct sum over configurations, area split as an abelian centre ----
    a, b, c, res, a_pred, ok = area_split_fit(0.25)
    print(f"\n[C] two-species direct sum (n1 = N/4), midpoint: S_mid - (1/2) ln D_tot = "
          f"-{a:.3f} sqrt(N) + {b:.2f} ln N + {c:.2f}  (max res {res:.1e}; exact dims {ok})")
    print(f"    predicted alpha from area-split fluctuations: {a_pred:.3f}  -> O(sqrt A), subleading")

    # ---- (D) Ghosh-Perez spins: Page point vs emission order ----
    print("\n[D] Ghosh-Perez-sampled spins (N=400): Page-point area fraction by emission order")
    for g in (0.2, 0.274067, 0.5):
        row = {o: gp_page_order(g, o, seed=1) for o in ('random', 'composition', 'small', 'large')}
        print(f"    gamma={g:.4f}: " + "  ".join(f"{o} {v:.3f}" for o, v in row.items()))
    a4, lnd, H, fr = gp_split(gp_gamma_star())
    print(f"    per puncture at gamma*: <a>/4 = {a4:.4f} = <ln d> {lnd:.4f} + H(label) {H:.4f}"
          f"  (fixed configuration holds {fr:.1%})")

    print("\n" + "=" * 72)
    print("Finite capacity fixes the ceiling and the exact return; under typicality the height is")
    print("the centre-corrected Page value and, for exchangeable or composition-preserving emission,")
    print("the Page point sits at half the area for every gamma0.  Typicality itself is assumed.")
    print("=" * 72)


if __name__ == "__main__":
    run()
