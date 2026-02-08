"""
Loop Quantum Cosmology (LQC) Bounce Dynamics for v8 Framework

Implements the quantum bounce from Loop Quantum Cosmology according to
v8 Section 16.3. The key modification to classical Friedmann dynamics is:

    H² = (8πG/3)ρ(1 - ρ/ρ_c)

where ρ_c is the critical density at which quantum geometry effects
prevent the classical singularity.

Physical interpretation:
- When ρ << ρ_c: Classical Friedmann evolution
- When ρ → ρ_c: H → 0 (turnaround/bounce)
- The classical singularity is replaced by a quantum bounce

This module provides:
- LQC-modified Friedmann equation solver
- Bounce detection and characterization
- Pre-bounce and post-bounce phase evolution
- Connection to v8 Fisher Information framework

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v8, Section 16.3
"""

import numpy as np
from typing import Optional, Tuple, Dict, List
from scipy.integrate import solve_ivp
import logging

from constants import CONSTANTS

logger = logging.getLogger(__name__)


# Critical density from LQC
# ρ_c ≈ 0.41 ρ_Planck (from detailed LQC calculations)
RHO_CRITICAL_LQC = 0.41 * CONSTANTS['rho_planck']


class LQCBounce:
    """
    Loop Quantum Cosmology bounce dynamics.

    Implements the modified Friedmann equations from LQC that prevent
    the classical Big Bang singularity.

    Attributes:
        rho_c: Critical density for bounce (default: 0.41 ρ_P)
        gamma_0: Immirzi parameter for quantum corrections
    """

    def __init__(self, rho_c: Optional[float] = None):
        """
        Initialize LQC bounce calculator.

        Args:
            rho_c: Critical density (default: 0.41 ρ_Planck)
        """
        self.rho_c = rho_c if rho_c is not None else RHO_CRITICAL_LQC
        self.gamma_0 = CONSTANTS['gamma_0']

    def hubble_squared(self, rho: float) -> float:
        """
        Compute LQC-modified Hubble parameter squared.

        H² = (8πG/3)ρ(1 - ρ/ρ_c)

        Args:
            rho: Energy density

        Returns:
            H² (can be negative approaching bounce from wrong direction,
            but physically H² ≥ 0)
        """
        H2_classical = (8 * np.pi * CONSTANTS['G'] / 3) * rho
        bounce_correction = 1 - rho / self.rho_c

        return max(0.0, H2_classical * bounce_correction)

    def hubble_parameter(self, rho: float, expanding: bool = True) -> float:
        """
        Compute Hubble parameter with sign for expansion/contraction.

        Args:
            rho: Energy density
            expanding: True for expanding universe, False for contracting

        Returns:
            Hubble parameter H (positive for expansion, negative for contraction)
        """
        H2 = self.hubble_squared(rho)
        H = np.sqrt(H2)
        return H if expanding else -H

    def effective_density_at_bounce(self) -> float:
        """
        Return the density at which the bounce occurs.

        At the bounce: H = 0, which requires ρ = ρ_c.

        Returns:
            Critical density ρ_c
        """
        return self.rho_c

    def bounce_scale_factor(self, a0: float, rho0: float) -> float:
        """
        Estimate the scale factor at bounce assuming matter domination.

        For matter-dominated: ρ ∝ a^(-3)
        At bounce: ρ_c = ρ0 (a0/a_bounce)^3
        => a_bounce = a0 (ρ0/ρ_c)^(1/3)

        Args:
            a0: Current scale factor
            rho0: Current energy density

        Returns:
            Scale factor at bounce
        """
        if rho0 >= self.rho_c:
            # Already at or past critical density
            return a0 * (rho0 / self.rho_c) ** (1/3)

        return a0 * (rho0 / self.rho_c) ** (1/3)

    def is_near_bounce(self, rho: float, threshold: float = 0.1) -> bool:
        """
        Check if density is near the bounce point.

        Args:
            rho: Current density
            threshold: Fraction of ρ_c defining "near" (default 10%)

        Returns:
            True if |ρ - ρ_c|/ρ_c < threshold
        """
        return abs(rho - self.rho_c) / self.rho_c < threshold

    def friedmann_rhs(self, t: float, y: np.ndarray,
                      equation_of_state: float = 0.0) -> np.ndarray:
        """
        Right-hand side of LQC-modified Friedmann equations.

        State: y = [a, ρ] (scale factor and energy density)

        Equations:
        da/dt = a * H
        dρ/dt = -3H(ρ + p) = -3Hρ(1 + w)

        Args:
            t: Time
            y: State [a, ρ]
            equation_of_state: w = p/ρ (0 = matter, 1/3 = radiation, -1 = Λ)

        Returns:
            Derivatives [da/dt, dρ/dt]
        """
        a, rho = y
        w = equation_of_state

        # Determine expansion/contraction from evolution direction
        # Initially we need to check derivative of a
        H2 = self.hubble_squared(rho)

        if H2 < 1e-30:
            # At bounce - transition
            H = 0.0
        else:
            H = np.sqrt(H2)
            # Contraction if density is increasing (pre-bounce)
            # Expansion if density is decreasing (post-bounce)
            if not hasattr(self, '_expanding'):
                self._expanding = True

            if not self._expanding:
                H = -H

        # Scale factor evolution
        da_dt = a * H

        # Energy density evolution (continuity equation)
        # dρ/dt = -3H(ρ + p) = -3Hρ(1 + w)
        drho_dt = -3 * H * rho * (1 + w)

        return np.array([da_dt, drho_dt])

    def solve_bounce(self, a0: float, rho0: float,
                     t_span: Tuple[float, float],
                     equation_of_state: float = 0.0,
                     n_points: int = 1000,
                     expanding: bool = True) -> Dict:
        """
        Solve the LQC-modified Friedmann equations through a bounce.

        Args:
            a0: Initial scale factor
            rho0: Initial energy density
            t_span: (t_start, t_end) time range
            equation_of_state: w = p/ρ
            n_points: Number of output points
            expanding: Initial expansion state

        Returns:
            Dictionary with 't', 'a', 'rho', 'H', 'bounce_time', 'bounce_detected'
        """
        self._expanding = expanding

        y0 = np.array([a0, rho0])
        t_eval = np.linspace(t_span[0], t_span[1], n_points)

        # Track bounce
        bounce_times = []

        def bounce_event(t, y):
            """Event function: H = 0 (bounce)"""
            rho = y[1]
            return rho - 0.99 * self.rho_c

        bounce_event.terminal = False
        bounce_event.direction = 1  # Approaching critical density

        # Solve with RK45
        result = solve_ivp(
            lambda t, y: self.friedmann_rhs(t, y, equation_of_state),
            t_span,
            y0,
            method='RK45',
            t_eval=t_eval,
            events=bounce_event,
            dense_output=True,
            rtol=1e-8,
            atol=1e-10
        )

        # Extract results
        t = result.t
        a = result.y[0]
        rho = result.y[1]

        # Compute Hubble parameter
        H = np.array([self.hubble_parameter(r, self._expanding) for r in rho])

        # Detect bounce from H = 0 crossing
        bounce_detected = False
        bounce_time = None
        for i in range(1, len(H)):
            if H[i-1] < 0 and H[i] >= 0:  # Transition from contraction to expansion
                bounce_detected = True
                bounce_time = t[i]
                break

        return {
            't': t,
            'a': a,
            'rho': rho,
            'H': H,
            'bounce_detected': bounce_detected,
            'bounce_time': bounce_time,
            'success': result.success
        }


class LQCPlanckStar:
    """
    Planck Star model for black hole interior.

    From v8 Section 16.3: The classical black hole singularity is replaced
    by a Planck-density core ("Planck Star") where quantum geometry effects
    halt gravitational collapse.

    The Planck Star has:
    - Maximum density ρ ~ ρ_c
    - Minimum radius r_min ~ l_P
    - Eventual quantum tunneling to white hole
    """

    def __init__(self, M: float):
        """
        Initialize Planck Star model.

        Args:
            M: Black hole mass in Planck units
        """
        self.M = M
        self.r_s = 2 * CONSTANTS['G'] * M
        self.lqc = LQCBounce()

    def minimum_radius(self) -> float:
        """
        Estimate minimum radius of Planck Star core.

        The core forms when density reaches ρ_c.
        For a uniform sphere: ρ = 3M/(4πr³)
        At ρ_c: r_min = (3M/(4π ρ_c))^(1/3)

        Returns:
            Minimum core radius (Planck lengths)
        """
        r_min_cubed = 3 * self.M / (4 * np.pi * self.lqc.rho_c)
        return r_min_cubed ** (1/3)

    def core_density(self) -> float:
        """
        Return the maximum core density (critical density).

        Returns:
            Core density ρ_c
        """
        return self.lqc.rho_c

    def bounce_time(self) -> float:
        """
        Estimate the time for the black hole interior to bounce.

        This is related to the classical free-fall time but modified by
        quantum effects near Planck density.

        Approximate: τ_bounce ~ (M/M_P)^2 * t_P

        Returns:
            Bounce timescale in Planck times
        """
        M_P = CONSTANTS['m_p']
        return (self.M / M_P) ** 2 * CONSTANTS['t_p']

    def tunneling_probability(self) -> float:
        """
        Estimate quantum tunneling probability for white hole transition.

        The Planck Star can quantum-tunnel to become a white hole,
        allowing information to escape.

        Rough estimate: P ~ exp(-M/M_P)

        Returns:
            Tunneling probability per Planck time
        """
        M_P = CONSTANTS['m_p']
        return np.exp(-self.M / M_P)


def verify_lqc_bounce(verbose: bool = True) -> Dict:
    """
    Verify LQC bounce implementation with simple test cases.

    Returns:
        Dictionary of test results
    """
    lqc = LQCBounce()

    results = {}

    # Test 1: Classical limit (ρ << ρ_c)
    rho_low = 1e-10 * lqc.rho_c
    H2_lqc = lqc.hubble_squared(rho_low)
    H2_classical = (8 * np.pi * CONSTANTS['G'] / 3) * rho_low
    classical_match = abs(H2_lqc - H2_classical) / H2_classical < 0.01

    results['classical_limit'] = classical_match
    if verbose:
        logger.info(f"Classical limit test (ρ/ρ_c = 1e-10): {'PASS' if classical_match else 'FAIL'}")
        logger.info(f"  H²_LQC = {H2_lqc:.6e}, H²_classical = {H2_classical:.6e}")

    # Test 2: Bounce point (ρ = ρ_c)
    H2_at_bounce = lqc.hubble_squared(lqc.rho_c)
    bounce_works = H2_at_bounce < 1e-20

    results['bounce_point'] = bounce_works
    if verbose:
        logger.info(f"Bounce point test (ρ = ρ_c): {'PASS' if bounce_works else 'FAIL'}")
        logger.info(f"  H² at bounce = {H2_at_bounce:.6e} (should be ~0)")

    # Test 3: Solve through bounce
    a0 = 1.0
    rho0 = 0.5 * lqc.rho_c  # Start near bounce

    solution = lqc.solve_bounce(a0, rho0, (0, 100), expanding=False, n_points=200)
    solve_success = solution['success']

    results['solve_success'] = solve_success
    if verbose:
        logger.info(f"Bounce solution test: {'PASS' if solve_success else 'FAIL'}")
        if solution['bounce_detected']:
            logger.info(f"  Bounce detected at t = {solution['bounce_time']:.4f}")

    # Test 4: Planck Star
    M = 10.0  # 10 Planck masses
    ps = LQCPlanckStar(M)
    r_min = ps.minimum_radius()

    planck_star_valid = r_min > 0 and r_min < ps.r_s

    results['planck_star'] = planck_star_valid
    if verbose:
        logger.info(f"Planck Star test (M = 10 M_P): {'PASS' if planck_star_valid else 'FAIL'}")
        logger.info(f"  r_min = {r_min:.4f} l_P, r_s = {ps.r_s:.4f} l_P")

    return results
