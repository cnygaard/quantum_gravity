"""
Geodesic Equation Solver for Quantum Gravity v8

Implements geodesic motion derived from the principle of least distinguishability
(Fisher length minimization) according to v8 Section 13:

    S_Fisher = ∫ √(G_μν^Fisher dx^μ/dτ dx^ν/dτ) dτ

The geodesic equation:

    d²x^μ/dτ² + Γ^μ_νλ (dx^ν/dτ)(dx^λ/dτ) = 0

Physical interpretation: Matter follows paths that minimize the rate of change
of quantum distinguishability with respect to the vacuum.

Supports:
- Timelike geodesics (massive particles, ds² < 0)
- Null geodesics (photons, ds² = 0)
- Schwarzschild and Kerr geometries

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v8, Section 13
"""

import numpy as np
from typing import Optional, Tuple, Callable, List, Union
from scipy.integrate import solve_ivp
import logging

from constants import CONSTANTS
from physics.fisher_metric import FisherMetric

logger = logging.getLogger(__name__)


class GeodesicSolver:
    """
    Geodesic equation solver for curved spacetime.

    Computes Christoffel symbols from the metric and integrates the
    geodesic equation to find particle trajectories.

    Attributes:
        metric_func: Function returning 4x4 metric tensor at position
        epsilon: Step size for numerical derivatives
    """

    def __init__(self, metric_func: Callable[[np.ndarray], np.ndarray],
                 epsilon: float = 1e-6):
        """
        Initialize geodesic solver.

        Args:
            metric_func: Function taking position array (t, r, θ, φ) and
                        returning 4x4 metric tensor g_μν
            epsilon: Step size for numerical differentiation
        """
        self.metric_func = metric_func
        self.epsilon = epsilon
        self._christoffel_cache = {}

    def compute_christoffel(self, x: np.ndarray) -> np.ndarray:
        """
        Compute Christoffel symbols Γ^μ_νλ at position x.

        Γ^μ_νλ = ½ g^μσ (∂_ν g_σλ + ∂_λ g_σν - ∂_σ g_νλ)

        Args:
            x: Position 4-vector (t, r, θ, φ)

        Returns:
            4x4x4 array of Christoffel symbols Γ^μ_νλ
        """
        # Get metric and inverse
        g = self.metric_func(x)
        g_inv = np.linalg.inv(g)

        # Compute metric derivatives numerically
        dg = np.zeros((4, 4, 4))  # dg[sigma, mu, nu] = ∂_sigma g_munu
        for sigma in range(4):
            dg[sigma] = self._metric_derivative(x, sigma)

        # Compute Christoffel symbols
        # Γ^mu_nu_lambda = ½ g^mu_sigma (∂_nu g_sigma_lambda + ∂_lambda g_sigma_nu - ∂_sigma g_nu_lambda)
        Gamma = np.zeros((4, 4, 4))

        for mu in range(4):
            for nu in range(4):
                for lam in range(4):
                    for sigma in range(4):
                        Gamma[mu, nu, lam] += 0.5 * g_inv[mu, sigma] * (
                            dg[nu, sigma, lam] +
                            dg[lam, sigma, nu] -
                            dg[sigma, nu, lam]
                        )

        return Gamma

    def _metric_derivative(self, x: np.ndarray, direction: int) -> np.ndarray:
        """
        Compute ∂_direction g_μν using central difference.

        Args:
            x: Position 4-vector
            direction: Spacetime direction (0-3)

        Returns:
            4x4 array of metric derivatives
        """
        x_plus = x.copy()
        x_minus = x.copy()

        # Adaptive step size based on coordinate magnitude
        h = self.epsilon * max(abs(x[direction]), 1.0)

        x_plus[direction] += h
        x_minus[direction] -= h

        g_plus = self.metric_func(x_plus)
        g_minus = self.metric_func(x_minus)

        return (g_plus - g_minus) / (2 * h)

    def geodesic_rhs(self, tau: float, y: np.ndarray) -> np.ndarray:
        """
        Right-hand side of geodesic equation for integration.

        State vector y = (x^μ, dx^μ/dτ) = (t, r, θ, φ, dt/dτ, dr/dτ, dθ/dτ, dφ/dτ)

        The equations are:
        dx^μ/dτ = v^μ
        dv^μ/dτ = -Γ^μ_νλ v^ν v^λ

        Args:
            tau: Proper time (or affine parameter for null geodesics)
            y: State vector [x^0, x^1, x^2, x^3, v^0, v^1, v^2, v^3]

        Returns:
            Derivative of state vector
        """
        x = y[:4]  # Position
        v = y[4:]  # Velocity dx^μ/dτ

        # Compute Christoffel symbols at current position
        Gamma = self.compute_christoffel(x)

        # Compute acceleration d²x^μ/dτ²
        a = np.zeros(4)
        for mu in range(4):
            for nu in range(4):
                for lam in range(4):
                    a[mu] -= Gamma[mu, nu, lam] * v[nu] * v[lam]

        # Return [velocity, acceleration]
        return np.concatenate([v, a])

    def solve(self, x0: np.ndarray, v0: np.ndarray,
              tau_span: Tuple[float, float],
              n_points: int = 1000,
              method: str = 'RK45',
              events: Optional[List[Callable]] = None) -> dict:
        """
        Solve geodesic equation with given initial conditions.

        Args:
            x0: Initial position (t, r, θ, φ)
            v0: Initial velocity (dt/dτ, dr/dτ, dθ/dτ, dφ/dτ)
            tau_span: (tau_start, tau_end) affine parameter range
            n_points: Number of output points
            method: Integration method ('RK45', 'DOP853', etc.)
            events: List of event functions for termination

        Returns:
            Dictionary with keys:
                'tau': Affine parameter values
                'x': Position history (n_points, 4)
                'v': Velocity history (n_points, 4)
                'ds2': Interval history (should be constant)
                'success': Integration success flag
        """
        y0 = np.concatenate([x0, v0])
        tau_eval = np.linspace(tau_span[0], tau_span[1], n_points)

        result = solve_ivp(
            self.geodesic_rhs,
            tau_span,
            y0,
            method=method,
            t_eval=tau_eval,
            events=events,
            rtol=1e-8,
            atol=1e-10
        )

        if not result.success:
            logger.warning(f"Geodesic integration warning: {result.message}")

        # Extract positions and velocities
        x_history = result.y[:4].T
        v_history = result.y[4:].T

        # Compute ds² = g_μν dx^μ dx^ν to verify geodesic type
        ds2_history = []
        for i, (x, v) in enumerate(zip(x_history, v_history)):
            g = self.metric_func(x)
            ds2 = np.einsum('i,ij,j', v, g, v)
            ds2_history.append(ds2)

        return {
            'tau': result.t,
            'x': x_history,
            'v': v_history,
            'ds2': np.array(ds2_history),
            'success': result.success,
            'message': result.message
        }

    def normalize_velocity(self, x: np.ndarray, v: np.ndarray,
                           geodesic_type: str = 'timelike') -> np.ndarray:
        """
        Normalize velocity vector for given geodesic type.

        For timelike: g_μν v^μ v^ν = -1
        For null: g_μν v^μ v^ν = 0

        Args:
            x: Position 4-vector
            v: Velocity 4-vector (will be normalized)
            geodesic_type: 'timelike' or 'null'

        Returns:
            Normalized velocity vector
        """
        g = self.metric_func(x)
        ds2 = np.einsum('i,ij,j', v, g, v)

        if geodesic_type == 'timelike':
            # Normalize to ds² = -1
            if ds2 >= 0:
                raise ValueError("Cannot normalize spacelike vector to timelike")
            scale = 1.0 / np.sqrt(-ds2)
        elif geodesic_type == 'null':
            # For null, we need to adjust to make ds² = 0
            # This is non-trivial; typically set v^0 from spatial components
            if abs(g[0, 0]) < 1e-15:
                raise ValueError("Cannot solve null condition with g_tt = 0")

            # Solve g_00 (v^0)² + 2 g_0i v^0 v^i + g_ij v^i v^j = 0
            v_space = v[1:4]
            g_space = g[1:4, 1:4]
            g_0space = g[0, 1:4]

            # Quadratic coefficients for v^0
            a = g[0, 0]
            b = 2 * np.dot(g_0space, v_space)
            c = np.einsum('i,ij,j', v_space, g_space, v_space)

            discriminant = b**2 - 4*a*c
            if discriminant < 0:
                raise ValueError("No null geodesic solution exists")

            # Choose the positive root for outgoing, negative for ingoing
            v0_new = (-b + np.sqrt(discriminant)) / (2*a)
            v_normalized = v.copy()
            v_normalized[0] = v0_new
            return v_normalized
        else:
            raise ValueError(f"Unknown geodesic type: {geodesic_type}")

        return v * scale


class SchwarzschildGeodesics:
    """
    Specialized geodesic solver for Schwarzschild geometry.

    Provides efficient methods for computing orbits and photon paths
    around non-rotating black holes.
    """

    def __init__(self, M: float):
        """
        Initialize Schwarzschild geodesic solver.

        Args:
            M: Black hole mass in Planck units
        """
        self.M = M
        self.r_s = 2 * CONSTANTS['G'] * M  # Schwarzschild radius

        # Create metric function
        def schwarzschild_metric(x: np.ndarray) -> np.ndarray:
            r = x[1]
            theta = x[2]

            if r <= self.r_s:
                logger.warning(f"Position inside horizon: r={r:.4f} < r_s={self.r_s:.4f}")
                r = self.r_s * 1.001  # Avoid singularity

            f = 1 - self.r_s / r

            g = np.diag([-f, 1/f, r**2, r**2 * np.sin(theta)**2])
            return g

        self.solver = GeodesicSolver(schwarzschild_metric)

    def circular_orbit_velocity(self, r: float) -> Tuple[float, float]:
        """
        Compute velocity for circular orbit at radius r.

        For circular orbits: dt/dτ and dφ/dτ are constant, dr/dτ = dθ/dτ = 0

        Derived from:
        1. Radial geodesic equation with d²r/dτ² = 0
        2. Normalization g_μν v^μ v^ν = -1

        Args:
            r: Orbital radius (must be > 3 r_s/2 for stable orbit)

        Returns:
            Tuple (dt/dτ, dφ/dτ)

        Raises:
            ValueError: If r < 3 r_s/2 (unstable orbit region)
        """
        if r < 1.5 * self.r_s:
            raise ValueError(f"No stable circular orbit for r < 3 r_s/2 = {1.5*self.r_s:.4f}")

        # Metric components
        f = 1 - self.r_s / r

        # Coordinate angular velocity Ω = dφ/dt from radial geodesic equation
        # Ω² = r_s / (2 r³) = G M / r³ (Kepler's law)
        Omega_squared = self.r_s / (2 * r**3)
        Omega = np.sqrt(Omega_squared)

        # From normalization: g_tt (dt/dτ)² + g_φφ (dφ/dτ)² = -1
        # => -f (dt/dτ)² + r² (Ω dt/dτ)² = -1
        # => (dt/dτ)² (r² Ω² - f) = -1
        # => (dt/dτ)² = 1 / (f - r² Ω²)

        denominator = f - r**2 * Omega_squared
        if denominator <= 0:
            raise ValueError(f"No timelike circular orbit at r = {r:.4f}")

        dt_dtau = 1.0 / np.sqrt(denominator)
        dphi_dtau = Omega * dt_dtau

        return dt_dtau, dphi_dtau

    def isco_radius(self) -> float:
        """
        Return the Innermost Stable Circular Orbit (ISCO) radius.

        For Schwarzschild: r_ISCO = 6 G M = 3 r_s

        Returns:
            ISCO radius in Planck units
        """
        return 3 * self.r_s

    def photon_sphere_radius(self) -> float:
        """
        Return the photon sphere radius.

        For Schwarzschild: r_ph = 3 G M = 1.5 r_s

        Returns:
            Photon sphere radius in Planck units
        """
        return 1.5 * self.r_s

    def compute_orbit(self, r0: float, phi0: float = 0.0,
                      n_orbits: float = 2.0,
                      n_points: int = 1000) -> dict:
        """
        Compute circular orbit trajectory.

        Args:
            r0: Initial orbital radius
            phi0: Initial azimuthal angle
            n_orbits: Number of orbits to compute
            n_points: Number of output points

        Returns:
            Geodesic solution dictionary
        """
        # Initial position (in equatorial plane)
        x0 = np.array([0.0, r0, np.pi/2, phi0])

        # Compute circular orbit velocity
        dt_dtau, dphi_dtau = self.circular_orbit_velocity(r0)
        v0 = np.array([dt_dtau, 0.0, 0.0, dphi_dtau])

        # Orbital period in proper time
        T = 2 * np.pi / dphi_dtau
        tau_end = n_orbits * T

        return self.solver.solve(x0, v0, (0, tau_end), n_points=n_points)


class KerrGeodesics:
    """
    Specialized geodesic solver for Kerr geometry.

    Provides methods for computing orbits around rotating black holes,
    including frame-dragging effects.
    """

    def __init__(self, M: float, a: float):
        """
        Initialize Kerr geodesic solver.

        Args:
            M: Black hole mass in Planck units
            a: Spin parameter (|a| ≤ r_s/2)
        """
        self.M = M
        self.a = a
        self.r_s = 2 * CONSTANTS['G'] * M

        if abs(a) > self.r_s / 2:
            logger.warning(f"Spin parameter |a|={abs(a):.4f} exceeds extremal limit")

        self.fisher_metric = FisherMetric()

        def kerr_metric(x: np.ndarray) -> np.ndarray:
            r, theta = x[1], x[2]
            return self.fisher_metric.compute_kerr_fisher(r, theta, M, a)

        self.solver = GeodesicSolver(kerr_metric)

    def outer_horizon_radius(self) -> float:
        """Return outer horizon radius r_+."""
        r_outer, _ = self.fisher_metric.kerr_horizon_radii(self.M, self.a)
        return r_outer

    def inner_horizon_radius(self) -> float:
        """Return inner horizon radius r_-."""
        _, r_inner = self.fisher_metric.kerr_horizon_radii(self.M, self.a)
        return r_inner

    def isco_radius(self, prograde: bool = True) -> float:
        """
        Compute ISCO radius for Kerr black hole.

        For prograde orbits (same direction as BH spin), ISCO is smaller.
        For retrograde orbits (opposite to spin), ISCO is larger.

        Args:
            prograde: True for prograde, False for retrograde orbit

        Returns:
            ISCO radius
        """
        # Using Bardeen-Press-Teukolsky formula
        a_star = self.a / (self.r_s / 2)  # Dimensionless spin

        z1 = 1 + (1 - a_star**2)**(1/3) * (
            (1 + a_star)**(1/3) + (1 - a_star)**(1/3)
        )
        z2 = np.sqrt(3 * a_star**2 + z1**2)

        if prograde:
            r_isco = (self.r_s / 2) * (3 + z2 - np.sqrt((3 - z1) * (3 + z1 + 2*z2)))
        else:
            r_isco = (self.r_s / 2) * (3 + z2 + np.sqrt((3 - z1) * (3 + z1 + 2*z2)))

        return r_isco

    def frame_dragging_at_radius(self, r: float, theta: float = np.pi/2) -> float:
        """
        Compute frame dragging angular velocity at given position.

        Args:
            r: Radial coordinate
            theta: Polar angle (default: equatorial)

        Returns:
            Frame dragging angular velocity ω
        """
        return self.fisher_metric.frame_dragging_angular_velocity(
            r, theta, self.M, self.a
        )


def horizon_event(r_horizon: float):
    """
    Create event function that triggers at horizon crossing.

    Args:
        r_horizon: Horizon radius

    Returns:
        Event function for solve_ivp
    """
    def event(tau: float, y: np.ndarray) -> float:
        r = y[1]
        return r - r_horizon * 1.01  # Small buffer outside horizon

    event.terminal = True
    event.direction = -1  # Trigger when crossing inward

    return event
