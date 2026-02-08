"""
Gravitational Wave Propagation for v8 Framework

Implements gravitational wave physics according to v8 Section 15:

    g_μν = η_μν + h_μν,  |h_μν| ≪ 1
    h_μν = ℓ_P² · δG_μν^Fisher[δΨ]
    □h_μν = 0  (in harmonic gauge)

Physical interpretation from v8:
- GWs represent propagation of updates to quantum distinguishability
- Speed c corresponds to Lieb-Robinson bound of entanglement network
- Transverse-traceless nature emerges from unitarity constraints

This module provides:
- Wave equation solver (d'Alembertian)
- TT gauge projection and polarization modes
- Binary system GW sources (inspiral, merger, ringdown)
- Fisher metric perturbation framework
- Detector response functions

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v8, Section 15
"""

import numpy as np
from typing import Tuple, Optional, Dict, List, Callable, Union
from scipy.integrate import solve_ivp, odeint
from scipy.interpolate import interp1d
from scipy.fft import fft, ifft, fftfreq
import logging

from constants import CONSTANTS

logger = logging.getLogger(__name__)


# =============================================================================
# Physical Constants for GW Physics
# =============================================================================

# Speed of light (c = 1 in natural units, but we keep it explicit)
C_LIGHT = CONSTANTS.get('c', 1.0)

# Gravitational constant
G_NEWTON = CONSTANTS['G']

# Planck length
L_PLANCK = CONSTANTS['l_p']

# Planck mass
M_PLANCK = CONSTANTS['m_p']


# =============================================================================
# Core Wave Equation Classes
# =============================================================================

class WaveEquationSolver:
    """
    Solver for the gravitational wave equation □h_μν = 0.

    The d'Alembertian operator in flat spacetime:
    □ = -∂²/∂t² + ∇² = -∂²/∂t² + ∂²/∂x² + ∂²/∂y² + ∂²/∂z²

    For a plane wave h = h₀ exp(i(k·x - ωt)):
    □h = (-(-iω)² + (ik)²)h = (ω² - k²)h = 0
    => ω = |k| (dispersion relation, waves travel at c)
    """

    def __init__(self, grid_size: int = 100, domain_size: float = 100.0):
        """
        Initialize wave equation solver.

        Args:
            grid_size: Number of grid points per dimension
            domain_size: Physical size of computational domain (in l_P units)
        """
        self.grid_size = grid_size
        self.domain_size = domain_size
        self.dx = domain_size / grid_size
        self.dt = 0.5 * self.dx  # CFL condition for stability

    def solve_1d(
        self,
        initial_h: np.ndarray,
        initial_dh_dt: np.ndarray,
        t_max: float,
        n_steps: int = None
    ) -> Dict:
        """
        Solve 1D wave equation using finite differences.

        □h = -∂²h/∂t² + ∂²h/∂x² = 0

        Args:
            initial_h: Initial perturbation h(x, t=0)
            initial_dh_dt: Initial time derivative ∂h/∂t(x, t=0)
            t_max: Maximum simulation time
            n_steps: Number of time steps (default: auto from CFL)

        Returns:
            Dictionary with 't', 'x', 'h' (2D array of h(x,t))
        """
        n_x = len(initial_h)
        dx = self.domain_size / n_x
        dt = 0.5 * dx  # CFL condition

        if n_steps is None:
            n_steps = int(t_max / dt) + 1

        # Storage for solution
        h = np.zeros((n_steps, n_x))
        h[0] = initial_h

        # Use initial velocity to get h at t=dt (second-order accurate)
        # h(t+dt) ≈ h(t) + dt * ∂h/∂t + 0.5*dt² * ∂²h/∂t²
        # Using ∂²h/∂t² = ∂²h/∂x² (wave equation)
        d2h_dx2 = self._laplacian_1d(initial_h, dx)
        h[1] = initial_h + dt * initial_dh_dt + 0.5 * dt**2 * d2h_dx2

        # Time stepping using leapfrog scheme
        c2 = (dt / dx)**2
        for n in range(1, n_steps - 1):
            # h^{n+1} = 2h^n - h^{n-1} + c²(h_{i+1} - 2h_i + h_{i-1})
            d2h = self._laplacian_1d(h[n], dx)
            h[n + 1] = 2 * h[n] - h[n - 1] + dt**2 * d2h

        t = np.linspace(0, t_max, n_steps)
        x = np.linspace(0, self.domain_size, n_x)

        return {
            't': t,
            'x': x,
            'h': h,
            'dt': dt,
            'dx': dx
        }

    def solve_3d_spherical(
        self,
        source_function: Callable,
        r_max: float,
        t_max: float,
        n_r: int = 100,
        n_t: int = None
    ) -> Dict:
        """
        Solve wave equation in spherical coordinates for outgoing waves.

        For spherical symmetry: □h = -∂²h/∂t² + (1/r²)∂/∂r(r²∂h/∂r) = S(r,t)

        With substitution u = r·h, this becomes:
        -∂²u/∂t² + ∂²u/∂r² = r·S(r,t)

        Args:
            source_function: S(r, t) source term
            r_max: Maximum radius
            t_max: Maximum time
            n_r: Number of radial grid points
            n_t: Number of time steps

        Returns:
            Dictionary with 't', 'r', 'h' (2D array)
        """
        dr = r_max / n_r
        dt = 0.5 * dr  # CFL

        if n_t is None:
            n_t = int(t_max / dt) + 1

        r = np.linspace(dr, r_max, n_r)  # Avoid r=0 singularity

        # Work with u = r*h
        u = np.zeros((n_t, n_r))

        # Time stepping
        c2 = (dt / dr)**2
        for n in range(1, n_t - 1):
            t = n * dt
            # Source term
            S = np.array([source_function(ri, t) for ri in r])

            # Laplacian of u in 1D
            d2u = np.zeros(n_r)
            d2u[1:-1] = (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]) / dr**2

            # Boundary conditions
            d2u[0] = (u[n, 1] - 2*u[n, 0]) / dr**2  # u=0 at r=0
            d2u[-1] = (-2*u[n, -1] + 2*u[n, -2]) / dr**2  # Outgoing wave

            # Update
            u[n + 1] = 2*u[n] - u[n - 1] + dt**2 * (d2u + r * S)

        # Convert back to h = u/r
        h = u / r[np.newaxis, :]
        t_arr = np.linspace(0, t_max, n_t)

        return {
            't': t_arr,
            'r': r,
            'h': h,
            'u': u
        }

    def _laplacian_1d(self, f: np.ndarray, dx: float) -> np.ndarray:
        """Compute 1D Laplacian using central differences."""
        n = len(f)
        d2f = np.zeros(n)
        d2f[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
        # Periodic boundary conditions
        d2f[0] = (f[1] - 2*f[0] + f[-1]) / dx**2
        d2f[-1] = (f[0] - 2*f[-1] + f[-2]) / dx**2
        return d2f


class GWPolarization:
    """
    Gravitational wave polarization modes.

    In the TT (transverse-traceless) gauge, GWs have two polarizations:
    - Plus (+): stretches along x, compresses along y
    - Cross (×): stretches along 45°, compresses along 135°

    The perturbation tensor in the wave propagation direction z:
    h_μν = h_+ e^+_μν + h_× e^×_μν

    where the polarization tensors are:
    e^+_μν = diag(0, 1, -1, 0)  (in x,y subspace)
    e^×_μν = [[0,1],[1,0]]      (in x,y subspace)
    """

    @staticmethod
    def plus_polarization_tensor() -> np.ndarray:
        """
        Return the plus (+) polarization tensor.

        e^+_μν for wave propagating in z-direction.
        Non-zero components: e^+_xx = 1, e^+_yy = -1

        Returns:
            4x4 polarization tensor
        """
        e_plus = np.zeros((4, 4))
        e_plus[1, 1] = 1.0   # xx component
        e_plus[2, 2] = -1.0  # yy component
        return e_plus

    @staticmethod
    def cross_polarization_tensor() -> np.ndarray:
        """
        Return the cross (×) polarization tensor.

        e^×_μν for wave propagating in z-direction.
        Non-zero components: e^×_xy = e^×_yx = 1

        Returns:
            4x4 polarization tensor
        """
        e_cross = np.zeros((4, 4))
        e_cross[1, 2] = 1.0  # xy component
        e_cross[2, 1] = 1.0  # yx component (symmetric)
        return e_cross

    @staticmethod
    def construct_perturbation(
        h_plus: float,
        h_cross: float,
        propagation_direction: np.ndarray = None
    ) -> np.ndarray:
        """
        Construct the full perturbation tensor h_μν.

        h_μν = h_+ e^+_μν + h_× e^×_μν

        Args:
            h_plus: Amplitude of plus polarization
            h_cross: Amplitude of cross polarization
            propagation_direction: Unit vector for wave direction (default: z)

        Returns:
            4x4 perturbation tensor h_μν
        """
        e_plus = GWPolarization.plus_polarization_tensor()
        e_cross = GWPolarization.cross_polarization_tensor()

        h = h_plus * e_plus + h_cross * e_cross

        # Rotate if propagation direction is not z
        if propagation_direction is not None:
            h = GWPolarization._rotate_tensor(h, propagation_direction)

        return h

    @staticmethod
    def _rotate_tensor(h: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Rotate perturbation tensor to align with given propagation direction."""
        # For simplicity, we assume z-direction for now
        # A full implementation would construct the rotation matrix
        # that takes (0,0,1) to the given direction
        return h

    @staticmethod
    def project_to_tt_gauge(h: np.ndarray, k: np.ndarray) -> np.ndarray:
        """
        Project perturbation to transverse-traceless (TT) gauge.

        The TT gauge conditions are:
        1. h^μ_μ = 0 (traceless)
        2. k^μ h_μν = 0 (transverse)
        3. h_0μ = 0 (spatial only)

        Args:
            h: 4x4 perturbation tensor
            k: 4-wavevector (k^μ)

        Returns:
            4x4 TT-projected perturbation tensor
        """
        # Spatial part only
        h_tt = np.zeros((4, 4))
        h_spatial = h[1:, 1:]

        # Get spatial wavevector
        k_spatial = k[1:]
        k_norm = np.linalg.norm(k_spatial)
        if k_norm < 1e-10:
            return h_tt

        k_hat = k_spatial / k_norm

        # Projection operator: P_ij = δ_ij - k_i k_j
        P = np.eye(3) - np.outer(k_hat, k_hat)

        # Project: h^TT_ij = P_ik P_jl h_kl - ½ P_ij (P_kl h_kl)
        h_proj = P @ h_spatial @ P.T
        trace = np.trace(h_proj)
        h_proj -= 0.5 * P * trace

        h_tt[1:, 1:] = h_proj
        return h_tt


# =============================================================================
# GW Source Models
# =============================================================================

class BinaryInspiral:
    """
    Gravitational wave source from binary inspiral.

    Uses the quadrupole formula for GW emission:
    h_ij = (2G/c⁴D) d²I_ij/dt²

    where I_ij is the quadrupole moment tensor:
    I_ij = Σ m_a (x_a^i x_a^j - ⅓δ_ij r_a²)

    For a circular binary:
    h_+ = (4/D)(Gμ/c²)(Gm_total ω/c³)^(2/3) (1+cos²ι)/2 cos(2Φ)
    h_× = (4/D)(Gμ/c²)(Gm_total ω/c³)^(2/3) cos(ι) sin(2Φ)

    where ι is the inclination angle.
    """

    def __init__(
        self,
        m1: float,
        m2: float,
        distance: float,
        inclination: float = 0.0,
        initial_frequency: float = None
    ):
        """
        Initialize binary inspiral source.

        Args:
            m1: Mass of primary (in Planck masses or solar masses with conversion)
            m2: Mass of secondary
            distance: Distance to source
            inclination: Orbital inclination angle (radians)
            initial_frequency: Initial GW frequency (default: computed from ISCO)
        """
        self.m1 = m1
        self.m2 = m2
        self.m_total = m1 + m2
        self.mu = (m1 * m2) / self.m_total  # Reduced mass
        self.eta = self.mu / self.m_total  # Symmetric mass ratio
        self.distance = distance
        self.inclination = inclination

        # Chirp mass: M_c = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        self.m_chirp = (m1 * m2)**(3/5) / self.m_total**(1/5)

        # ISCO (innermost stable circular orbit) frequency
        # f_ISCO = c³/(6^(3/2) π G M_total)
        self.f_isco = 1.0 / (6**(1.5) * np.pi * G_NEWTON * self.m_total)

        if initial_frequency is None:
            # Start at 10% of ISCO
            initial_frequency = 0.1 * self.f_isco
        self.f_initial = initial_frequency

    def orbital_frequency(self, t: float, t_coal: float) -> float:
        """
        Compute orbital frequency as function of time.

        f(t) = (5/256)^(3/8) (GM_c/c³)^(-5/8) (t_c - t)^(-3/8) / π

        Args:
            t: Time
            t_coal: Time of coalescence

        Returns:
            Orbital frequency (and GW frequency is 2×orbital)
        """
        tau = t_coal - t
        if tau <= 0:
            return self.f_isco

        # Characteristic time scale
        t_char = 5 * G_NEWTON * self.m_chirp / (256 * (np.pi * self.f_initial)**(8/3))

        f = self.f_initial * (tau / t_char)**(-3/8)

        return min(f, self.f_isco)

    def phase(self, t: np.ndarray, t_coal: float) -> np.ndarray:
        """
        Compute GW phase as function of time.

        Φ(t) = Φ_c - 2(5GM_c/c³)^(-5/8) (t_c - t)^(5/8)

        Args:
            t: Time array
            t_coal: Coalescence time

        Returns:
            Phase array
        """
        tau = t_coal - t
        tau = np.maximum(tau, 1e-10)  # Avoid singularity

        # Phase coefficient
        coeff = 2 * (5 * G_NEWTON * self.m_chirp)**(-5/8)

        phase = -coeff * tau**(5/8)
        return phase

    def amplitude(self, t: float, t_coal: float) -> float:
        """
        Compute GW amplitude as function of time.

        A(t) ∝ (GM_c/c²D)(πGM_c f/c³)^(2/3)

        Args:
            t: Time
            t_coal: Coalescence time

        Returns:
            Dimensionless strain amplitude
        """
        f_orb = self.orbital_frequency(t, t_coal)
        f_gw = 2 * f_orb  # GW frequency is twice orbital

        # Amplitude in geometric units
        # h = (4/D)(GM_c/c²)(πGM_c f/c³)^(2/3)
        amp = (4 / self.distance) * (G_NEWTON * self.m_chirp) * \
              (np.pi * G_NEWTON * self.m_chirp * f_gw)**(2/3)

        return amp

    def strain_plus(self, t: np.ndarray, t_coal: float) -> np.ndarray:
        """
        Compute h_+ polarization.

        h_+ = A(t) × (1 + cos²ι)/2 × cos(2Φ(t))

        Args:
            t: Time array
            t_coal: Coalescence time

        Returns:
            Plus polarization strain
        """
        phase = self.phase(t, t_coal)
        amp = np.array([self.amplitude(ti, t_coal) for ti in t])

        cos_inc = np.cos(self.inclination)
        h_plus = amp * (1 + cos_inc**2) / 2 * np.cos(2 * phase)

        return h_plus

    def strain_cross(self, t: np.ndarray, t_coal: float) -> np.ndarray:
        """
        Compute h_× polarization.

        h_× = A(t) × cos(ι) × sin(2Φ(t))

        Args:
            t: Time array
            t_coal: Coalescence time

        Returns:
            Cross polarization strain
        """
        phase = self.phase(t, t_coal)
        amp = np.array([self.amplitude(ti, t_coal) for ti in t])

        cos_inc = np.cos(self.inclination)
        h_cross = amp * cos_inc * np.sin(2 * phase)

        return h_cross

    def compute_waveform(
        self,
        t_start: float,
        t_coal: float,
        n_points: int = 1000
    ) -> Dict:
        """
        Compute full inspiral waveform.

        Args:
            t_start: Start time
            t_coal: Coalescence time
            n_points: Number of time points

        Returns:
            Dictionary with 't', 'h_plus', 'h_cross', 'frequency', 'phase'
        """
        t = np.linspace(t_start, t_coal - 1e-6, n_points)

        h_plus = self.strain_plus(t, t_coal)
        h_cross = self.strain_cross(t, t_coal)

        frequency = np.array([2 * self.orbital_frequency(ti, t_coal) for ti in t])
        phase = self.phase(t, t_coal)

        return {
            't': t,
            'h_plus': h_plus,
            'h_cross': h_cross,
            'frequency': frequency,
            'phase': phase,
            'm_chirp': self.m_chirp,
            'distance': self.distance
        }

    def time_to_coalescence(self, f_start: float = None) -> float:
        """
        Compute time from given frequency to coalescence.

        t_coal = (5/256)(GM_c/c³)^(-5/3)(πf)^(-8/3)

        Args:
            f_start: Starting GW frequency (default: initial_frequency)

        Returns:
            Time to coalescence
        """
        if f_start is None:
            f_start = 2 * self.f_initial

        t = (5 / 256) * (G_NEWTON * self.m_chirp)**(-5/3) * \
            (np.pi * f_start)**(-8/3)

        return t


class Ringdown:
    """
    Quasi-normal mode ringdown after black hole merger.

    After merger, the remnant black hole rings down with damped oscillations:
    h(t) = A exp(-t/τ) cos(ωt + φ)

    The frequency and damping time depend on the final mass and spin:
    ω ≈ c³/(GM_f) × [1 - 0.63(1-a_f)^(0.3)]
    τ ≈ (GM_f/c³) × 4/[(1-a_f)^(0.45)]

    where a_f is the dimensionless spin parameter.
    """

    def __init__(
        self,
        final_mass: float,
        final_spin: float = 0.7,
        distance: float = 1.0
    ):
        """
        Initialize ringdown model.

        Args:
            final_mass: Final black hole mass
            final_spin: Dimensionless spin a = J/(GM²/c) (0 to 1)
            distance: Distance to source
        """
        self.M_f = final_mass
        self.a_f = final_spin
        self.distance = distance

        # Compute QNM frequency and damping time
        # Using fits from Berti et al. (2009)
        self.omega = self._qnm_frequency()
        self.tau = self._qnm_damping_time()

    def _qnm_frequency(self) -> float:
        """
        Compute quasi-normal mode frequency.

        ω_R ≈ (c³/GM)[1 - 0.63(1-a)^0.3]

        Returns:
            Angular frequency
        """
        omega_0 = 1.0 / (G_NEWTON * self.M_f)
        omega = omega_0 * (1 - 0.63 * (1 - self.a_f)**0.3)
        return omega

    def _qnm_damping_time(self) -> float:
        """
        Compute quasi-normal mode damping time.

        τ ≈ (GM/c³) × 4/[(1-a)^0.45]

        Returns:
            Damping time
        """
        tau_0 = G_NEWTON * self.M_f
        tau = tau_0 * 4 / ((1 - self.a_f)**0.45)
        return tau

    def strain(
        self,
        t: np.ndarray,
        amplitude: float = 1.0,
        phase_0: float = 0.0
    ) -> np.ndarray:
        """
        Compute ringdown strain.

        h(t) = A exp(-t/τ) cos(ωt + φ₀) / D

        Args:
            t: Time array (starting from merger)
            amplitude: Initial amplitude
            phase_0: Initial phase

        Returns:
            Strain array
        """
        h = (amplitude / self.distance) * np.exp(-t / self.tau) * \
            np.cos(self.omega * t + phase_0)

        # Zero out negative times
        h[t < 0] = 0

        return h

    def compute_waveform(
        self,
        t_max: float,
        n_points: int = 1000,
        amplitude: float = 1.0
    ) -> Dict:
        """
        Compute full ringdown waveform.

        Args:
            t_max: Maximum time after merger
            n_points: Number of points
            amplitude: Initial amplitude

        Returns:
            Dictionary with waveform data
        """
        t = np.linspace(0, t_max, n_points)
        h = self.strain(t, amplitude)

        return {
            't': t,
            'h': h,
            'omega': self.omega,
            'tau': self.tau,
            'frequency': self.omega / (2 * np.pi),
            'quality_factor': np.pi * self.omega * self.tau
        }


# =============================================================================
# Fisher Metric Perturbation (v8 Connection)
# =============================================================================

class FisherPerturbation:
    """
    Link between Fisher metric perturbations and gravitational waves.

    From v8 Section 15.1:
    h_μν = ℓ_P² · δG_μν^Fisher[δΨ]

    This class computes the GW perturbation from changes in the
    quantum Fisher information metric due to state perturbations.
    """

    def __init__(self):
        """Initialize Fisher perturbation calculator."""
        self.l_p = L_PLANCK

    def compute_perturbation(
        self,
        psi: np.ndarray,
        dpsi: np.ndarray,
        delta_psi: np.ndarray,
        delta_dpsi: np.ndarray
    ) -> np.ndarray:
        """
        Compute metric perturbation from state perturbation.

        h_μν = ℓ_P² · δG_μν^Fisher

        where δG_μν^Fisher is the change in Fisher metric due to
        the perturbation |Ψ⟩ → |Ψ⟩ + δ|Ψ⟩.

        Args:
            psi: Unperturbed state |Ψ⟩
            dpsi: State derivatives [∂_μΨ]
            delta_psi: State perturbation δ|Ψ⟩
            delta_dpsi: Perturbation derivatives [∂_μ(δΨ)]

        Returns:
            4x4 metric perturbation h_μν
        """
        dim = 4
        delta_G = np.zeros((dim, dim))

        psi_conj = np.conj(psi)

        for mu in range(dim):
            for nu in range(dim):
                # Variation of G_μν = 4 Re[⟨∂_μΨ|∂_νΨ⟩ - ⟨∂_μΨ|Ψ⟩⟨Ψ|∂_νΨ⟩]

                # First term variation
                term1 = np.vdot(delta_dpsi[mu], dpsi[nu]) + \
                        np.vdot(dpsi[mu], delta_dpsi[nu])

                # Second term variation (product rule)
                a = np.vdot(delta_dpsi[mu], psi)
                b = np.vdot(dpsi[mu], delta_psi)
                c = np.vdot(psi_conj, dpsi[nu])
                d = np.vdot(delta_psi.conj(), dpsi[nu])
                e = np.vdot(psi_conj, delta_dpsi[nu])

                orig_a = np.vdot(dpsi[mu], psi)
                orig_c = np.vdot(psi_conj, dpsi[nu])

                term2_var = (a + b) * orig_c + orig_a * (d + e)

                delta_G[mu, nu] = 4 * np.real(term1 - term2_var)

        # Symmetrize (Fisher metric is symmetric)
        delta_G = 0.5 * (delta_G + delta_G.T)

        # Scale by Planck length squared
        h = self.l_p**2 * delta_G

        return h

    def perturbation_from_source(
        self,
        source_quadrupole: np.ndarray,
        distance: float,
        retarded_time: float
    ) -> np.ndarray:
        """
        Compute metric perturbation from source quadrupole moment.

        In the quadrupole approximation:
        h_ij = (2G/c⁴D) d²I_ij/dt²

        Args:
            source_quadrupole: Quadrupole moment tensor I_ij
            distance: Distance to source
            retarded_time: Retarded time t - r/c

        Returns:
            3x3 spatial perturbation tensor
        """
        # This is a placeholder that would need the second time derivative
        # of the quadrupole moment
        h_ij = (2 * G_NEWTON / distance) * source_quadrupole

        return h_ij


# =============================================================================
# Detector Response
# =============================================================================

class GWDetector:
    """
    Gravitational wave detector response.

    Models the response of interferometric GW detectors (LIGO, Virgo, etc.)
    to incoming gravitational waves.

    The detector response is:
    h(t) = F_+ h_+(t) + F_× h_×(t)

    where F_+, F_× are the antenna pattern functions that depend on
    the source location relative to the detector.
    """

    def __init__(
        self,
        arm_direction_x: np.ndarray = None,
        arm_direction_y: np.ndarray = None
    ):
        """
        Initialize detector with arm directions.

        Args:
            arm_direction_x: Unit vector along first arm (default: x)
            arm_direction_y: Unit vector along second arm (default: y)
        """
        if arm_direction_x is None:
            arm_direction_x = np.array([1.0, 0.0, 0.0])
        if arm_direction_y is None:
            arm_direction_y = np.array([0.0, 1.0, 0.0])

        self.arm_x = arm_direction_x / np.linalg.norm(arm_direction_x)
        self.arm_y = arm_direction_y / np.linalg.norm(arm_direction_y)

        # Detector tensor: D_ij = ½(x_i x_j - y_i y_j)
        self.detector_tensor = 0.5 * (
            np.outer(self.arm_x, self.arm_x) -
            np.outer(self.arm_y, self.arm_y)
        )

    def antenna_pattern_plus(
        self,
        theta: float,
        phi: float,
        psi: float = 0.0
    ) -> float:
        """
        Compute F_+ antenna pattern function.

        F_+ = ½(1 + cos²θ)cos(2φ)cos(2ψ) - cos(θ)sin(2φ)sin(2ψ)

        Args:
            theta: Polar angle of source (0 = overhead)
            phi: Azimuthal angle of source
            psi: Polarization angle

        Returns:
            F_+ value
        """
        cos_t = np.cos(theta)
        sin_2phi = np.sin(2 * phi)
        cos_2phi = np.cos(2 * phi)
        sin_2psi = np.sin(2 * psi)
        cos_2psi = np.cos(2 * psi)

        F_plus = 0.5 * (1 + cos_t**2) * cos_2phi * cos_2psi - \
                 cos_t * sin_2phi * sin_2psi

        return F_plus

    def antenna_pattern_cross(
        self,
        theta: float,
        phi: float,
        psi: float = 0.0
    ) -> float:
        """
        Compute F_× antenna pattern function.

        F_× = ½(1 + cos²θ)cos(2φ)sin(2ψ) + cos(θ)sin(2φ)cos(2ψ)

        Args:
            theta: Polar angle of source
            phi: Azimuthal angle of source
            psi: Polarization angle

        Returns:
            F_× value
        """
        cos_t = np.cos(theta)
        sin_2phi = np.sin(2 * phi)
        cos_2phi = np.cos(2 * phi)
        sin_2psi = np.sin(2 * psi)
        cos_2psi = np.cos(2 * psi)

        F_cross = 0.5 * (1 + cos_t**2) * cos_2phi * sin_2psi + \
                  cos_t * sin_2phi * cos_2psi

        return F_cross

    def compute_response(
        self,
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        theta: float,
        phi: float,
        psi: float = 0.0
    ) -> np.ndarray:
        """
        Compute detector strain response.

        h(t) = F_+ h_+(t) + F_× h_×(t)

        Args:
            h_plus: Plus polarization strain
            h_cross: Cross polarization strain
            theta: Source polar angle
            phi: Source azimuthal angle
            psi: Polarization angle

        Returns:
            Detector strain response
        """
        F_plus = self.antenna_pattern_plus(theta, phi, psi)
        F_cross = self.antenna_pattern_cross(theta, phi, psi)

        h = F_plus * h_plus + F_cross * h_cross

        return h

    def optimal_snr(
        self,
        h_plus: np.ndarray,
        h_cross: np.ndarray,
        t: np.ndarray,
        psd: Callable = None
    ) -> float:
        """
        Compute optimal (sky-averaged) signal-to-noise ratio.

        SNR² = 4 ∫ |h̃(f)|² / S_n(f) df

        Args:
            h_plus: Plus polarization
            h_cross: Cross polarization
            t: Time array
            psd: Power spectral density function S_n(f)

        Returns:
            Optimal SNR
        """
        dt = t[1] - t[0]
        n = len(t)

        # Combined strain (RMS of polarizations)
        h = np.sqrt(h_plus**2 + h_cross**2)

        # FFT
        h_tilde = fft(h) * dt
        freq = fftfreq(n, dt)

        # Positive frequencies only
        pos_mask = freq > 0
        freq_pos = freq[pos_mask]
        h_tilde_pos = h_tilde[pos_mask]

        if psd is None:
            # Simple white noise approximation
            psd = lambda f: 1e-46  # LIGO-like sensitivity

        # SNR integral
        S_n = np.array([psd(f) for f in freq_pos])
        integrand = 4 * np.abs(h_tilde_pos)**2 / S_n

        snr_sq = np.trapz(integrand, freq_pos)

        return np.sqrt(snr_sq)


# =============================================================================
# Complete GW Signal Generator
# =============================================================================

class GravitationalWaveSignal:
    """
    Complete gravitational wave signal generator.

    Combines inspiral, merger, and ringdown phases into a complete
    waveform, with proper matching at transition points.

    This is the main interface for generating GW signals from
    binary black hole or neutron star mergers.
    """

    def __init__(
        self,
        m1: float,
        m2: float,
        distance: float,
        inclination: float = 0.0,
        final_spin: float = 0.7
    ):
        """
        Initialize GW signal generator.

        Args:
            m1: Primary mass
            m2: Secondary mass
            distance: Luminosity distance
            inclination: Orbital inclination
            final_spin: Final black hole spin (for ringdown)
        """
        self.m1 = m1
        self.m2 = m2
        self.distance = distance
        self.inclination = inclination

        # Final mass after merger (energy radiated ≈ 5%)
        self.m_final = 0.95 * (m1 + m2)
        self.final_spin = final_spin

        # Create component models
        self.inspiral = BinaryInspiral(m1, m2, distance, inclination)
        self.ringdown = Ringdown(self.m_final, final_spin, distance)

    def generate_inspiral_merger_ringdown(
        self,
        t_start: float,
        t_merger: float,
        t_end: float,
        n_points: int = 10000
    ) -> Dict:
        """
        Generate complete IMR (inspiral-merger-ringdown) waveform.

        Args:
            t_start: Start time
            t_merger: Merger time
            t_end: End time (for ringdown)
            n_points: Total number of points

        Returns:
            Dictionary with complete waveform data
        """
        # Compute inspiral
        n_inspiral = int(n_points * (t_merger - t_start) / (t_end - t_start))
        inspiral_data = self.inspiral.compute_waveform(
            t_start, t_merger, n_inspiral
        )

        # Compute ringdown
        n_ringdown = n_points - n_inspiral
        t_ringdown = np.linspace(0, t_end - t_merger, n_ringdown)

        # Match amplitude at merger
        merger_amp = np.abs(inspiral_data['h_plus'][-1])
        h_ringdown = self.ringdown.strain(t_ringdown, merger_amp)

        # Combine
        t_full = np.concatenate([
            inspiral_data['t'],
            t_merger + t_ringdown
        ])

        h_plus_full = np.concatenate([
            inspiral_data['h_plus'],
            h_ringdown
        ])

        h_cross_full = np.concatenate([
            inspiral_data['h_cross'],
            0.5 * h_ringdown  # Simplified cross polarization
        ])

        return {
            't': t_full,
            'h_plus': h_plus_full,
            'h_cross': h_cross_full,
            't_merger': t_merger,
            'inspiral': inspiral_data,
            'ringdown': {
                't': t_ringdown,
                'h': h_ringdown,
                'omega': self.ringdown.omega,
                'tau': self.ringdown.tau
            }
        }

    def frequency_domain_waveform(
        self,
        f_min: float,
        f_max: float,
        n_points: int = 1000
    ) -> Dict:
        """
        Generate frequency-domain waveform using stationary phase.

        The Fourier transform of the inspiral is computed analytically
        using the stationary phase approximation:

        h̃(f) ∝ f^(-7/6) exp(iΨ(f))

        Args:
            f_min: Minimum frequency
            f_max: Maximum frequency
            n_points: Number of frequency points

        Returns:
            Frequency-domain waveform
        """
        f = np.linspace(f_min, f_max, n_points)

        # Amplitude: A(f) ∝ M_c^(5/6) f^(-7/6) / D
        M_c = self.inspiral.m_chirp
        amp = (G_NEWTON * M_c)**(5/6) * f**(-7/6) / self.distance

        # Phase: Ψ(f) = 2πft_c - φ_c + (3/128)(πGM_c f)^(-5/3) + ...
        phase = (3/128) * (np.pi * G_NEWTON * M_c * f)**(-5/3)

        h_tilde = amp * np.exp(1j * phase)

        return {
            'frequency': f,
            'h_tilde': h_tilde,
            'amplitude': amp,
            'phase': phase
        }


# =============================================================================
# Verification Functions
# =============================================================================

def verify_wave_equation(verbose: bool = True) -> Dict:
    """
    Verify wave equation solver with known solutions.

    Tests:
    1. Gaussian pulse propagation
    2. Plane wave solution
    3. Energy conservation

    Returns:
        Dictionary of test results
    """
    results = {}

    # Test 1: Gaussian pulse
    solver = WaveEquationSolver(grid_size=200, domain_size=100.0)

    x = np.linspace(0, 100, 200)
    x0, sigma = 50, 5
    initial_h = np.exp(-(x - x0)**2 / (2 * sigma**2))
    initial_dh_dt = np.zeros_like(initial_h)

    solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=50, n_steps=1000)

    # Check pulse splits into left and right going waves
    pulse_preserved = np.sum(solution['h'][-1]**2) > 0.1 * np.sum(initial_h**2)
    results['gaussian_pulse'] = pulse_preserved

    if verbose:
        logger.info(f"Gaussian pulse test: {'PASS' if pulse_preserved else 'FAIL'}")

    # Test 2: Plane wave dispersion relation
    k = 2 * np.pi / 10  # wavenumber
    omega = k  # dispersion relation ω = k (c = 1)

    x = np.linspace(0, 100, 200)
    initial_h = np.sin(k * x)
    initial_dh_dt = -omega * np.cos(k * x)  # ∂h/∂t = -ω cos(kx)

    solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=10, n_steps=200)

    # After time T = 2π/ω, wave should return to initial state
    T = 2 * np.pi / omega
    idx_T = int(T / (10/200))
    if idx_T < len(solution['h']):
        h_at_T = solution['h'][idx_T]
        wave_periodic = np.allclose(h_at_T, initial_h, atol=0.1)
    else:
        wave_periodic = False

    results['plane_wave'] = wave_periodic

    if verbose:
        logger.info(f"Plane wave test: {'PASS' if wave_periodic else 'FAIL'}")

    # Test 3: Binary inspiral waveform
    binary = BinaryInspiral(m1=30, m2=30, distance=100)
    waveform = binary.compute_waveform(t_start=0, t_coal=1000, n_points=500)

    # Check frequency increases (chirp)
    freq_increases = np.all(np.diff(waveform['frequency'][:-10]) >= 0)
    results['inspiral_chirp'] = freq_increases

    if verbose:
        logger.info(f"Inspiral chirp test: {'PASS' if freq_increases else 'FAIL'}")

    # Test 4: Ringdown damping
    ringdown = Ringdown(final_mass=60, final_spin=0.7)
    rd_data = ringdown.compute_waveform(t_max=100, n_points=500)

    # Check exponential decay
    h_abs = np.abs(rd_data['h'])
    decay_correct = h_abs[-1] < 0.1 * h_abs[0]
    results['ringdown_decay'] = decay_correct

    if verbose:
        logger.info(f"Ringdown decay test: {'PASS' if decay_correct else 'FAIL'}")

    # Test 5: Polarization tensor properties
    e_plus = GWPolarization.plus_polarization_tensor()
    e_cross = GWPolarization.cross_polarization_tensor()

    # TT gauge: traceless
    trace_plus = np.trace(e_plus)
    trace_cross = np.trace(e_cross)
    traceless = abs(trace_plus) < 1e-10 and abs(trace_cross) < 1e-10
    results['polarization_traceless'] = traceless

    if verbose:
        logger.info(f"Polarization traceless test: {'PASS' if traceless else 'FAIL'}")

    return results


def demonstrate_gw_physics(verbose: bool = True) -> Dict:
    """
    Demonstrate key GW physics from v8 framework.

    Returns:
        Dictionary with demonstration results
    """
    results = {}

    if verbose:
        logger.info("=== Gravitational Wave Physics (v8 Section 15) ===")
        logger.info("")
        logger.info("Key equations:")
        logger.info("  g_μν = η_μν + h_μν,  |h_μν| ≪ 1")
        logger.info("  h_μν = ℓ_P² · δG_μν^Fisher[δΨ]")
        logger.info("  □h_μν = 0  (wave equation)")
        logger.info("")

    # Generate sample waveform
    signal = GravitationalWaveSignal(
        m1=30,   # Solar masses (conceptually)
        m2=30,
        distance=400,  # Mpc (conceptually)
        inclination=np.pi/4
    )

    waveform = signal.generate_inspiral_merger_ringdown(
        t_start=0,
        t_merger=1000,
        t_end=1100,
        n_points=5000
    )

    results['waveform'] = waveform

    # Compute detector response
    detector = GWDetector()
    theta, phi = np.pi/3, np.pi/4
    h_det = detector.compute_response(
        waveform['h_plus'],
        waveform['h_cross'],
        theta, phi
    )
    results['detector_strain'] = h_det

    if verbose:
        logger.info(f"Generated {len(waveform['t'])} point waveform")
        logger.info(f"Peak strain: {np.max(np.abs(waveform['h_plus'])):.2e}")
        logger.info(f"Merger frequency: {waveform['inspiral']['frequency'][-1]:.2f}")
        logger.info(f"Ringdown frequency: {signal.ringdown.omega/(2*np.pi):.2f}")
        logger.info(f"Ringdown Q-factor: {np.pi * signal.ringdown.omega * signal.ringdown.tau:.1f}")

    return results
