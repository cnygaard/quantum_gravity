"""
Fisher Information Metric for Quantum Gravity v7

Implements the quantum Fisher information metric which measures the
distinguishability of nearby quantum states:

    G_μν^Fisher = 4 Re[⟨∂_μΨ|∂_νΨ⟩ - ⟨∂_μΨ|Ψ⟩⟨Ψ|∂_νΨ⟩]

This metric encodes how spacetime geometry emerges from quantum information
geometry according to the v7 master equation:

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v7
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

from constants import CONSTANTS, coherence_length

logger = logging.getLogger(__name__)


class FisherMetric:
    """
    Quantum Fisher Information Metric calculator.

    The Fisher metric measures the distinguishability of quantum states
    and provides the geometric term in the v7 master equation.

    Attributes:
        l_p: Planck length
        gamma_0: Immirzi parameter (0.274)
        epsilon: Step size for numerical derivatives
    """

    def __init__(self, epsilon: float = 1e-6):
        """
        Initialize Fisher metric calculator.

        Args:
            epsilon: Step size for numerical differentiation
        """
        self.l_p = CONSTANTS['l_p']
        self.gamma_0 = CONSTANTS['gamma_0']
        self.epsilon = epsilon
        self._cache = {}

    def compute_metric(self, state, mu: int, nu: int) -> float:
        """
        Compute Fisher metric component G_μν^Fisher.

        G_μν^Fisher = 4 Re[⟨∂_μΨ|∂_νΨ⟩ - ⟨∂_μΨ|Ψ⟩⟨Ψ|∂_νΨ⟩]

        Args:
            state: Quantum state object with wavefunction data
            mu: First spacetime index (0=t, 1=r, 2=θ, 3=φ)
            nu: Second spacetime index

        Returns:
            Fisher metric component (dimensionless, scaled by l_p^-2)
        """
        # Get state derivatives
        dpsi_mu = self._compute_state_derivative(state, mu)
        dpsi_nu = self._compute_state_derivative(state, nu)
        psi = self._get_state_vector(state)

        # Compute inner products
        # ⟨∂_μΨ|∂_νΨ⟩
        term1 = np.vdot(dpsi_mu, dpsi_nu)

        # ⟨∂_μΨ|Ψ⟩⟨Ψ|∂_νΨ⟩
        inner_mu = np.vdot(dpsi_mu, psi)
        inner_nu = np.vdot(psi, dpsi_nu)
        term2 = inner_mu * inner_nu

        # G_μν = 4 Re[term1 - term2]
        G_munu = 4.0 * np.real(term1 - term2)

        return G_munu

    def compute_full_metric(self, state) -> np.ndarray:
        """
        Compute full 4x4 Fisher metric tensor.

        Args:
            state: Quantum state object

        Returns:
            4x4 numpy array of Fisher metric components
        """
        G = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(mu, 4):
                G[mu, nu] = self.compute_metric(state, mu, nu)
                if mu != nu:
                    G[nu, mu] = G[mu, nu]  # Symmetric tensor
        return G

    def compute_schwarzschild_fisher(self, r: float, M: float,
                                      sigma: Optional[float] = None) -> np.ndarray:
        """
        Compute Fisher metric for Schwarzschild-like geometry.

        For a coherent state with position uncertainty σ(r), the Fisher
        metric components are:

        G_tt = (4/ℏ²)⟨(ΔE)²⟩ ∝ 1/(1 - r_s/r)  (thermal energy fluctuations)
        G_rr = (1/σ²)[1 + 2σ²(d ln σ/dr)²]    (position distinguishability)

        Args:
            r: Radial coordinate (in l_p units)
            M: Mass (in m_p units)
            sigma: Coherence length (computed if None)

        Returns:
            4x4 diagonal Fisher metric tensor
        """
        # Schwarzschild radius
        r_s = 2 * CONSTANTS['G'] * M

        # Coherence length from Tolman-Ehrenfest
        if sigma is None:
            sigma = coherence_length(r, r_s)

        # Avoid division by zero
        sigma = max(sigma, self.l_p)

        # Metric components
        G = np.zeros((4, 4))

        # Time-time component: thermal energy fluctuations
        # G_tt ∝ 1/(1 - r_s/r) from T(r)² scaling
        redshift_factor = max(1 - r_s / r, 1e-10)
        G[0, 0] = 1.0 / (sigma**2 * redshift_factor)

        # Radial component: position + width distinguishability
        # G_rr = (1/σ²)[1 + 2σ²(d ln σ/dr)²]
        if r > r_s:
            # d ln σ/dr = (1/2) * (r_s/r²) / (1 - r_s/r)
            dlnsigma_dr = 0.5 * (r_s / r**2) / redshift_factor
            G[1, 1] = (1.0 / sigma**2) * (1 + 2 * sigma**2 * dlnsigma_dr**2)
        else:
            G[1, 1] = 1.0 / sigma**2

        # Angular components (simplified spherical symmetry)
        G[2, 2] = 1.0 / (sigma**2 * r**2) if r > 0 else 0
        G[3, 3] = G[2, 2]  # Spherical symmetry

        return G

    def _compute_state_derivative(self, state, direction: int) -> np.ndarray:
        """
        Compute partial derivative of state in given direction.

        Uses central difference for numerical stability:
        ∂_μΨ ≈ (Ψ(x + εe_μ) - Ψ(x - εe_μ)) / (2ε)

        Args:
            state: Quantum state object
            direction: Spacetime direction (0-3)

        Returns:
            Array of state derivative values
        """
        psi = self._get_state_vector(state)

        # For states without explicit coordinate dependence,
        # use the coefficients as the state vector
        if hasattr(state, 'grid') and state.grid is not None:
            # Numerical differentiation on grid
            return self._numerical_derivative(state, direction)
        else:
            # Return zero for static states (no coordinate dependence)
            return np.zeros_like(psi)

    def _numerical_derivative(self, state, direction: int) -> np.ndarray:
        """
        Numerical derivative using grid structure.

        Args:
            state: State with grid attribute
            direction: Direction for derivative

        Returns:
            Derivative array
        """
        psi = self._get_state_vector(state)
        n = len(psi)
        dpsi = np.zeros_like(psi, dtype=complex)

        # Use grid spacing for derivative
        if hasattr(state, 'grid') and hasattr(state.grid, 'points'):
            points = state.grid.points
            if len(points) > 1:
                # Central difference where possible
                for i in range(1, n - 1):
                    if direction < len(points[i]):
                        dx = points[i + 1][direction] - points[i - 1][direction]
                        if abs(dx) > 1e-15:
                            dpsi[i] = (psi[i + 1] - psi[i - 1]) / dx

        return dpsi

    def _get_state_vector(self, state) -> np.ndarray:
        """
        Extract state vector from state object.

        Args:
            state: Quantum state object

        Returns:
            Complex array of state coefficients
        """
        if hasattr(state, 'coefficients'):
            return np.asarray(state.coefficients, dtype=complex)
        elif hasattr(state, 'psi'):
            return np.asarray(state.psi, dtype=complex)
        elif hasattr(state, 'wavefunction'):
            return np.asarray(state.wavefunction, dtype=complex)
        else:
            # Fallback: create minimal state
            return np.array([1.0 + 0j])

    def clear_cache(self):
        """Clear the metric computation cache."""
        self._cache = {}


def compute_v7_spacetime_metric(fisher_metric: np.ndarray,
                                 strain_tensor: np.ndarray,
                                 gamma_0: Optional[float] = None) -> np.ndarray:
    """
    Compute the emergent spacetime metric from v7 master equation.

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

    Args:
        fisher_metric: 4x4 Fisher information metric
        strain_tensor: 4x4 entanglement strain tensor
        gamma_0: Immirzi parameter (default: 0.274)

    Returns:
        4x4 emergent spacetime metric
    """
    if gamma_0 is None:
        gamma_0 = CONSTANTS['gamma_0']

    l_p_sq = CONSTANTS['l_p']**2

    g = l_p_sq * (fisher_metric + gamma_0 * strain_tensor)

    return g
