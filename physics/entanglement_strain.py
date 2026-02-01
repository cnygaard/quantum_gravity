"""
Entanglement Strain Tensor for Quantum Gravity v7

Implements the entanglement strain tensor which measures how entanglement
entropy varies across spacetime:

    E_μν = ∂_μS_ent ∂_νS_ent - ½η_μν(∂S_ent)²

where S_ent is the entanglement entropy following the Ryu-Takayanagi formula:

    S_ent = Area(γ_A) / (4ℓ_P²)

This tensor provides the entanglement contribution to the v7 master equation:

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v7
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

from constants import CONSTANTS, coherence_length

logger = logging.getLogger(__name__)


class EntanglementStrainTensor:
    """
    Entanglement Strain Tensor calculator.

    The strain tensor measures how entanglement entropy gradients
    contribute to spacetime geometry in the v7 formulation.

    Attributes:
        l_p: Planck length
        gamma_0: Immirzi parameter
    """

    def __init__(self):
        """Initialize the entanglement strain tensor calculator."""
        self.l_p = CONSTANTS['l_p']
        self.gamma_0 = CONSTANTS['gamma_0']
        self._eta = np.diag([-1, 1, 1, 1])  # Minkowski metric

    def compute_strain(self, state, position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the entanglement strain tensor E_μν.

        E_μν = ∂_μS_ent ∂_νS_ent - ½η_μν(∂S_ent)²

        Args:
            state: Quantum state object with entanglement information
            position: Spacetime position (optional)

        Returns:
            4x4 strain tensor
        """
        # Compute entropy gradient
        dS = self.compute_entropy_gradient(state, position)

        # (∂S)² = η^μν ∂_μS ∂_νS (with raised indices)
        dS_squared = -dS[0]**2 + dS[1]**2 + dS[2]**2 + dS[3]**2

        # Build strain tensor
        E = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                E[mu, nu] = dS[mu] * dS[nu] - 0.5 * self._eta[mu, nu] * dS_squared

        return E

    def compute_entropy_gradient(self, state,
                                  position: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute the gradient of entanglement entropy.

        For a spherically symmetric system, the entropy follows the
        area law S = A/(4l_p²), giving:

        ∂_r S = (2πr)/(l_p²)  (radial gradient)

        Args:
            state: Quantum state object
            position: Spacetime position [t, r, θ, φ]

        Returns:
            4-vector of entropy gradients [∂_t S, ∂_r S, ∂_θ S, ∂_φ S]
        """
        dS = np.zeros(4)

        # Extract radius from state or position
        r = self._get_radius(state, position)

        if r > 0:
            # Area law: S = A/(4l_p²) = πr²/l_p²
            # ∂_r S = 2πr/l_p²
            dS[1] = 2 * np.pi * r / self.l_p**2

            # Time derivative (for evolving systems)
            if hasattr(state, 'entropy_rate'):
                dS[0] = state.entropy_rate
            elif hasattr(state, 'dS_dt'):
                dS[0] = state.dS_dt

            # Angular derivatives are zero for spherical symmetry
            # but could be non-zero for more general geometries

        return dS

    def compute_entropy(self, state) -> float:
        """
        Compute entanglement entropy from area law.

        S_ent = Area / (4ℓ_P²)

        Args:
            state: Quantum state with characteristic size

        Returns:
            Entanglement entropy (dimensionless)
        """
        r = self._get_characteristic_radius(state)

        if r > 0:
            # Bekenstein-Hawking / Ryu-Takayanagi: S = A/(4l_p²)
            area = 4 * np.pi * r**2
            S = area / (4 * self.l_p**2)
            return S

        return 0.0

    def compute_schwarzschild_strain(self, r: float, M: float) -> np.ndarray:
        """
        Compute strain tensor for Schwarzschild geometry.

        Near a black hole, the entanglement entropy follows:
        S(r) = πr²/l_p² + O(log corrections)

        Args:
            r: Radial coordinate (in l_p units)
            M: Mass (in m_p units)

        Returns:
            4x4 strain tensor at radius r
        """
        r_s = 2 * CONSTANTS['G'] * M

        # Entropy gradient: ∂_r S = 2πr/l_p²
        dS_r = 2 * np.pi * r / self.l_p**2 if r > 0 else 0

        # Near horizon correction
        if r > r_s and r > 0:
            # Include quantum correction from coherence length
            sigma = coherence_length(r, r_s)
            correction = 1 + self.gamma_0 * (self.l_p / sigma)**2
            dS_r *= correction

        dS = np.array([0, dS_r, 0, 0])

        # (∂S)² with Minkowski signature
        dS_squared = dS_r**2  # Only radial component

        # Strain tensor
        E = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                E[mu, nu] = dS[mu] * dS[nu] - 0.5 * self._eta[mu, nu] * dS_squared

        return E

    def compute_for_galaxy(self, r: float, M_total: float,
                           r_scale: float) -> np.ndarray:
        """
        Compute strain tensor for galactic scales.

        At galactic scales, the entanglement entropy is dominated by
        the dark matter distribution and follows a modified scaling.

        Args:
            r: Galactic radius (in kpc)
            M_total: Total mass including dark matter (in M_sun)
            r_scale: Scale radius for the mass profile

        Returns:
            4x4 strain tensor
        """
        # Convert to natural units
        r_natural = r * CONSTANTS['kpc']
        M_natural = M_total * CONSTANTS['M_sun']

        # Galactic entropy scaling (different from black holes)
        # S ~ log(r/l_p) at large scales
        if r > 0:
            dS_r = 1.0 / r_natural  # ∂_r(log r) = 1/r

            # Apply dark matter enhancement
            dm_ratio = CONSTANTS['dark_matter_ratio']
            enhancement = 1 + self.gamma_0 * np.log(1 + r / r_scale)
            dS_r *= enhancement
        else:
            dS_r = 0

        dS = np.array([0, dS_r, 0, 0])
        dS_squared = dS_r**2

        E = np.zeros((4, 4))
        for mu in range(4):
            for nu in range(4):
                E[mu, nu] = dS[mu] * dS[nu] - 0.5 * self._eta[mu, nu] * dS_squared

        return E

    def _get_radius(self, state, position: Optional[np.ndarray]) -> float:
        """Extract radius from state or position."""
        if position is not None and len(position) > 1:
            return abs(position[1])

        return self._get_characteristic_radius(state)

    def _get_characteristic_radius(self, state) -> float:
        """Get characteristic radius from state object."""
        if hasattr(state, 'horizon_radius'):
            return state.horizon_radius
        elif hasattr(state, 'r_h'):
            return state.r_h
        elif hasattr(state, 'radius'):
            return state.radius
        elif hasattr(state, 'characteristic_radius'):
            return state.characteristic_radius
        elif hasattr(state, 'mass'):
            # For black holes: r_h = 2GM
            return 2 * CONSTANTS['G'] * state.mass
        else:
            return 1.0  # Default to Planck scale


def compute_dark_matter_ratio() -> float:
    """
    Compute the predicted dark matter to baryonic matter ratio.

    From v7 theory:
        M_DM/M_b = π/(2γ₀) ≈ 5.73

    where γ₀ = 0.274 is the Immirzi parameter.

    Returns:
        Dark matter ratio (dimensionless)
    """
    gamma_0 = CONSTANTS['gamma_0']
    xi = CONSTANTS['xi_geometric']  # π/2

    ratio = xi / gamma_0
    return ratio


def compute_entanglement_susceptibility(r: float, r_0: float,
                                         S_ent: float, S_BH: float) -> float:
    """
    Compute entanglement susceptibility for rotation curves.

    χ_E(r) = γ₀ × (S_ent/S_BH) × (1 - exp(-r/r_0))

    This describes how entanglement contributes to effective mass.

    Args:
        r: Galactic radius
        r_0: Core radius (scale length)
        S_ent: Entanglement entropy at radius r
        S_BH: Reference black hole entropy (for normalization)

    Returns:
        Entanglement susceptibility (dimensionless)
    """
    gamma_0 = CONSTANTS['gamma_0']

    if S_BH > 0:
        chi = gamma_0 * (S_ent / S_BH) * (1 - np.exp(-r / r_0))
    else:
        chi = 0

    return chi
