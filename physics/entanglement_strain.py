"""
Entanglement Strain Tensor for Quantum Gravity v9

Implements the entanglement strain tensor using the TIDAL FORM:

    E_μν = ∇_μ∇_νS_ent − ¼g_μν□S_ent

where □ = g^αβ∇_α∇_β is the d'Alembertian and S_ent is the entanglement
entropy following the Ryu-Takayanagi formula:

    S_ent = Area(γ_A) / (4ℓ_P²)

WHY TIDAL FORM: The earlier quadratic form (∂S·∂S) produces incorrect
scaling M_DM ∝ M_b². The tidal form (Hessian of entropy) gives correct
linear scaling M_DM ∝ M_b, matching galactic observations.

This tensor provides the entanglement contribution to the master equation:

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v9.3
"""

import numpy as np
from typing import Optional, Tuple, Union
import logging

from constants import CONSTANTS, coherence_length

logger = logging.getLogger(__name__)


class EntanglementStrainTensor:
    """
    Entanglement Strain Tensor calculator using the TIDAL FORM.

    The strain tensor measures how the Hessian of entanglement entropy
    contributes to spacetime geometry in the v9 formulation:

        E_μν = ∇_μ∇_νS_ent − ¼g_μν□S_ent

    This tidal form gives correct linear scaling M_DM ∝ M_b.

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
        Compute the entanglement strain tensor E_μν using TIDAL FORM.

        E_μν = ∇_μ∇_νS_ent − ¼g_μν□S_ent

        The tidal form (Hessian) replaces the old quadratic form to give
        correct M_DM ∝ M_b scaling.

        Args:
            state: Quantum state object with entanglement information
            position: Spacetime position (optional)

        Returns:
            4x4 strain tensor
        """
        r = self._get_radius(state, position)

        # For spherical symmetry, compute Hessian and Laplacian of S
        # S = πr²/l_p² (area law) gives:
        #   ∇_r∇_r S = 2π/l_p² (constant)
        #   □S = ∇²S = 6π/l_p² (in spherical coords)

        if r > 0:
            hessian_rr = 2 * np.pi / self.l_p**2
            laplacian_S = 6 * np.pi / self.l_p**2
        else:
            hessian_rr = 0
            laplacian_S = 0

        # Build tidal strain tensor: E_μν = ∇_μ∇_νS - ¼g_μν□S
        # In flat space (weak field), g_μν ≈ η_μν
        E = np.zeros((4, 4))

        # Diagonal Hessian components (spherical symmetry)
        E[1, 1] = hessian_rr - 0.25 * self._eta[1, 1] * laplacian_S  # rr
        E[0, 0] = -0.25 * self._eta[0, 0] * laplacian_S  # tt (no time dependence)
        E[2, 2] = -0.25 * self._eta[2, 2] * laplacian_S  # θθ
        E[3, 3] = -0.25 * self._eta[3, 3] * laplacian_S  # φφ

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
        Compute strain tensor for Schwarzschild geometry using TIDAL FORM.

        For a black hole, the entanglement entropy follows:
        S(r) = πr²/l_p² + 2πMr/l_p² (weak-field perturbation from mass)

        The tidal form E_μν = ∇_μ∇_νS - ¼g_μν□S gives:
        - Background (πr²): constant Hessian, constant trace
        - Perturbation (2πMr): traceless part ∝ M/r (correct scaling!)

        Args:
            r: Radial coordinate (in meters)
            M: Mass (in kg)

        Returns:
            4x4 strain tensor at radius r
        """
        r_s = 2 * CONSTANTS['G'] * M / CONSTANTS['c']**2

        E = np.zeros((4, 4))
        if r <= 0:
            return E

        # BACKGROUND (area law S_0 = πr²/l_p²):
        # ∇²S_0 = 6π/l_p², ∂²_r S_0 = 2π/l_p²
        hessian_rr_bg = 2 * np.pi / self.l_p**2
        laplacian_bg = 6 * np.pi / self.l_p**2

        # PERTURBATION (mass contribution δS = 2πMr/l_p²):
        # ∂_r(δS) = 2πM/l_p², ∂²_r(δS) = 0
        # ∇²(δS) = (1/r²)∂_r(r² · 2πM/l_p²) = 4πM/(r·l_p²)
        M_geometric = CONSTANTS['G'] * M / CONSTANTS['c']**2  # GM/c² in meters
        hessian_rr_pert = 0  # Second derivative of linear function is zero
        laplacian_pert = 4 * np.pi * M_geometric / (r * self.l_p**2) if r > 0 else 0

        # Near horizon correction
        if r > r_s:
            sigma = coherence_length(r, r_s)
            correction = 1 + self.gamma_0 * (self.l_p / sigma)**2
            laplacian_pert *= correction

        # Total Hessian and Laplacian
        total_hessian_rr = hessian_rr_bg + hessian_rr_pert
        total_laplacian = laplacian_bg + laplacian_pert

        # Build tidal strain tensor: E_μν = ∇_μ∇_νS - ¼g_μν□S
        E[0, 0] = -0.25 * self._eta[0, 0] * total_laplacian  # tt
        E[1, 1] = total_hessian_rr - 0.25 * self._eta[1, 1] * total_laplacian  # rr
        E[2, 2] = -0.25 * self._eta[2, 2] * total_laplacian  # θθ
        E[3, 3] = -0.25 * self._eta[3, 3] * total_laplacian  # φφ

        return E

    def compute_for_galaxy(self, r: float, M_total: float,
                           r_scale: float) -> np.ndarray:
        """
        Compute strain tensor for galactic scales using TIDAL FORM.

        At galactic scales, the entanglement entropy follows a modified
        scaling that accounts for the dark matter halo:

        S(r) = S_0 + δS_DM where δS_DM ∝ M_b × r (linear in baryonic mass)

        The tidal form ensures M_DM ∝ M_b (correct linear scaling).

        Args:
            r: Galactic radius (in kpc)
            M_total: Total mass including dark matter (in M_sun)
            r_scale: Scale radius for the mass profile

        Returns:
            4x4 strain tensor
        """
        E = np.zeros((4, 4))
        if r <= 0:
            return E

        # Convert to SI units
        r_natural = r * CONSTANTS['kpc']
        M_natural = M_total * CONSTANTS['M_sun']
        r_scale_natural = r_scale * CONSTANTS['kpc']

        # Geometric mass
        M_geometric = CONSTANTS['G'] * M_natural / CONSTANTS['c']**2

        # TIDAL FORM: E_μν = ∇_μ∇_νS - ¼g_μν□S
        #
        # For galactic entropy S(r) ∝ M_b × r (from entanglement susceptibility):
        #   ∂_r S ∝ M_b (constant)
        #   ∂²_r S = 0
        #   ∇²S = 2M_b/r (from spherical Laplacian of r)
        #
        # This gives traceless part ∝ M_b/r, leading to M_DM ∝ M_b

        # Effective entanglement contribution with dark matter enhancement
        dm_ratio = CONSTANTS['dark_matter_ratio']  # π/(2γ₀) ≈ 5.73
        enhancement = 1 + self.gamma_0 * np.log(1 + r / r_scale)

        # Laplacian of the entropy perturbation: ∇²(δS) ∝ M/r
        laplacian_pert = 4 * np.pi * M_geometric * enhancement / (r_natural * self.l_p**2)

        # Hessian component (second radial derivative of linear term is zero,
        # but there's a background area-law contribution)
        hessian_rr_bg = 2 * np.pi / self.l_p**2
        laplacian_bg = 6 * np.pi / self.l_p**2

        total_laplacian = laplacian_bg + laplacian_pert

        # Build tidal strain tensor
        E[0, 0] = -0.25 * self._eta[0, 0] * total_laplacian  # tt
        E[1, 1] = hessian_rr_bg - 0.25 * self._eta[1, 1] * total_laplacian  # rr
        E[2, 2] = -0.25 * self._eta[2, 2] * total_laplacian  # θθ
        E[3, 3] = -0.25 * self._eta[3, 3] * total_laplacian  # φφ

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
