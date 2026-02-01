"""
Entanglement Geometry Handler for Quantum Gravity v7

Implements the unified entanglement-geometry relation using the v7 formulation:

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

where E_μν is the entanglement strain tensor derived from entropy gradients.

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v7
"""

import numpy as np
from constants import CONSTANTS, coherence_length
from physics.fisher_metric import FisherMetric, compute_v7_spacetime_metric
from physics.entanglement_strain import (
    EntanglementStrainTensor,
    compute_dark_matter_ratio,
    compute_entanglement_susceptibility
)


class EntanglementGeometryHandler:
    """
    Handle unified entanglement-geometry relation for v7 formulation.

    The v7 formulation replaces the old dS² = dE² + γ²dI² master equation
    with the Fisher Information + Entanglement Strain formulation.

    Attributes:
        gamma_0: Immirzi parameter (0.274)
        l_p: Planck length
        fisher: Fisher information metric calculator
        strain: Entanglement strain tensor calculator
    """

    def __init__(self, gamma_0: float = None):
        """
        Initialize entanglement geometry handler.

        Args:
            gamma_0: Immirzi parameter (default: from CONSTANTS)
        """
        self.gamma_0 = gamma_0 if gamma_0 is not None else CONSTANTS['gamma_0']
        self.l_p = CONSTANTS['l_p']

        # v7 core components
        self.fisher = FisherMetric()
        self.strain = EntanglementStrainTensor()

    def compute_effective_coupling(self) -> float:
        """
        Compute effective coupling from v7 formulation.

        In v7, the effective coupling is the Immirzi parameter γ₀.

        Returns:
            Effective coupling (dimensionless)
        """
        return self.gamma_0

    def compute_spacetime_metric(self, state,
                                  position: np.ndarray = None) -> np.ndarray:
        """
        Compute emergent spacetime metric using v7 master equation.

        g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

        This replaces the old dS² = dE² + γ²dI² relation.

        Args:
            state: Quantum state object
            position: Spacetime position (optional)

        Returns:
            4x4 spacetime metric tensor
        """
        # Compute Fisher metric
        G_fisher = self.fisher.compute_full_metric(state)

        # Compute entanglement strain tensor
        E_strain = self.strain.compute_strain(state, position)

        # Apply v7 master equation
        g = compute_v7_spacetime_metric(G_fisher, E_strain, self.gamma_0)

        return g

    def compute_spacetime_interval(self, entanglement: float,
                                    information: float) -> float:
        """
        Compute spacetime interval for backward compatibility.

        Note: This is a simplified interface. For full v7 calculations,
        use compute_spacetime_metric() instead.

        In v7, this maps to:
            ds² ~ ℓ_P² × (information + γ₀ × entanglement)

        Args:
            entanglement: Entanglement measure (entropy-related)
            information: Information metric (Fisher-related)

        Returns:
            Spacetime interval ds²
        """
        # v7 approximation: combine Fisher (information) and strain (entanglement)
        l_p_sq = self.l_p**2
        ds2 = l_p_sq * (information + self.gamma_0 * entanglement)
        return ds2

    def compute_information_metric(self, spacetime_interval: float,
                                    entanglement: float) -> float:
        """
        Compute information metric from spacetime interval.

        Solves for information from:
            ds² = ℓ_P² × (information + γ₀ × entanglement)

        Args:
            spacetime_interval: Interval ds²
            entanglement: Entanglement measure

        Returns:
            Information metric component
        """
        l_p_sq = self.l_p**2
        information = (spacetime_interval / l_p_sq) - self.gamma_0 * entanglement
        return max(information, 0)  # Ensure non-negative

    def compute_entanglement_contribution(self, state,
                                           position: np.ndarray = None) -> float:
        """
        Compute the entanglement contribution to geometry.

        This computes the trace of γ₀ × E_μν.

        Args:
            state: Quantum state object
            position: Spacetime position

        Returns:
            Entanglement contribution (scalar)
        """
        E_strain = self.strain.compute_strain(state, position)
        # Trace with Minkowski metric
        eta = np.diag([-1, 1, 1, 1])
        trace = np.sum(eta * E_strain)
        return self.gamma_0 * trace

    def compute_fisher_contribution(self, state) -> float:
        """
        Compute the Fisher information contribution to geometry.

        This computes the trace of G_μν^Fisher.

        Args:
            state: Quantum state object

        Returns:
            Fisher contribution (scalar)
        """
        G_fisher = self.fisher.compute_full_metric(state)
        # Trace with Minkowski metric
        eta = np.diag([-1, 1, 1, 1])
        trace = np.sum(eta * G_fisher)
        return trace

    def compute_schwarzschild_interval(self, r: float, M: float,
                                        dr: float = 0, dt: float = 1) -> float:
        """
        Compute spacetime interval for Schwarzschild-like geometry.

        Uses the v7 metric with coherence length corrections.

        Args:
            r: Radial coordinate (in l_p units)
            M: Mass (in m_p units)
            dr: Radial displacement
            dt: Time displacement

        Returns:
            Spacetime interval ds²
        """
        r_s = 2 * CONSTANTS['G'] * M
        sigma = coherence_length(r, r_s)

        # Compute v7 metric components
        G_fisher = self.fisher.compute_schwarzschild_fisher(r, M, sigma)
        E_strain = self.strain.compute_schwarzschild_strain(r, M)

        g = compute_v7_spacetime_metric(G_fisher, E_strain, self.gamma_0)

        # Compute interval
        ds2 = g[0, 0] * dt**2 + g[1, 1] * dr**2
        return ds2

    def compute_dark_matter_ratio(self) -> float:
        """
        Compute v7 dark matter to baryonic matter ratio.

        M_DM/M_b = π/(2γ₀) ≈ 5.73

        Returns:
            Dark matter ratio
        """
        return compute_dark_matter_ratio()

    def compute_susceptibility(self, r: float, r_0: float,
                               S_ent: float, S_BH: float) -> float:
        """
        Compute entanglement susceptibility for rotation curves.

        χ_E(r) = γ₀ × (S_ent/S_BH) × (1 - exp(-r/r_0))

        Args:
            r: Galactic radius
            r_0: Core radius
            S_ent: Entanglement entropy at r
            S_BH: Reference black hole entropy

        Returns:
            Entanglement susceptibility
        """
        return compute_entanglement_susceptibility(r, r_0, S_ent, S_BH)
