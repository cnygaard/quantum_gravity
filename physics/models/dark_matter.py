import numpy as np
from constants import CONSTANTS, SI_UNITS, geometric_factor_ds, dm_ratio_v8 as compute_dm_ratio_v8
from physics.quantum_geometry import QuantumGeometry

# v7/v8 Holographic Fisher Geometry parameters
# γ₀ = 0.274 (Immirzi parameter from LQG, Engle-Noui-Perez 2010, SU(2))
#
# v7 flat-space prediction: M_DM/M_b = π/(2γ₀) ≈ 5.73 (7.4σ from Planck)
# v8 de Sitter corrected:   M_DM/M_b = (π/2γ₀)(sin√Ω_m)/√Ω_m ≈ 5.43 (1.4σ from Planck)
#
# The improvement comes from accounting for the positive cosmological constant Λ > 0
# which curves the information geometry into S³, reducing the holographic path ratio.
GAMMA_0 = CONSTANTS.get('gamma_0', 0.274)
DM_RATIO_V7 = CONSTANTS.get('dark_matter_ratio', np.pi / (2 * GAMMA_0))  # 5.73
DM_RATIO_V8 = CONSTANTS.get('dm_ratio_v8', 5.43)  # de Sitter corrected


class DarkMatterAnalysis:
    def __init__(self,
                 observed_mass: float,
                 total_mass: float,
                 radius: float,
                 velocity_dispersion: float,
                 dark_mass: float = None,
                 visible_mass: float = None):
        self.observed_mass = observed_mass  # Solar masses
        self.total_mass = total_mass        # Solar masses
        self.radius = radius                # Light years
        self.visible_mass = visible_mass    # Solar masses
        self.velocity_dispersion = velocity_dispersion  # km/s
        self.mass = total_mass
        # Initialize quantum geometry
        self.qg = QuantumGeometry()

        # v7 Immirzi parameter
        self.gamma_0 = GAMMA_0

        # Compute quantum parameters from cluster properties
        self.beta = self._compute_beta()
        self.gamma_eff = self._compute_gamma()
        
    def compute_beta_parameter(self):
        # Normalize beta to ensure 0 < β < 1
        G = CONSTANTS['G']
        c = CONSTANTS['c']
        raw_beta = self.mass * G / (c * c * self.radius)
        return np.clip(raw_beta, 0, 0.99)  # Ensure strictly less than 1

    def _compute_beta(self):
        """Compute quantum coupling parameter beta"""
        M_scale = self.total_mass / CONSTANTS['M_sun']
        r_scale = self.radius / CONSTANTS['R_sun']
        return (1/M_scale) * np.exp(-r_scale/self.qg.phi)

    def _compute_gamma_parameter(self):
        """Compute quantum geometric coupling gamma using v7 formulation."""
        beta = self.compute_beta_parameter()

        # v7 formulation: γ_eff = γ₀ × (1 + β²)
        # where γ₀ = 0.274 is the Immirzi parameter
        gamma = self.gamma_0 * (1 + beta**2)

        return gamma

    def _compute_gamma(self):
        """Compute quantum geometric coupling gamma using v7 formulation."""
        beta = self.compute_beta_parameter()

        # v7 formulation: γ_eff = γ₀ × (1 + β²)
        gamma = self.gamma_0 * (1 + beta**2)

        return gamma

    def compute_geometric_enhancement(self, use_v8=True):
        """
        Compute geometric enhancement factor for dark matter.

        v7 formulation (flat-space): M_DM/M_b = π/(2γ₀) ≈ 5.73
            - 7.4σ discrepancy from Planck 2018

        v8 formulation (de Sitter corrected): M_DM/M_b = (π/2γ₀)(sin√Ω_m)/√Ω_m ≈ 5.43
            - 1.4σ discrepancy from Planck 2018
            - Accounts for positive cosmological constant (Λ > 0)
            - Information geometry is S³, not flat R³

        Args:
            use_v8: If True, use v8 de Sitter correction (default: True)

        Returns:
            Dark matter to baryonic matter ratio
        """
        if use_v8:
            # v8 de Sitter corrected ratio
            return DM_RATIO_V8
        else:
            # v7 flat-space ratio (for backward compatibility)
            return DM_RATIO_V7

    def compute_geometric_enhancement_v8(self, omega_m=None):
        """
        Compute v8 de Sitter corrected dark matter ratio with custom Ω_m.

        This allows computing the ratio for different matter densities,
        useful for exploring the self-consistency equation.

        Args:
            omega_m: Matter density parameter (default: 0.317)

        Returns:
            Dark matter to baryonic matter ratio
        """
        if omega_m is None:
            omega_m = CONSTANTS.get('omega_m_v8', 0.317)
        return compute_dm_ratio_v8(gamma_0=self.gamma_0, omega_m=omega_m)


    def compare_with_observations(self):
        """Compare predicted vs observed mass discrepancy"""
        observed_ratio = self.total_mass / self.observed_mass
        predicted_ratio = self.compute_geometric_enhancement()
        
        return {
            'observed_ratio': observed_ratio,
            'predicted_ratio': predicted_ratio,
            'discrepancy': abs(observed_ratio - predicted_ratio)/observed_ratio
        }

    def compute_dark_matter_ratio(self):
        """Compute ratio of dark matter to visible matter"""
        return (self.total_mass - self.visible_mass) / self.visible_mass

    def quantum_nfw_profile(self, r):
        """
        Quantum-corrected NFW profile using v7 formulation.

        Q(r) = 1 + c(M) × exp(-r/(rs×φ)) × γ₀
        where γ₀ = 0.274 is the Immirzi parameter.
        """
        rs = self.scale_radius
        mass_coupling = self.compute_mass_coupling()
        phi = (1 + np.sqrt(5)) / 2

        quantum_term = mass_coupling * np.exp(-r / (rs * phi))
        # v7: use γ₀ instead of √(Λ/(24×φ))
        geometric_factor = self.gamma_0

        return 1 + quantum_term * geometric_factor

    def _compute_geometric_entanglement(self):
        """
        Compute geometric entanglement using v7 formulation.

        v7: E = β × γ₀ (replaces β × √(Λ/24))
        """
        beta = self._compute_beta()
        # v7 formulation: use γ₀ instead of Leech lattice factor
        return beta * self.gamma_0 * 0.4

    def quantum_correction_factor(self):
        """Calculate quantum correction factor for geometric enhancement"""
        # Base quantum correction using golden ratio
        phi = (1 + np.sqrt(5)) / 2
        
        # Scale with mass and radius
        mass_scale = self.total_mass / CONSTANTS['M_sun']
        radius_scale = self.radius / CONSTANTS['R_sun']
        
        # Quantum correction formula
        correction = (1/phi) * np.exp(-radius_scale/mass_scale)
        
        # Normalize to physical range
        return np.clip(correction, 0, 1)
