import numpy as np
from constants import CONSTANTS

class DarkMatterAnalysis:
    def __init__(self, 
                 observed_mass: float,
                 total_mass: float, 
                 radius: float,
                 velocity_dispersion: float):
        self.observed_mass = observed_mass  # Solar masses
        self.total_mass = total_mass        # Solar masses 
        self.radius = radius                # Light years
        self.velocity_dispersion = velocity_dispersion  # km/s

        # Compute quantum parameters from cluster properties
        self.beta = self._compute_beta()
        self.gamma_eff = self._compute_gamma()
        
    def _compute_beta(self):
    # Add mass-dependent scaling factor
        mass_scale = (self.total_mass/1e15)**1.0  # Reference to Coma mass
        return 2.32e14 * (self.total_mass/self.observed_mass/10.0) * mass_scale

    def _compute_gamma(self):
        """Derive coupling strength from cluster dynamics"""
        velocity_factor = (self.velocity_dispersion/1000.0)**1.0  # Normalized to Coma
        return 8.15e15 * velocity_factor
        
    #     return enhancement
    def compute_geometric_enhancement(self):
        """Calculate force enhancement from quantum geometric effects"""
        # Scale quantum parameters to cluster size
        scale_factor = self.radius / CONSTANTS['R_sun']
        beta_cluster = self.beta * scale_factor
        
        # Include velocity dispersion effects
        velocity_factor = (self.velocity_dispersion / CONSTANTS['c'])**2
        
        # Enhanced Leech lattice contribution
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        lattice_factor = np.sqrt(points/dimension)  # Stronger geometric factor
        
        enhancement = self.gamma_eff * lattice_factor * beta_cluster * velocity_factor
        
        return enhancement

    def compare_with_observations(self):
        """Compare predicted vs observed mass discrepancy"""
        observed_ratio = self.total_mass / self.observed_mass
        predicted_ratio = self.compute_geometric_enhancement()
        
        return {
            'observed_ratio': observed_ratio,
            'predicted_ratio': predicted_ratio,
            'discrepancy': abs(observed_ratio - predicted_ratio)/observed_ratio
        }
