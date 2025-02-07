from typing import Dict
import numpy as np
from constants import CONSTANTS

class RealisticEOS:
    """Handle realistic equation of state including nuclear and quantum effects"""
    
    def __init__(self, include_nuclear: bool = True, include_degeneracy: bool = True,
                 mass: float = None, radius: float = None):
        self.include_nuclear = include_nuclear
        self.include_degeneracy = include_degeneracy
        self.relativistic_corrections = True  # Assuming this attribute is needed
        # Store mass and radius in solar units
        self.mass = mass  # In solar masses
        self.radius = radius  # In solar radii
        
    def update_parameters(self, mass: float, radius: float):
        """Update stellar parameters"""
        self.mass = mass
        self.radius = radius
        
    def compute_degeneracy_pressure(self, density: float, composition: Dict) -> float:
        """Compute electron/neutron degeneracy pressure"""
        if not self.include_degeneracy:
            return 0.0
            
        # Calculate Fermi energy
        E_F = self._compute_fermi_energy(density, composition)
        
        # Get degeneracy pressure including relativistic effects
        P_deg = self._relativistic_degeneracy_pressure(E_F, density)
        
        return P_deg
        
    def quantum_density_factor(self, r_normalized: float) -> float:
        """Compute quantum corrections to density profile."""
        if self.mass is None or self.radius is None:
            return 1.0
            
        # Calculate compactness parameter
        compactness = (self.mass * CONSTANTS['M_sun']) / (self.radius * CONSTANTS['R_sun'])
        compactness *= CONSTANTS['G'] / CONSTANTS['c']**2
        
        # Enhanced quantum corrections for compact objects
        if compactness > 0.1:  # Neutron stars/white dwarfs
            base_factor = np.exp(10 * compactness)
            quantum_factor = 1.0 + base_factor * (1.0 - r_normalized)
        else:  # Main sequence stars
            # Regular stellar quantum effects
            quantum_factor = 1.0 + 0.2 * np.exp(-r_normalized**2 / 0.1)
            
        # Add degeneracy effects
        if self.include_degeneracy and self.radius < 0.01:  # Compact objects
            fermi_factor = 1.0 / (1.0 + np.exp(-20*(1.0 - r_normalized)))  # Sharper transition
            quantum_factor *= (1.0 + fermi_factor)
            
        return quantum_factor

    def compute_pressure(self, density: float, composition: Dict) -> float:
        """Multi-component pressure calculation"""
        # Gas pressure
        P_gas = self._compute_ideal_gas_pressure(density, composition)
        
        # Radiation pressure (important for massive stars)
        P_rad = self._compute_radiation_pressure(density, composition)
        
        # Degeneracy pressure (for compact objects)
        P_deg = self._compute_degeneracy_pressure(density, composition)
        
        # Quantum vacuum pressure
        P_vacuum = self._compute_vacuum_pressure(density)
        
        return P_gas + P_rad + P_deg + P_vacuum
