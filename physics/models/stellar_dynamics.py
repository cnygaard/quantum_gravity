from physics.models.dark_matter import DarkMatterAnalysis
import numpy as np
from constants import CONSTANTS

class StellarDynamics(DarkMatterAnalysis):
    def __init__(self, orbital_velocity, radius, mass):
        # Initialize parent class with stellar parameters
        super().__init__(
            observed_mass=mass,
            total_mass=mass*10,  # Typical dark matter ratio
            radius=radius,
            velocity_dispersion=orbital_velocity
        )
        self.orbital_velocity = orbital_velocity
        self.radius = radius
        self.mass = mass
        
    def compute_rotation_curve(self):
        """
        Computes the rotation curve accounting for quantum gravity effects
        Returns velocity in km/s
        """
        # Calculate quantum geometric coupling
        beta_factor = self.beta * (self.radius/CONSTANTS['R_sun'])
        
        # Quantum enhancement factor
        quantum_factor = np.sqrt(1 + beta_factor)
        
        # Apply quantum correction to orbital velocity
        corrected_velocity = self.orbital_velocity * quantum_factor
        
        # Account for dark matter contribution
        dark_matter_mass = self.calculate_universal_dark_matter()
        dm_velocity_factor = np.sqrt(dark_matter_mass / self.mass)
        
        # Final rotation velocity 
        rotation_velocity = corrected_velocity * dm_velocity_factor
        
        return rotation_velocity



    def calculate_universal_dark_matter(self):
        # Leech lattice parameters
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']  # 24
        points = CONSTANTS['LEECH_LATTICE_POINTS']        # 196560
        lattice_factor = np.sqrt(points/dimension)        # ~90.5
        
        # Standard dark matter ratio for spiral galaxies (5:1 to 10:1)
        dark_matter_factor = 1
        
        # Scale geometric coupling with radius
        radius_scale = (self.radius/CONSTANTS['R_sun']) * 1e-15
        beta_universal = self.beta * lattice_factor * radius_scale
        
        # Total mass with radius-dependent scaling
        total_mass = self.mass * dark_matter_factor * (1 + beta_universal)
        
        return total_mass

