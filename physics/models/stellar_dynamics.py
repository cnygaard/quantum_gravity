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
        """Calculate expected rotation curve with quantum effects"""
        # Scale quantum parameters to stellar scales
        beta_stellar = self.beta * (self.radius/CONSTANTS['R_sun'])
        return self.orbital_velocity * np.sqrt(1 + beta_stellar)
