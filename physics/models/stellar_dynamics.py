import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis
import numpy as np
from constants import CONSTANTS, SI_UNITS

class StellarDynamics(DarkMatterAnalysis):
    def __init__(self, orbital_velocity, radius, mass):

        # Set mass attribute before parent initialization
        self.mass = mass
        # Initialize parent class with stellar parameters
        super().__init__(
            observed_mass=mass,
            total_mass=mass*10,  # Typical dark matter ratio
            radius=radius,
            velocity_dispersion=orbital_velocity
        )
        self.orbital_velocity = orbital_velocity
        self.radius = radius

    # def compute_rotation_curve(self):
    #     """Calculate rotation curve with quantum corrections"""
    #     # Convert to SI units
    #     G = SI_UNITS['G_si']
    #     M = self.mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Classical Keplerian velocity
    #     v_classical = np.sqrt(G * M / R)
        
    #     # Modified gravity with proper scaling
    #     quantum_factor = self.compute_quantum_factor()
    #     v_quantum = v_classical * np.sqrt(quantum_factor)
        
    #     return v_quantum * (1 - np.exp(-R/(20 * SI_UNITS['ly_si'])))  # Smooth transition
    def compute_rotation_curve(self):
        """Calculate rotation curve with proper scaling"""
        # Convert units
        G = SI_UNITS['G_si']
        M = self.mass * SI_UNITS['M_sun_si']
        R = self.radius * SI_UNITS['ly_si']
        
        # Classical velocity
        v_classical = np.sqrt(G * M / R)
        
        # Apply quantum correction with proper scaling
        quantum_factor = self.compute_quantum_factor()
        v_quantum = v_classical * np.sqrt(quantum_factor)
        
        # Add radial dependence for flat rotation curves
        r_scale = 20 * SI_UNITS['ly_si']
        flattening = np.tanh(R/r_scale)
        
        return v_quantum * flattening


    # def compute_rotation_curve(self):
    #     """Calculate rotation curve with quantum corrections"""
    #     # Convert units
    #     G = SI_UNITS['G_si']
    #     M = self.mass * SI_UNITS['M_sun_si']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     # Classical Keplerian velocity
    #     v_classical = np.sqrt(G * M / R)
        
    #     # Apply quantum correction
    #     quantum_factor = self.compute_quantum_factor()
    #     return v_classical * np.sqrt(quantum_factor)

    # def compute_rotation_curve(self):
    #     """Computes rotation curve with proper quantum corrections"""
    #     quantum_factor = self.compute_quantum_factor()
        
    #     # Calculate Keplerian velocity with quantum enhancement
    #     G = CONSTANTS['G']
    #     M = self.mass * CONSTANTS['M_sun']
    #     R = self.radius * SI_UNITS['ly_si']
        
    #     v_kepler = np.sqrt(G * M / R)
    #     return v_kepler * np.sqrt(quantum_factor)
       
    # def compute_rotation_curve(self):
    #     """
    #     Computes the rotation curve accounting for quantum gravity effects
    #     Returns velocity in km/s
    #     """
    #     quantum_factor = self.compute_quantum_factor()
    #     geometric_mass = self.mass * quantum_factor
    #     # Calculate velocity using enhanced mass...
        
    #     # Apply quantum correction to orbital velocity
    #     corrected_velocity = self.orbital_velocity * quantum_factor
        
    #     # Account for dark matter contribution
    #     dark_matter_mass = self.calculate_universal_dark_matter()
    #     dm_velocity_factor = np.sqrt(dark_matter_mass / self.mass)
        
    #     # Final rotation velocity 
    #     rotation_velocity = corrected_velocity 
    #     #* dm_velocity_factor
        
    #     return rotation_velocity

    def calculate_universal_dark_matter(self):
        # Leech lattice parameters
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']  # 24
        points = CONSTANTS['LEECH_LATTICE_POINTS']        # 196560
        lattice_factor = np.sqrt(points/dimension)        # ~90.5
        
        # Standard dark matter ratio for spiral galaxies (5:1 to 10:1)
        dark_matter_factor = 7.2
        
        # Scale geometric coupling with radius
        radius_scale = (self.radius/CONSTANTS['R_sun']) * 1e-15
        beta_universal = self.beta * lattice_factor * radius_scale
        
        # Total mass with radius-dependent scaling
        total_mass = self.mass * dark_matter_factor * (1 + beta_universal)
        return total_mass    
    

    # def compute_quantum_factor(self):
    #     beta = 2.32e-44 * (self.radius/CONSTANTS['R_sun'])
    #     gamma_eff = 8.15e-45
    #     geometric_factor = 1 + gamma_eff * np.sqrt(0.407)
    #     return geometric_factor


    # def compute_quantum_factor(self):
    #     scale = (self.radius/CONSTANTS['R_sun']) * (self.mass/CONSTANTS['M_sun'])
    #     beta = 2.32e-44 / scale
    #     gamma_eff = 8.15e-45 / scale
    #     return 1 + gamma_eff * np.sqrt(0.407)  # Ensures approach to 1.0

    # def compute_quantum_factor(self):
    #     # Improved scaling with proper normalization
    #     scale = (self.radius/CONSTANTS['R_sun']) * np.sqrt(self.mass/CONSTANTS['M_sun'])
    #     beta = 2.32e-44 / scale
    #     gamma_eff = 8.15e-45 / scale
    #     return 1 + gamma_eff * np.sqrt(0.407)
    # def compute_quantum_factor(self):
    #     """Compute quantum geometric enhancement with proper scaling"""
    #     # Convert to Planck units
    #     r_planck = self.radius * SI_UNITS['ly_si'] / CONSTANTS['l_p']
    #     m_planck = self.mass * SI_UNITS['M_sun_si'] / CONSTANTS['m_p']
        
    #     # Calculate dimensionless parameters
    #     beta = 1.0 / r_planck
    #     gamma_eff = 0.407 / np.sqrt(m_planck)
        
    #     # Quantum correction with proper scaling
    #     return 1.0 + gamma_eff * beta
    # def compute_quantum_factor(self):
    #     """Compute quantum geometric enhancement with galactic scaling"""
    #     # Convert to dimensionless parameters
    #     r_scale = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
    #     m_scale = self.mass / CONSTANTS['M_sun']
        
    #     # Quantum coupling with proper scaling
    #     beta = 2.32e-44 * np.sqrt(m_scale/r_scale)
    #     gamma_eff = 0.407 * beta
        
    #     return 1.0 + gamma_eff * np.exp(-r_scale/1000)  # Exponential suppression at large scales

    def compute_quantum_factor(self):
        """Compute quantum geometric enhancement with proper galactic scaling"""
        # Convert to natural units
        r_natural = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
        m_natural = self.mass / CONSTANTS['M_sun']
        
        # Scale-dependent coupling
        beta = 1e-6 * np.sqrt(m_natural/r_natural)  # Adjusted coupling strength
        gamma = 0.407 * beta  # Geometric factor
        
        # Smooth transition function
        transition = np.exp(-r_natural/1e4)  # Characteristic scale ~10 kpc
        
        return 1.0 + gamma * transition

    # def kinetic_energy(self):
    #     """Calculate kinetic energy"""
    #     v = self.orbital_velocity * 1000  # km/s to m/s
    #     return 0.5 * self.mass * CONSTANTS['M_sun'] * v**2
    def kinetic_energy(self):
        """Calculate kinetic energy in SI units"""
        v = self.orbital_velocity * 1000  # km/s to m/s
        M = self.mass * SI_UNITS['M_sun_si']
        return 0.5 * M * v * v
        
    # def potential_energy(self):
    #     """Calculate gravitational potential energy"""
    #     R = self.radius * SI_UNITS['ly_si']
    #     return -CONSTANTS['G'] * self.mass * CONSTANTS['M_sun'] / R
    def potential_energy(self):
        """Calculate potential energy in SI units"""
        G = SI_UNITS['G_si']
        M = self.mass * SI_UNITS['M_sun_si']
        R = self.radius * SI_UNITS['ly_si']
        return -G * M * M / R