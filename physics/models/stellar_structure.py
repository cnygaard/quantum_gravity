import numpy as np
from constants import CONSTANTS, SI_UNITS

class StarSimulation:
    """Quantum-enhanced stellar structure simulation"""
    
    def __init__(self, mass, radius):
        self.mass = np.float128(mass)
        self.radius = np.float128(radius)
        self.M_star = self.mass * CONSTANTS['M_sun']
        self.R_star = self.radius * CONSTANTS['R_sun']
        self.setup_quantum_parameters()
        
    def setup_quantum_parameters(self):
        """Initialize quantum geometric parameters"""
        self.beta = CONSTANTS['l_p'] / self.R_star
        # Enhanced quantum coupling
        self.gamma_eff = 0.407 * self.beta * np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/24)
        
    def compute_quantum_factor(self):
        r_natural = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
        m_natural = self.mass / CONSTANTS['M_sun']
        
        # Leech lattice geometric factors
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        lattice_factor = np.sqrt(points/dimension)
        
        # Normalized quantum enhancement
        scale_factor = np.exp(-r_natural/1e4)
        quantum_enhancement = scale_factor * lattice_factor * (m_natural)**0.25
        
        # Ensure result is between 1.0 and 1.1
        return 1.0 + 0.1 * np.tanh(quantum_enhancement * 1e-6)

    def compute_total_pressure(self):
        """Calculate total pressure including quantum effects"""
        G = SI_UNITS['G_si']
        
        # Base pressure calculation
        if self.radius < 0.01:  # Compact objects
            P_classical = (CONSTANTS['c']**4 / (12*np.pi*G)) * (self.mass/1.4)**2 * (0.01/self.radius)**4
        else:  # Normal stars
            P_classical = (3 * G * self.M_star**2) / (8 * np.pi * self.R_star**4)
        
        # Quantum enhancement with proper scaling
        quantum_factor = self.compute_quantum_factor()
        return P_classical * quantum_factor

    def compute_gravitational_pressure(self):
        """Calculate gravitational pressure"""
        G = SI_UNITS['G_si']
        # Match base pressure calculation
        return (3 * G * self.M_star**2) / (8 * np.pi * self.R_star**4)
            
    def total_energy(self):
        """Calculate total energy with quantum corrections"""
        G = SI_UNITS['G_si']
        E_grav = -3 * G * self.M_star**2 / (5 * self.R_star)
        E_quantum = E_grav * self.compute_quantum_factor()
        return E_grav + E_quantum
        
    def evolve(self, timesteps):
        """Evolve star forward in time"""
        dt = 1000 * 365 * 24 * 3600  # 1000 years in seconds
        for _ in range(timesteps):
            self.R_star *= (1 + self.gamma_eff * dt)
            
    def compute_entanglement_entropy(self):
        """Calculate quantum entanglement entropy"""
        # Scale with mass and radius
        mass_scale = self.mass / CONSTANTS['M_sun']
        radius_scale = self.radius / CONSTANTS['R_sun']
        S_classical = CONSTANTS['k_B'] * mass_scale
        return S_classical * (1 + self.gamma_eff * np.log(radius_scale))
        
    # def compute_surface_temperature(self):
    #     """Calculate surface temperature with quantum effects"""
    #     T_classical = 5778 * (self.mass**0.5) * (self.radius**-0.5)
    #     return T_classical * (1 + 0.1 * self.gamma_eff)
    def compute_surface_temperature(self):
        """Calculate surface temperature with quantum effects"""
        # Classical surface temperature from stellar physics
        T_classical = 5778 * (self.mass**0.5) * (self.radius**-0.5)
        
        # Quantum enhancement with Leech lattice contribution
        leech_factor = np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/24)
        quantum_correction = 1 + self.gamma_eff * self.beta * leech_factor
        
        return T_classical * quantum_correction

    # def compute_temperature_profile(self):
    #     """Calculate temperature structure with quantum effects"""
    #     class TempProfile:
    #         def __init__(self, core, surface):
    #             self.core = core
    #             self.surface = surface
                
    #     T_core = 1.57e7 * (self.mass**0.5)
    #     T_surface = 5778 * (self.mass**0.5) * (self.radius**-0.5)
    #     return TempProfile(T_core, T_surface)
    
    def compute_surface_temperature(self):
        """Calculate surface temperature with quantum effects"""
        # Classical surface temperature from stellar physics
        T_classical = 5778 * (self.mass**0.5) * (self.radius**-0.5)
        
        # Quantum enhancement with Leech lattice contribution
        leech_factor = np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/24)
        quantum_correction = 1 + self.gamma_eff * self.beta * leech_factor
        
        return T_classical * quantum_correction

    def compute_temperature_profile(self):
        """Calculate temperature structure with quantum effects"""
        class TempProfile:
            def __init__(self, core, surface):
                self.core = core
                self.surface = surface
                
        # Core temperature with quantum corrections
        T_core = 1.57e7 * (self.mass**0.5) * (1 + self.gamma_eff * self.beta * np.sqrt(196560/24))
        
        # Surface temperature with quantum corrections 
        T_surface = self.compute_surface_temperature()
        
        return TempProfile(T_core, T_surface)



    # def compute_density_enhancement(self):
    #     """Calculate quantum-enhanced core density"""
    #     T_surface = 3042 * (self.mass**0.505) * quantum_factor
    #     quantum_factor = self.compute_quantum_factor()
    #     return 1.0 + 0.05 * (quantum_factor - 1.0)

    # def compute_density_enhancement(self):
    #     """Calculate quantum-enhanced core density"""
    #     if self.radius < 0.01:  # Compact objects
    #         enhancement = 1.0 + 0.2 * quantum_factor * (0.01/self.radius)**0.5
    #     elif self.mass > 10:  # Massive stars
    #         enhancement = 1.0 + 0.05 * quantum_factor * np.log(self.mass/10)
    #     else:  # Main sequence
    #         enhancement = 1.0 + 0.05 * (quantum_factor - 1.0)
            
    #     return enhancement

    def compute_density_enhancement(self):
        """Calculate quantum-enhanced core density"""
        quantum_factor = self.compute_quantum_factor()
        if self.radius < 0.01:  # Compact objects
            enhancement = 1.0 + 0.2 * quantum_factor * (0.01/self.radius)**0.5
        elif self.mass > 10:  # Massive stars
            enhancement = 1.0 + 0.05 * quantum_factor * np.log(self.mass/10)
        else:  # Main sequence
            enhancement = 1.0 + 0.05 * (quantum_factor - 1.0)
        return enhancement

        #return TempProfile(T_core, T_surface)
    
    # def compute_density_enhancement(self):
    #     """Calculate quantum-enhanced core density"""
    #     quantum_factor = self.compute_quantum_factor()
    #     return 1.0 + 0.05 * (quantum_factor - 1.0)

    #     quantum_factor = self.compute_quantum_factor()
    #     T_surface = 3600 * (self.mass/10)**0.18 * quantum_factor
    #     T_surface = 5778 * (self.mass**0.505) * quantum_factor
            
        
    #      T_core = 1.57e7 * (self.mass**0.7) * quantum_factor
    #     else:  # Main sequence
    #     return 1.0 + 0.05 * (quantum_factor - 1.0)

    #     # Enhanced density based on stellar type
    #     if self.radius < 0.01:  # Compact objects
        
    #         T_core = 3.5e7 * (self.mass/16.5)**0.3 * quantum_factor
    #     elif self.mass > 10:  # Massive stars