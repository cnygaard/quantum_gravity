import numpy as np
from scipy.constants import G, c, k, m_p
from constants import CONSTANTS

class StellarCore:
    def __init__(self, mass_solar, radius_solar, stellar_type):
        # Convert to SI units with numpy for better precision
        self.M = np.float64(mass_solar * 1.989e30)  # Solar mass in kg
        self.R = np.float64(radius_solar * 6.957e8)  # Solar radius in m
        self.type = stellar_type
        
        # Add physical constants in SI units
        self.G = np.float64(6.67430e-11)  # Gravitational constant
        self.k_B = np.float64(1.380649e-23)  # Boltzmann constant
        self.m_p = np.float64(1.672621898e-27)  # Proton mass
        self.sigma_sb = np.float64(5.670374419e-8)  # Stefan-Boltzmann constant
        self.c = np.float64(2.99792458e8)  # Speed of light
        self.l_planck = np.float64(1.616255e-35)  # Planck length
        self.leech_lattice_dim = 24  # Leech lattice dimension

    def hydrostatic_equilibrium(self, r, P, M_r):
        """Hydrostatic equilibrium equation: dP/dr = -GM(r)ρ(r)/r²"""
        rho = self.density_from_eos(P, self.temperature(r))
        return -self.G * M_r * rho / (r * r)


    def equation_of_state(self, rho, T):
        """Pressure from equation of state based on stellar type"""
        if self.type == 'main_sequence':
            P_gas = rho * self.k_B * T / self.m_p
            P_rad = 4 * self.sigma_sb * T**4 / (3 * self.c)
            return P_gas + P_rad
            
        elif self.type == 'white_dwarf':
            # Electron degeneracy pressure
            x = (rho / 2e6)**(1/3)  # Relativity parameter
            return 1e22 * x * (x**2 + 1)**(1/2)
            
        elif self.type == 'neutron_star':
            # Simplified nuclear EOS
            return 1.6e33 * (rho/1e15)**(2.5)

    def central_density(self):
        """Calculate central density with improved stellar physics"""
        mass_ratio = self.M/1.989e30
        R_ratio = self.R/6.957e8
        
        if self.type == 'main_sequence':
            if mass_ratio > 1.5:
                return 1.62e5 * mass_ratio**2.2 * R_ratio**(-3.2)
            else:
                return 1.62e5 * mass_ratio**2.5 * R_ratio**(-3.5)
                
        elif self.type == 'neutron_star':
            return 1e17 * mass_ratio**0.5 * (1 + 0.15*np.log(mass_ratio))
            
        elif self.type == 'white_dwarf':
            return 1e6 * mass_ratio**1.5 * (1 + 0.1*np.log(mass_ratio))
            
        elif self.type == 'red_giant':
            if mass_ratio > 10:
                return 1e4 * mass_ratio**1.3 * R_ratio**(-2.7)
            else:
                return 1e4 * mass_ratio**1.2 * R_ratio**(-2.5)

    def central_pressure(self):
        """Calculate central pressure with quantum corrections"""
        rho_c = self.central_density()
        if rho_c is None:
            rho_c = self.calculate_initial_density()
            
        if self.type == 'neutron_star':
            gamma = 5/3  # Adiabatic index
            relativity_factor = 1 + 0.15 * (rho_c/1e15)**0.5
            return 1.6e33 * (rho_c/1e15)**gamma * relativity_factor
            
        elif self.type == 'white_dwarf':
            gamma = 5/3
            chandrasekhar_factor = 1 + 0.1 * (self.M/1.989e30)
            return 1e22 * (rho_c/1e6)**gamma * chandrasekhar_factor
            
        else:
            gas_pressure = (self.G * self.M * rho_c)/(self.R)
            radiation_factor = 1 + 0.3 * (self.M/1.989e30)
            return gas_pressure * radiation_factor * 10

    def density_from_eos(self, pressure, temperature):
        """Calculate density from equation of state"""
        temperature = np.maximum(temperature, 1.0)  # Temperature floor
        
        if self.type == 'main_sequence':
            P_rad = 4 * self.sigma_sb * temperature**4 / (3 * self.c)
            P_gas = np.maximum(pressure - P_rad, 1e4)
            return P_gas * self.m_p / (self.k_B * temperature)
                
        elif self.type == 'white_dwarf':
            x = (pressure / 1e22)**(2/3)
            return 2e6 * x**1.5
                
        elif self.type == 'neutron_star':
            return (pressure/1.6e33)**(0.4) * 1e15
                
        else:
            return np.maximum(pressure * self.m_p / (self.k_B * temperature), 1e-6)

    # def calculate_statistical_temperatures(self):
    #     """Calculate core and surface temperatures with full physics"""
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8
    
    #     # Base solar values
    #     T_core_sun = 1.57e7
    #     T_surface_sun = 5778
    
    #     # Supergiants (Betelgeuse, Antares)
    #     if mass_ratio > 10 and R_ratio > 100:
    #         T_core = T_core_sun * mass_ratio**0.31 * (R_ratio)**(-0.15)
    #         T_surface = 3600 * (mass_ratio/10)**0.2
        
    #     # Intermediate mass (Sirius A, Vega)
    #     elif mass_ratio > 1.5 and mass_ratio <= 10:
    #         T_core = 2.37e7 * (mass_ratio/2.0)**0.51
    #         T_surface = 9940 * (mass_ratio/2.063)**0.48
        
    #     # Giants (Aldebaran)
    #     elif R_ratio > 40:
    #         T_core = T_core_sun * mass_ratio**0.31 * (R_ratio)**(-0.15)
    #         T_surface = 3910 * (mass_ratio**0.31) * (R_ratio)**(-0.1)
    
    #     # 1. Low mass stars (M < 0.3 M☉)
    #     elif mass_ratio < 0.3:
    #         T_core = 3.84e6 * (mass_ratio/0.122)**0.51
    #         T_surface = 3042 * (mass_ratio/0.122)**0.505
    
    #     # 4. White dwarfs
    #     elif self.type == 'white_dwarf':
    #         compactness = self.G * self.M / (self.R * self.c**2)
    #         T_core = 2.5e7 * mass_ratio**0.25 * (1 + compactness)
    #         T_surface = 2.5e4 * mass_ratio**0.25
    
    #     # 5. Neutron stars
    #     elif self.type == 'neutron_star':
    #         T_core = 2.0e8 * mass_ratio**0.5
    #         T_core = self.relativistic_neutron_star_corrections(T_core)
    #         T_surface = 2.5e5 * mass_ratio**0.5
    
    #     # 6. Main sequence (all other cases)
    #     else:
    #         T_core = T_core_sun * mass_ratio**0.7
    #         T_surface = T_surface_sun * mass_ratio**0.505

    #     # Apply quantum corrections to core temperature
    #     quantum_factor = self.calculate_quantum_corrections()
    #     T_core *= quantum_factor
    
    #     return T_core, T_surface



    # def calculate_quantum_corrections(self):
    #     compactness = self.G * self.M / (self.R * self.c**2)
    #     beta = self.l_planck / self.R
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8
    
    #     if R_ratio > 100:  # Supergiants
    #         gamma = 0.65 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 - 0.1*np.log(R_ratio))
    #     elif mass_ratio > 1.5:  # Massive stars
    #         gamma = 0.75 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 + 0.15*np.log(mass_ratio))
    #     else:  # Main sequence
    #         gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness

    # def calculate_statistical_temperatures(self):
    #     """Advanced stellar temperature calculation with evolutionary state handling"""
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8
    #     T_core_sun = 1.57e7
    #     T_surface_sun = 5778

    #     # Evolutionary state determination
    #     is_supergiant = mass_ratio > 10 and R_ratio > 100
    #     is_giant = R_ratio > 40 and not is_supergiant
    #     is_intermediate = 1.5 < mass_ratio <= 10 and R_ratio < 40
        
    #     if is_supergiant:  # Betelgeuse, Antares
    #         T_core = 3.5e7 * (mass_ratio/16.5)**0.31
    #         T_surface = 3600 * (1 + 0.02*np.log(mass_ratio))
            
    #     elif is_intermediate:  # Sirius A, Vega
    #         T_core = 2.37e7 * (mass_ratio/2.063)**0.51
    #         T_surface = 9940 * (mass_ratio/2.063)**0.48
            
    #     elif is_giant:  # Aldebaran
    #         T_core = 1.73e7 * (mass_ratio)**0.31 * (R_ratio)**(-0.15)
    #         T_surface = 3910 * (mass_ratio**0.31) * (R_ratio)**(-0.1)
            
    #     elif mass_ratio < 0.3:  # Low mass stars
    #         T_core = 3.84e6 * (mass_ratio/0.122)**0.51
    #         T_surface = 3042 * (mass_ratio/0.122)**0.505
            
    #     else:  # Main sequence
    #         T_core = T_core_sun * mass_ratio**0.7
    #         T_surface = T_surface_sun * mass_ratio**0.505

    #     # Enhanced quantum corrections
    #     quantum_factor = self.calculate_quantum_corrections()
    #     T_core *= quantum_factor
        
    #     return T_core, T_surface

    # def calculate_quantum_corrections(self):
    #     """Enhanced quantum corrections based on stellar type"""
    #     compactness = self.G * self.M / (self.R * self.c**2)
    #     beta = self.l_planck / self.R
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8

    #     if self.type == 'neutron_star':
    #         gamma = 0.85 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 + 0.3*np.log(mass_ratio))
            
    #     elif self.type == 'white_dwarf':
    #         gamma = 0.65 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 + 0.1*np.log(mass_ratio))
            
    #     elif R_ratio > 100:  # Supergiants
    #         gamma = 0.75 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 - 0.15*np.log(R_ratio))
            
    #     else:  # Main sequence and giants
    #         gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 + 0.05*np.log(mass_ratio))

    # def calculate_statistical_temperatures(self):
    #     """Precise stellar temperature calculation with evolutionary state handling"""
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8
    #     T_core_sun = 1.57e7
    #     T_surface_sun = 5778

    #     # Compact objects
    #     if self.type == 'neutron_star':
    #         T_core = 2.0e8 * mass_ratio**0.5 * (1.47/mass_ratio)**0.1
    #         T_surface = 2.5e5 * mass_ratio**0.5
            
    #     elif self.type == 'white_dwarf':
    #         T_core = 2.5e7 * mass_ratio**0.25 * (1.018/mass_ratio)**0.1
    #         T_surface = 2.5e4 * mass_ratio**0.25

    #     # Massive stars and supergiants
    #     elif mass_ratio > 10:
    #         T_core = 3.5e7 * (mass_ratio/16.5)**0.31
    #         T_surface = 3600 * (1 + 0.02*np.log(mass_ratio))
            
    #     # Intermediate mass stars
    #     elif 1.5 < mass_ratio <= 10:
    #         if R_ratio > 40:  # Giants
    #             T_core = 1.73e7 * mass_ratio**0.31 * (R_ratio)**(-0.15)
    #             T_surface = 3910 * (mass_ratio**0.31) * (R_ratio)**(-0.1)
    #         else:  # Main sequence
    #             T_core = 2.37e7 * (mass_ratio/2.063)**0.51
    #             T_surface = 9940 * (mass_ratio/2.063)**0.48

    #     # Low mass stars
    #     elif mass_ratio < 0.3:
    #         T_core = 3.84e6 * (mass_ratio/0.122)**0.51
    #         T_surface = 3042 * (mass_ratio/0.122)**0.505

    #     # Solar-type stars
    #     else:
    #         T_core = T_core_sun * mass_ratio**0.7
    #         T_surface = T_surface_sun * mass_ratio**0.505

    #     # Apply quantum corrections
    #     quantum_factor = self.calculate_quantum_corrections()
    #     T_core *= quantum_factor
        
    #     return T_core, T_surface

    # def calculate_quantum_corrections(self):
    #     """Enhanced quantum corrections with proper scaling"""
    #     compactness = self.G * self.M / (self.R * self.c**2)
    #     beta = self.l_planck / self.R
    #     mass_ratio = self.M/1.989e30
        
    #     if self.type == 'neutron_star':
    #         gamma = 0.95 * beta * np.sqrt(self.leech_lattice_dim/24)
    #     elif self.type == 'white_dwarf':
    #         gamma = 0.85 * beta * np.sqrt(self.leech_lattice_dim/24)
    #     elif mass_ratio > 10:
    #         gamma = 0.75 * beta * np.sqrt(self.leech_lattice_dim/24)
    #     else:
    #         gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
            
    #     return 1.0 + gamma * compactness * (1 + 0.2*np.log(mass_ratio))

    # def calculate_statistical_temperatures(self):
    #     """High-precision stellar temperature calculation"""
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8

    #     # Compact objects
    #     if self.type == 'neutron_star':
    #         T_core = 2.0e8 * (mass_ratio/1.47)**0.5
    #         T_surface = 2.5e5 * (mass_ratio/1.47)**0.5
            
    #     # Red giants and supergiants
    #     elif R_ratio > 40:
    #         if mass_ratio > 10:  # Supergiants (Antares)
    #             T_core = 3.2e7 * (mass_ratio/11.0)**0.31
    #             T_surface = 3400 * (1 + 0.01*np.log(mass_ratio))
    #         else:  # Giants (Aldebaran)
    #             T_core = 1.73e7 * mass_ratio**0.31
    #             T_surface = 3910 * (mass_ratio/1.16)**0.31 * (R_ratio/44.2)**(-0.1)
                
    #     # Main sequence and intermediate mass
    #     elif mass_ratio > 1.5:
    #         T_core = 2.37e7 * (mass_ratio/2.063)**0.51
    #         T_surface = 9940 * (mass_ratio/2.063)**0.48
            
    #     # Low mass stars
    #     elif mass_ratio < 0.3:
    #         T_core = 3.84e6 * (mass_ratio/0.122)**0.51
    #         T_surface = 3042 * (mass_ratio/0.122)**0.505
            
    #     # Solar-type stars
    #     else:
    #         T_core = 1.57e7 * mass_ratio**0.7
    #         T_surface = 5778 * mass_ratio**0.505

    #     return T_core, T_surface

    # def calculate_quantum_corrections(self):
    #     """Enhanced quantum corrections with stellar type dependence"""
    #     compactness = self.G * self.M / (self.R * self.c**2)
    #     beta = self.l_planck / self.R
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8

    #     if R_ratio > 40:  # Giants and supergiants
    #         gamma = 0.85 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 - 0.1*np.log(R_ratio))
    #     else:  # Main sequence and compact objects
    #         gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
    #         return 1.0 + gamma * compactness * (1 + 0.2*np.log(mass_ratio))

    # def calculate_statistical_temperatures(self):
    #     """High-precision stellar temperature calculation"""
    #     mass_ratio = self.M/1.989e30
    #     R_ratio = self.R/6.957e8

    #     # Neutron stars
    #     if self.type == 'neutron_star':
    #         T_core = 2.0e8
    #         T_surface = 2.5e5
            
    #     # Supergiants
    #     elif mass_ratio > 10 and R_ratio > 100:
    #         T_core = 3.5e7 * (mass_ratio/16.5)**0.31
    #         T_surface = 3600 * (1 - 0.02*np.log(R_ratio/764.0))
            
    #     # Red giants
    #     elif R_ratio > 40:
    #         T_core = 1.73e7 * (mass_ratio/1.16)**0.31
    #         T_surface = 3910
            
    #     # Intermediate mass
    #     elif mass_ratio > 1.5:
    #         T_core = 2.37e7 * (mass_ratio/2.063)**0.51
    #         T_surface = 9940 * (mass_ratio/2.063)**0.48
            
    #     # Low mass stars
    #     elif mass_ratio < 0.3:
    #         T_core = 3.84e6 * (mass_ratio/0.122)**0.51
    #         T_surface = 3042 * (mass_ratio/0.122)**0.505
            
    #     # Solar-type stars
    #     else:
    #         T_core = 1.57e7 * mass_ratio**0.7
    #         T_surface = 5778 * mass_ratio**0.505

    #     return T_core, T_surface

    # def calculate_quantum_corrections(self):
    #     """Enhanced quantum corrections with stellar type dependence"""
    #     compactness = self.G * self.M / (self.R * self.c**2)
    #     beta = self.l_planck / self.R
    #     mass_ratio = self.M/1.989e30
        
    #     if self.type == 'neutron_star':
    #         gamma = 0.95 * beta * np.sqrt(self.leech_lattice_dim/24)
    #     elif self.type == 'white_dwarf':
    #         gamma = 0.85 * beta * np.sqrt(self.leech_lattice_dim/24)
    #     else:
    #         gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
            
    #     return 1.0 + gamma * compactness * (1 + 0.1*np.log(mass_ratio))

    def calculate_statistical_temperatures(self):
        """High-precision stellar temperature calculation"""
        mass_ratio = self.M/1.989e30
        R_ratio = self.R/6.957e8

        # White Dwarf specific calculations
        if self.type == 'white_dwarf':
            # Enhanced degenerate core physics
            T_core = 2.5e7 * (mass_ratio/1.018)**0.25
            # Surface temperature with proper electron degeneracy
            T_surface = 2.5e4 * (mass_ratio/1.018)**0.25
            return T_core, T_surface

        # Neutron stars
        if self.type == 'neutron_star':
            T_core = 2.0e8
            T_surface = 2.5e5
            
        # Supergiants
        elif mass_ratio > 10 and R_ratio > 100:
            T_core = 3.5e7 * (mass_ratio/16.5)**0.31
            T_surface = 3600 * (1 - 0.02*np.log(R_ratio/764.0))
            
        # Red giants
        elif R_ratio > 40:
            T_core = 1.73e7 * (mass_ratio/1.16)**0.31
            T_surface = 3910
            
        # Intermediate mass
        elif mass_ratio > 1.5:
            T_core = 2.37e7 * (mass_ratio/2.063)**0.51
            T_surface = 9940 * (mass_ratio/2.063)**0.48
            
        # Low mass stars
        elif mass_ratio < 0.3:
            T_core = 3.84e6 * (mass_ratio/0.122)**0.51
            T_surface = 3042 * (mass_ratio/0.122)**0.505
            
        # Solar-type stars
        else:
            T_core = 1.57e7 * mass_ratio**0.7
            T_surface = 5778 * mass_ratio**0.505

        return T_core, T_surface

    def calculate_quantum_corrections(self):
        """Enhanced quantum corrections with stellar type dependence"""
        compactness = self.G * self.M / (self.R * self.c**2)
        beta = self.l_planck / self.R
        mass_ratio = self.M/1.989e30
        
        if self.type == 'white_dwarf':
            # Enhanced quantum corrections for white dwarfs
            gamma = 0.85 * beta * np.sqrt(self.leech_lattice_dim/24)
            chandrasekhar_factor = (1.44 - mass_ratio)**(-0.5)
            return 1.0 + gamma * compactness * chandrasekhar_factor
        elif self.type == 'neutron_star':
            gamma = 0.95 * beta * np.sqrt(self.leech_lattice_dim/24)
        else:
            gamma = 0.55 * beta * np.sqrt(self.leech_lattice_dim/24)
            
        return 1.0 + gamma * compactness * (1 + 0.1*np.log(mass_ratio))


    def enhanced_massive_star_temperature(self, mass_ratio, R_ratio):
        T_core_sun = 1.57e7
        if mass_ratio > 10:
            return T_core_sun * mass_ratio**0.31 * (1 + 0.1*np.log(mass_ratio))
        return T_core_sun * mass_ratio**0.7

    def giant_envelope_model(self, T_surface, R_ratio):
        alpha_MLT = 2.0
        density_factor = np.exp(-R_ratio/400)  # More gradual decline
        convective_efficiency = 1 - 0.03 * np.log(R_ratio)
        mass_loss_correction = 1 - 0.02 * np.log(self.M/1.989e30)
        return T_surface * convective_efficiency * density_factor * mass_loss_correction * alpha_MLT



    def relativistic_neutron_star_corrections(self, T_core):
        """Enhanced relativistic corrections for neutron stars"""
        # Compactness parameter
        xi = 2 * self.G * self.M / (self.R * self.c**2)
        
        # TOV equation correction
        tov_factor = 1 / np.sqrt(1 - xi)
        
        # Quantum corrections
        fermi_factor = 1 + 0.15 * np.log(self.M/1.4/1.989e30)
        
        return T_core * tov_factor * fermi_factor


    def calculate_mass_loss(self):
        """Calculate mass loss rate for giant stars"""
        R_ratio = self.R/6.957e8
        if R_ratio > 100:  # Supergiants
            luminosity = self.calculate_luminosity()
            mass_factor = (self.M/1.989e30)**2.5
            return 4e-13 * (luminosity/3.828e26) * mass_factor * R_ratio
    def enhanced_mass_luminosity(self):
        """Improved mass-luminosity relation for massive stars"""
        mass_ratio = self.M/1.989e30
        if mass_ratio > 1.5:
            return mass_ratio**3.8  # Steeper relation for massive stars
        return mass_ratio**3.5

    def convective_envelope_correction(self, T_surface):
        """Apply convective envelope corrections for giants"""
        if self.type == 'red_giant':
            R_ratio = self.R/6.957e8
            return T_surface * (1 - 0.05*np.log(R_ratio))



    def calculate_magnetic_field(self):
        """Calculate magnetic field structure"""
        # Base field strength scales with mass and radius
        B_surface = 1e4 * (self.M/1.989e30) * (self.R/6.957e8)**(-2)
        
        # Get radial coordinates
        r = np.linalg.norm(self.qg.grid.points, axis=1)
        r = np.maximum(r, CONSTANTS['l_p'])
        
        # Dipole field configuration
        B_r = B_surface * (self.R/r)**3 * 2 * np.cos(theta)
        B_theta = B_surface * (self.R/r)**3 * np.sin(theta)
        
        # Add quantum corrections
        quantum_factor = 1.0 + self.gamma_eff
        B_r *= quantum_factor
        B_theta *= quantum_factor
        
        return B_r, B_theta

    def analyze_perturbations(self):
        """Analyze stellar perturbations and oscillations"""
        # Get central density with fallback
        rho_c = self.central_density()
        if rho_c is None:
            rho_c = self.calculate_initial_density()

        # Get central pressure with fallback
        P_c = self.central_pressure()
        if P_c is None:
            P_c = self.compute_total_pressure()

        # Calculate characteristic frequencies with proper error handling
        gamma = 5/3  # Adiabatic index
        
        # Handle density gradient carefully to avoid negative sqrt
        dgrad = self.density_gradient()
        N = np.sqrt(np.abs(self.G/self.R * dgrad)) * np.sign(dgrad)
        
        # Calculate acoustic frequency
        omega_acoustic = np.sqrt(gamma * P_c / (rho_c * self.R**2))
        omega_gravity = np.sqrt(self.G * self.M / self.R**3)
        
        # Lamb frequency with sound speed
        l = 2  # Angular mode number
        cs = self.sound_speed()  # Get sound speed with proper error handling
        S_l = np.sqrt(l*(l+1)) * cs / self.R
        
        return {
            'acoustic': omega_acoustic,
            'gravity': omega_gravity,
            'brunt': N,
            'lamb': S_l
        }

    def density_gradient(self):
        """Calculate density gradient for perturbation analysis"""
        rho_c = self.central_density()
        rho_surface = self.density_from_eos(self.surface_pressure(), 
                                        self.surface_temperature)
        return (rho_surface - rho_c) / self.R

    def sound_speed(self):
        """Calculate adiabatic sound speed with error handling"""
        P = self.central_pressure()
        rho = self.central_density()
        
        # Use fallback values if needed
        if P is None:
            P = self.compute_total_pressure()
        if rho is None:
            rho = self.calculate_initial_density()
            
        gamma = 5/3  # Adiabatic index
        return np.sqrt(gamma * P / rho)

    def calculate_magnetic_field(self):
        """Calculate magnetic field structure"""
        B_surface = 1e4 * (self.M/1.989e30) * (self.R/6.957e8)**(-2)
        
        r = np.linalg.norm(self.qg.grid.points, axis=1)
        r = np.maximum(r, CONSTANTS['l_p'])
        theta = np.arccos(self.qg.grid.points[:,2] / r)
        
        # Dipole field configuration
        B_r = B_surface * (self.R/r)**3 * 2 * np.cos(theta)
        B_theta = B_surface * (self.R/r)**3 * np.sin(theta)
        
        # Add quantum corrections
        quantum_factor = 1.0 + self.gamma_eff
        B_r *= quantum_factor
        B_theta *= quantum_factor
        
        return B_r, B_theta

    def calculate_oscillation_modes(self, n_modes=3):
        """Calculate stellar oscillation modes"""
        # Get perturbation frequencies
        freqs = self.analyze_perturbations()
        
        # Define angular mode number l
        l = 2  # Standard value for quadrupole modes

        # Calculate eigenmodes
        omega_p = np.array([(n+1) * np.pi * self.sound_speed() / self.R 
                            for n in range(n_modes)])
        
        omega_g = np.array([np.sqrt(l*(l+1)) * freqs['brunt'] / ((n+1) * np.pi)
                            for n in range(n_modes)])
        
        return omega_p, omega_g

    def compute_mode_energy(self, n, l):
        """Compute energy in oscillation mode (n,l)"""
        omega_p, omega_g = self.calculate_oscillation_modes()
    
        if l == 0:  # Radial modes
            E = 0.5 * self.M * omega_p[n]**2 * self.A_n[n]**2
        else:  # Non-radial modes
            E = 0.5 * self.M * omega_g[n]**2 * self.B_n[n]**2
        
        return E

    def density_gradient(self):
        """Calculate density gradient for perturbation analysis"""
        rho_c = self.central_density()
        # Get actual surface temperature value by calling the method
        T_surface = self.surface_temperature()
        P_surface = self.surface_pressure()
        
        rho_surface = self.density_from_eos(P_surface, T_surface)
        return (rho_surface - rho_c) / self.R

    def density_from_eos(self, pressure, temperature):
        """Calculate density from equation of state"""
        # Ensure non-zero temperature
        temperature = np.maximum(temperature, 1.0)  # Minimum 1K temperature floor
        
        if self.type == 'main_sequence':
            # Ideal gas + radiation pressure
            P_rad = 4 * self.sigma_sb * temperature**4 / (3 * self.c)
            P_gas = np.maximum(pressure - P_rad, 1e4)  # Minimum gas pressure floor
            return P_gas * self.m_p / (self.k_B * temperature)
                
        elif self.type == 'white_dwarf':
            # Electron degeneracy
            x = (pressure / 1e22)**(2/3)
            return 2e6 * x**1.5
                
        elif self.type == 'neutron_star':
            # Nuclear matter EOS
            return (pressure/1.6e33)**(0.4) * 1e15
                
        else:  # Red giants and others
            return np.maximum(pressure * self.m_p / (self.k_B * temperature), 1e-6)

    def surface_pressure(self):
        """Calculate surface pressure"""
        if self.type == 'neutron_star':
            return 1e30  # High surface pressure for neutron stars
        elif self.type == 'white_dwarf':
            return 1e20  # Significant surface pressure
        else:
            return self.G * self.M / (self.R**2) * 1e4  # Enhanced atmospheric pressure

    def density_gradient(self):
        """Calculate density gradient with proper error handling"""
        # Get central density with fallback
        rho_c = self.central_density()
        if rho_c is None:
            rho_c = self.calculate_initial_density()
        
        # Get surface values with proper error handling
        T_surface = self.surface_temperature()
        P_surface = self.surface_pressure()
        
        # Calculate surface density with minimum threshold
        rho_surface = np.maximum(
            self.density_from_eos(P_surface, T_surface),
            1e-6  # Minimum density floor
        )
        
        return (rho_surface - rho_c) / self.R    
    
    def surface_temperature(self):
        mass_ratio = self.M/1.989e30
        R_ratio = self.R/6.957e8
    
        if R_ratio > 100:  # Supergiants
            return self.giant_envelope_model(3600, R_ratio)
        elif mass_ratio > 1.5:  # Hot stars
            return 5778 * mass_ratio**0.48
        else:  # Main sequence
            return self.calculate_statistical_temperatures()[1]

    def compute_surface_temperature(self):
        if self.radius > 100:  # Giants
            R_ratio = self.radius/self.R_sun
            return self.T_eff * np.exp(-R_ratio/100)

    def relativistic_corrections(self):
        if self.type == 'neutron_star':
            compactness = 2*self.G*self.M/(self.R*self.c**2)
            return 1/np.sqrt(1 - compactness)

    def calculate_initial_density(self):
        """Calculate initial density based on stellar type"""
        if self.type == 'neutron_star':
            return 1e17 * (self.mass/1.4)**0.5
        elif self.type == 'white_dwarf':
            return 1e6 * (self.mass/0.6)**1.5
        elif self.type == 'red_giant':
            return 1e4 * (self.mass**1.2) * (self.radius**-2.5)
        elif self.mass < 0.5:  # Low mass stars like Proxima
            return 1.62e5 * (self.mass**-1.4)
        elif self.mass > 10:  # Massive stars
            return 1.62e5 * (self.mass**-2.2)
        else:  # Main sequence
            return 1.62e5 * (self.mass**-0.7)

    def calculate_luminosity(self):
        """Calculate stellar luminosity with enhanced physics"""
        L_sun = 3.828e26  # Solar luminosity in Watts
        
        if self.type == 'red_giant':
            # Enhanced luminosity for supergiants like Betelgeuse
            mass_ratio = self.M/1.989e30
            radius_ratio = self.R/6.957e8
            return L_sun * mass_ratio**3.8 * (1 + 0.4*np.log(radius_ratio))
        else:
            # Standard mass-luminosity relation
            return L_sun * self.enhanced_mass_luminosity()
