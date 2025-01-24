# Physical constants (in natural units)
import numpy as np

CONSTANTS = {
    'hbar': 1.0,                    # ℏ = 1
    'h': 2 * np.pi,                 # h = 2πℏ
    'c': 1.0,                       # c = 1
    'G': 1.0,                       # G = 1
    'l_p': 1.0,                     # Planck length
    't_p': 1.0,                     # Planck time
    'm_p': 1.0,                     # Planck mass
    'lambda': 1e-52,                # Cosmological constant
    'rho_planck': 1.0,              # Planck density (c⁵/ℏG²)
    'M_sun': 1.989e30 / 2.176e-8,   # Solar mass in Planck units
    'R_sun': 6.957e8 / 1.616e-35,   # Solar radius in Planck units
    'k_B': 1.380649e-23,            # Boltzmann constant in J/K
    'L_sun': 3.828e26,              # Solar luminosity in watts
    'LEECH_LATTICE_POINTS': 196560, # Default leech lattice size in number of points
    'LEECH_LATTICE_DIMENSION': 24,   # Leech lattice dimension
    'light_year': 9.461e15,         # Light year in meters  
}
PLANCK_UNITS = CONSTANTS

# SI Units (for galactic/stellar calculations)
SI_UNITS = {
    'G_si': 6.674e-11,        # Gravitational constant [m³ kg⁻¹ s⁻²]
    'c_si': 2.998e8,          # Speed of light [m/s]
    'M_sun_si': 1.989e30,     # Solar mass [kg]
    'R_sun_si': 6.957e8,      # Solar radius [m]
    'ly_si': 9.461e15,        # Light year [m]
    'k_B_si': 1.380649e-23    # Boltzmann constant [J/K]
}

# Conversion Factors (SI to Planck)
CONVERSIONS = {
    'mass_to_planck': SI_UNITS['M_sun_si'] / 2.176e-8,    # Solar mass in Planck units
    'length_to_planck': SI_UNITS['R_sun_si'] / 1.616e-35, # Solar radius in Planck units
    'time_to_planck': SI_UNITS['ly_si'] / 5.391e-44       # Light year in Planck time
}

GALAXY_DATA = {
    'andromeda': {
        'visible_mass': 1.5e11,  # Solar masses
        'dark_ratio': 8.0,
        'dark_mass': 1.2e12,
        'radius': 152000,        # Light years
        'velocity': 250,         # km/s
        'mass': 1.5e12
    },
    'milky_way': {
        'visible_mass': 1.0e11,
        'dark_ratio': 7.2,
        'dark_mass': 7.2e11,
        'radius': 87400,
        'velocity': 220,
        'mass': 2.06e11
    },
    'triangulum': {  # M33
        'visible_mass': 4.5e9,
        'dark_ratio': 5.5,
        'dark_mass': 5.05e10,
        'radius': 60000,
        'velocity': 130,
        'mass': 1.95e11
    },
    'sombrero': {    # M104
        'visible_mass': 2.4e11,
        'dark_ratio': 9.0,
        'dark_mass': 1.8e12,
        'radius': 50000,
        'velocity': 280,
        'mass': 1.5e13
    }
}
