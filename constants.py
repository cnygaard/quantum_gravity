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

GALAXY_DATA = {
    'milky_way': {
        'visible_mass': 1e11,
        'dark_ratio': 7.0,
        'radius': 50000,
        'velocity': 220
    },
    'andromeda': {
        'visible_mass': 1.5e11,
        'dark_ratio': 8.0,
        'dark_mass': 1.2e12,
        'radius': 110000,
        'velocity': 250
    },
    'ngc3198': {
        'visible_mass': 7e10,
        'dark_ratio': 7.0,
        'radius': 45000,
        'velocity': 150
    }
}
