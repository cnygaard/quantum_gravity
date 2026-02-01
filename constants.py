# Physical constants (in natural units)
# Updated to v7 mathematics: Fisher Information + Entanglement Strain formulation
import numpy as np

CONSTANTS = {
    # Fundamental constants (natural units)
    'hbar': 1.0,                    # ℏ = 1
    'h': 2 * np.pi,                 # h = 2πℏ
    'c': 1.0,                       # c = 1
    'G': 1.0,                       # G = 1
    'l_p': 1.0,                     # Planck length
    't_p': 1.0,                     # Planck time
    'm_p': 1.0,                     # Planck mass
    'lambda': 1e-52,                # Cosmological constant
    'rho_planck': 1.0,              # Planck density (c⁵/ℏG²)

    # Astrophysical constants
    'M_sun': 1.989e30 / 2.176e-8,   # Solar mass in Planck units
    'R_sun': 6.957e8 / 1.616e-35,   # Solar radius in Planck units
    'k_B': 1.380649e-23,            # Boltzmann constant in J/K
    'L_sun': 3.828e26,              # Solar luminosity in watts
    'light_year': 9.461e15,         # Light year in meters
    'kpc': 3.086e19,                # Kiloparsec in Planck lengths

    # v7 Quantum Gravity Constants (Fisher Information formulation)
    'gamma_0': 0.274,               # Immirzi parameter (Meissner 2004)
    'dark_matter_ratio': np.pi / (2 * 0.274),  # π/(2γ₀) ≈ 5.73
    'sigma_0': 1.0,                 # Base coherence length (in l_p units)
    'xi_geometric': np.pi / 2,      # Holographic path ratio π/2

    # DEPRECATED: Leech lattice constants (kept for backward compatibility)
    # In v7, these are replaced by gamma_0 and the cosmic factor π/γ₀
    'LEECH_LATTICE_POINTS': 196560,   # DEPRECATED - use gamma_0
    'LEECH_LATTICE_DIMENSION': 24,    # DEPRECATED - use gamma_0
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
        'dark_ratio': 5.73,      # v7: π/(2γ₀)
        'dark_mass': 8.6e11,     # Updated for v7 ratio
        'radius': 152000,        # Light years
        'velocity': 250,         # km/s
        'mass': 1.01e12          # Updated total mass
    },
    'milky_way': {
        'visible_mass': 1.0e11,
        'dark_ratio': 5.73,      # v7: π/(2γ₀)
        'dark_mass': 5.73e11,    # Updated for v7 ratio
        'radius': 87400,
        'velocity': 220,
        'mass': 6.73e11          # Updated total mass
    },
    'triangulum': {  # M33
        'visible_mass': 4.5e9,
        'dark_ratio': 5.73,      # v7: π/(2γ₀)
        'dark_mass': 2.58e10,    # Updated for v7 ratio
        'radius': 55000,
        'velocity': 130,
        'mass': 3.03e10          # Updated total mass
    }
}


def coherence_length(r, r_s, sigma_0=None):
    """
    Compute the coherence length σ(r) from Tolman-Ehrenfest relation.

    σ(r) = σ₀√(1 - r_s/r)

    This derives from thermal equilibrium: T(r)√(-g_tt(r)) = T_∞ = constant

    Args:
        r: Radial coordinate
        r_s: Schwarzschild radius (2GM/c²)
        sigma_0: Base coherence length (defaults to CONSTANTS['sigma_0'] * l_p)

    Returns:
        Coherence length with Planck cutoff
    """
    if sigma_0 is None:
        sigma_0 = CONSTANTS['sigma_0'] * CONSTANTS['l_p']

    # Avoid singularity at horizon
    ratio = max(1 - r_s / r, 0) if r > 0 else 0
    sigma = sigma_0 * np.sqrt(ratio)

    # Planck length cutoff
    return max(sigma, CONSTANTS['l_p'])
#'mass': 1.95e11
        
#
#    'sombrero': {    # M104
#        'visible_mass': 2.4e11,
#        'dark_ratio': 9.0,
#        'dark_mass': 1.8e12,
#        'radius': 50000,
#        'velocity': 280,
#        'mass': 1.5e13
#    }