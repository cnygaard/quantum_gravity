# Physical constants (in natural units)
# Updated to v8 mathematics: Fisher Information + Entanglement Strain formulation
# with de Sitter corrected dark matter prediction
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

    # v8 Quantum Gravity Constants (Fisher Information formulation)
    'gamma_0': 0.274,               # Immirzi parameter (Engle-Noui-Perez 2010, SU(2))
    'sigma_0': 1.0,                 # Base coherence length (in l_p units)

    # v7 flat-space predictions (kept for backward compatibility and comparison)
    'xi_geometric': np.pi / 2,      # Flat-space holographic path ratio π/2 = 1.571
    'dark_matter_ratio': np.pi / (2 * 0.274),  # v7 flat-space: π/(2γ₀) ≈ 5.73

    # v8 de Sitter corrected predictions (see theory/quantum-gravity-proposal-v8.md)
    # The key insight: our universe has Λ > 0, curving information geometry into S³
    'omega_b': 0.0493,              # Baryonic matter density (Planck 2018 / BBN)
    'omega_m_v8': 0.317,            # v8 self-consistent matter density prediction
    'omega_lambda_v8': 0.683,       # v8 dark energy density (1 - omega_m)
    'omega_c_v8': 0.268,            # v8 cold dark matter density (omega_m - omega_b)
    'xi_geometric_v8': 1.487,       # de Sitter corrected: (π/2)(sin√Ω_m)/√Ω_m
    'dm_ratio_v8': 5.43,            # v8 de Sitter corrected: ξ_dS/γ₀ ≈ 5.43

    # DEPRECATED: Leech lattice constants (kept for backward compatibility)
    # In v7/v8, these are replaced by gamma_0 and the cosmic factor π/γ₀
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
        'dark_ratio': 5.43,      # v8: (π/2γ₀)(sin√Ω_m)/√Ω_m
        'dark_mass': 8.145e11,   # Updated for v8 ratio (1.5e11 × 5.43)
        'radius': 152000,        # Light years
        'velocity': 250,         # km/s
        'mass': 9.645e11         # Updated total mass (visible + dark)
    },
    'milky_way': {
        'visible_mass': 1.0e11,
        'dark_ratio': 5.43,      # v8: (π/2γ₀)(sin√Ω_m)/√Ω_m
        'dark_mass': 5.43e11,    # Updated for v8 ratio
        'radius': 87400,
        'velocity': 220,
        'mass': 6.43e11          # Updated total mass
    },
    'triangulum': {  # M33
        'visible_mass': 4.5e9,
        'dark_ratio': 5.43,      # v8: (π/2γ₀)(sin√Ω_m)/√Ω_m
        'dark_mass': 2.44e10,    # Updated for v8 ratio (4.5e9 × 5.43)
        'radius': 55000,
        'velocity': 130,
        'mass': 2.89e10          # Updated total mass
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


def coherence_length_sds(r, r_s, L, sigma_0=None):
    """
    Compute the Schwarzschild-de Sitter coherence length (v8 unified formula).

    σ_SdS(r) = ℓ_P√(1 - r_s/r - r²/L²) = ℓ_P√f(r)

    This is derived in v8 theory Section 9 from the spatial QFIM combined
    with the vacuum EFE with cosmological constant.

    Args:
        r: Radial coordinate
        r_s: Schwarzschild radius (2GM/c²)
        L: de Sitter radius = √(3/Λ)
        sigma_0: Base coherence length (defaults to CONSTANTS['sigma_0'] * l_p)

    Returns:
        Coherence length with Planck cutoff

    Limiting cases:
        - Λ → 0 (L → ∞): Recovers Schwarzschild σ(r) = ℓ_P√(1 - r_s/r)
        - M → 0 (r_s → 0): Pure de Sitter σ(r) = ℓ_P√(1 - r²/L²)
        - Both → 0: Flat space σ = ℓ_P
    """
    if sigma_0 is None:
        sigma_0 = CONSTANTS['sigma_0'] * CONSTANTS['l_p']

    # f(r) = 1 - r_s/r - r²/L² for SdS metric
    if r > 0 and L > 0:
        f = 1 - r_s / r - (r / L)**2
        ratio = max(f, 0)
    else:
        ratio = 0

    sigma = sigma_0 * np.sqrt(ratio)

    # Planck length cutoff
    return max(sigma, CONSTANTS['l_p'])


def geometric_factor_ds(omega_m):
    """
    Compute the de Sitter corrected geometric factor on S³.

    ξ_dS(x) = (π/2)(sin x)/x

    where x = √Ω_m is the patch parameter on the information sphere S³.

    This factor accounts for the positive curvature of the information geometry
    due to the cosmological constant Λ > 0. See v8 theory Section 11.1.

    Args:
        omega_m: Matter density parameter Ω_m (typically ~0.315-0.317)

    Returns:
        Geometric factor ξ_dS (typically ~1.487 for Ω_m = 0.317)

    Properties:
        - Flat-space limit (Ω_m → 0): ξ_dS → π/2 = 1.571
        - Full hemisphere (x = π/2): ξ_dS = 1
        - Always ξ_dS < π/2 for Ω_m > 0 (positive curvature reduces ratio)
    """
    x = np.sqrt(omega_m)
    if x < 1e-10:
        # Taylor expansion: sin(x)/x ≈ 1 - x²/6 for small x
        return np.pi / 2
    return (np.pi / 2) * np.sin(x) / x


def dm_ratio_v8(gamma_0=None, omega_m=None):
    """
    Compute the v8 de Sitter corrected dark matter ratio.

    M_DM/M_b = ξ_dS(x)/γ₀ = (π/2γ₀)(sin√Ω_m)/√Ω_m

    This brings the prediction from 7.4σ discrepancy (v7: 5.73) to 1.4σ (v8: 5.43).
    See v8 theory Section 11.

    Args:
        gamma_0: Immirzi parameter (default: 0.274)
        omega_m: Matter density parameter (default: 0.317 from self-consistency)

    Returns:
        Dark matter to baryonic matter ratio (typically ~5.43)
    """
    if gamma_0 is None:
        gamma_0 = CONSTANTS['gamma_0']
    if omega_m is None:
        omega_m = CONSTANTS['omega_m_v8']

    xi_ds = geometric_factor_ds(omega_m)
    return xi_ds / gamma_0


def solve_self_consistent_omega_m(omega_b=None, gamma_0=None, tol=1e-8):
    """
    Solve the v8 self-consistency equation for matter density Ω_m.

    Ω_m = Ω_b(1 + (π/2γ₀)(sin√Ω_m)/√Ω_m)

    This equation links the cosmic energy budget through the quantum gravity
    parameter γ₀. With just two inputs (Ω_b from BBN and γ₀ from LQG), we
    can predict the full cosmological energy budget.

    See v8 theory Section 11.4.

    Args:
        omega_b: Baryonic matter density (default: 0.0493)
        gamma_0: Immirzi parameter (default: 0.274)
        tol: Convergence tolerance

    Returns:
        Self-consistent Ω_m (approximately 0.317)
    """
    from scipy.optimize import brentq

    if omega_b is None:
        omega_b = CONSTANTS['omega_b']
    if gamma_0 is None:
        gamma_0 = CONSTANTS['gamma_0']

    def residual(omega_m):
        x = np.sqrt(omega_m)
        sinc_x = np.sin(x) / x if x > 1e-10 else 1.0
        rhs = omega_b * (1 + (np.pi / (2 * gamma_0)) * sinc_x)
        return omega_m - rhs

    # Root exists in (omega_b, 1) - see v8 theory Section 11.4 for proof
    return brentq(residual, omega_b, 1.0, xtol=tol)


def predict_cosmological_parameters(omega_b=None, gamma_0=None):
    """
    Predict the full cosmological energy budget from v8 self-consistency.

    With just two inputs:
        - Ω_b = 0.0493 (baryonic matter from BBN)
        - γ₀ = 0.274 (Immirzi parameter from LQG SU(2))

    The framework predicts (all within 1.5σ of Planck 2018):
        - Ω_m ≈ 0.317 (Planck: 0.315 ± 0.007) - 0.3σ
        - Ω_Λ ≈ 0.683 (Planck: 0.685 ± 0.007) - 0.3σ
        - Ω_c ≈ 0.268 (Planck: 0.264 ± 0.008) - 0.5σ
        - Ω_c/Ω_b ≈ 5.43 (Planck: 5.36 ± 0.05) - 1.4σ

    See v8 theory Section 11.4-11.5.

    Args:
        omega_b: Baryonic matter density (default: 0.0493)
        gamma_0: Immirzi parameter (default: 0.274)

    Returns:
        Dictionary with omega_m, omega_lambda, omega_c, omega_b, dm_ratio, xi_ds
    """
    if omega_b is None:
        omega_b = CONSTANTS['omega_b']
    if gamma_0 is None:
        gamma_0 = CONSTANTS['gamma_0']

    omega_m = solve_self_consistent_omega_m(omega_b, gamma_0)
    omega_lambda = 1 - omega_m
    omega_c = omega_m - omega_b
    xi_ds = geometric_factor_ds(omega_m)
    ratio = xi_ds / gamma_0

    return {
        'omega_m': omega_m,
        'omega_lambda': omega_lambda,
        'omega_c': omega_c,
        'omega_b': omega_b,
        'dm_ratio': ratio,
        'xi_ds': xi_ds,
        'gamma_0': gamma_0,
    }


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