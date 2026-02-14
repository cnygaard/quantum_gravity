"""
SPARC Database Validation for Master Equation Rotation Curves

Tests the master equation g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν) against
175 galaxies from the SPARC database (Lelli et al. 2016).

This provides a rigorous test with REAL observational data, not
just 3 hand-picked galaxies.

Reference: Lelli, McGaugh & Schombert 2016, AJ, 152, 157
Data: https://astroweb.case.edu/SPARC/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Dict, List, Tuple
import logging

from physics.rotation_curves_from_master_equation import MasterEquationRotationCurve
from constants import CONSTANTS, SI_UNITS

logger = logging.getLogger(__name__)


def parse_sparc_data(filepath: str) -> List[Dict]:
    """
    Parse SPARC MRT format data file.

    Returns list of galaxy dictionaries with:
    - name: Galaxy name
    - L_36: Luminosity at 3.6μm in 10^9 L_sun
    - R_disk: Disk scale length in kpc
    - R_HI: HI radius at 1 Msun/pc² in kpc (where Vflat is measured)
    - V_flat: Flat rotation velocity in km/s
    - M_HI: HI gas mass in 10^9 M_sun
    - quality: Quality flag (1=high, 2=medium, 3=low)
    """
    galaxies = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find where data starts (after the last "---" line)
    data_start = 0
    for i, line in enumerate(lines):
        if line.startswith('---'):
            data_start = i + 1

    # Parse data lines using split() - more robust than fixed positions
    # Column order: Name, T, D, e_D, f_D, Inc, e_Inc, L[3.6], e_L, Reff, SBeff,
    #               Rdisk, SBdisk, MHI, RHI, Vflat, e_Vflat, Q, Ref
    # Indices:      0     1  2  3    4    5    6      7       8    9     10
    #               11     12      13   14   15     16       17  18+
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 18:  # Need at least 18 columns
            continue

        try:
            name = parts[0]
            # Skip if name looks like a header/note
            if '=' in name or name.startswith(('Note', 'Byte', '---')):
                continue

            L_36 = float(parts[7])    # Luminosity at 3.6μm (10^9 L_sun)
            R_disk = float(parts[11]) # Disk scale length (kpc)
            M_HI = float(parts[13])   # HI mass (10^9 M_sun)
            R_HI = float(parts[14])   # HI radius (kpc) - where Vflat is measured!
            V_flat = float(parts[15]) # Flat rotation velocity (km/s)
            quality = int(parts[17])  # Quality flag

            if V_flat > 0 and L_36 > 0:  # Only include galaxies with valid data
                galaxies.append({
                    'name': name,
                    'L_36': L_36,
                    'R_disk': R_disk,
                    'R_HI': R_HI,  # Actual measurement radius
                    'M_HI': M_HI,
                    'V_flat': V_flat,
                    'quality': quality
                })
        except (ValueError, IndexError):
            continue

    return galaxies


def luminosity_to_stellar_mass(L_36: float, M_star_L: float = 0.5) -> float:
    """
    Convert 3.6μm luminosity to stellar mass.

    Args:
        L_36: Luminosity at 3.6μm in 10^9 L_sun
        M_star_L: Mass-to-light ratio (typical: 0.5 for disk galaxies)

    Returns:
        Stellar mass in M_sun
    """
    return L_36 * 1e9 * M_star_L


def validate_against_sparc(sparc_data: List[Dict], quality_filter: int = 2) -> Dict:
    """
    Validate master equation predictions against SPARC database.

    Args:
        sparc_data: List of galaxy dictionaries from parse_sparc_data
        quality_filter: Only include galaxies with quality <= this value

    Returns:
        Validation statistics
    """
    calc = MasterEquationRotationCurve()

    results = {
        'galaxies': [],
        'deviations': [],
        'abs_deviations': [],
        'n_total': 0,
        'n_good': 0,  # |deviation| < 20%
        'n_moderate': 0,  # 20% < |deviation| < 50%
        'n_poor': 0,  # |deviation| > 50%
    }

    for galaxy in sparc_data:
        if galaxy['quality'] > quality_filter:
            continue
        if galaxy['V_flat'] <= 0 or galaxy['L_36'] <= 0:
            continue

        # Convert luminosity to stellar mass
        M_stellar = luminosity_to_stellar_mass(galaxy['L_36'])

        # Add gas mass (multiply by 1.33 for helium)
        M_gas = galaxy['M_HI'] * 1e9 * 1.33 if galaxy['M_HI'] > 0 else 0

        # Total baryonic mass
        M_baryon = M_stellar + M_gas

        # Use disk scale length, or estimate from luminosity if not available
        r_scale = galaxy['R_disk'] if galaxy['R_disk'] > 0 else 2.0

        # Use actual HI radius where Vflat is measured (critical for dwarfs!)
        # Dwarfs have RHI/Rdisk ~ 10-20, while large spirals have ~3-5
        r_obs = galaxy.get('R_HI', 3.5 * r_scale)
        if r_obs <= 0:
            r_obs = 3.5 * r_scale

        # Predict rotation velocity
        try:
            v_pred = calc.compute_rotation_velocity(r_obs, M_baryon, r_scale)
        except Exception as e:
            continue

        v_obs = galaxy['V_flat']

        # Calculate deviation
        deviation = (v_pred - v_obs) / v_obs * 100
        abs_dev = abs(deviation)

        results['galaxies'].append({
            'name': galaxy['name'],
            'M_baryon': M_baryon,
            'r_scale': r_scale,
            'v_obs': v_obs,
            'v_pred': v_pred,
            'deviation': deviation,
        })

        results['deviations'].append(deviation)
        results['abs_deviations'].append(abs_dev)
        results['n_total'] += 1

        if abs_dev < 20:
            results['n_good'] += 1
        elif abs_dev < 50:
            results['n_moderate'] += 1
        else:
            results['n_poor'] += 1

    # Compute statistics
    if results['deviations']:
        results['mean_deviation'] = np.mean(results['deviations'])
        results['std_deviation'] = np.std(results['deviations'])
        results['mean_abs_deviation'] = np.mean(results['abs_deviations'])
        results['median_abs_deviation'] = np.median(results['abs_deviations'])
        results['rms_deviation'] = np.sqrt(np.mean(np.array(results['deviations'])**2))

    return results


def run_sparc_validation():
    """
    Run full SPARC validation and print results.
    """
    print("=" * 80)
    print("SPARC DATABASE VALIDATION")
    print("Master Equation: g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν)")
    print("=" * 80)

    # Try to load SPARC data
    sparc_file = '/tmp/sparc_data.txt'

    try:
        sparc_data = parse_sparc_data(sparc_file)
        print(f"\nLoaded {len(sparc_data)} galaxies from SPARC database")
    except FileNotFoundError:
        print(f"\nSPARC data file not found: {sparc_file}")
        print("Please download from: https://zenodo.org/records/16284118")
        return

    # Validate against high and medium quality data
    print("\n" + "-" * 80)
    print("VALIDATION RESULTS (Quality 1-2 galaxies only)")
    print("-" * 80)

    results = validate_against_sparc(sparc_data, quality_filter=2)

    print(f"\nTotal galaxies tested: {results['n_total']}")
    print(f"  Good (|Δ| < 20%):     {results['n_good']} ({100*results['n_good']/results['n_total']:.1f}%)")
    print(f"  Moderate (20-50%):    {results['n_moderate']} ({100*results['n_moderate']/results['n_total']:.1f}%)")
    print(f"  Poor (|Δ| > 50%):     {results['n_poor']} ({100*results['n_poor']/results['n_total']:.1f}%)")

    print(f"\nStatistics:")
    print(f"  Mean deviation:       {results['mean_deviation']:+.1f}%")
    print(f"  Std deviation:        {results['std_deviation']:.1f}%")
    print(f"  Mean |deviation|:     {results['mean_abs_deviation']:.1f}%")
    print(f"  Median |deviation|:   {results['median_abs_deviation']:.1f}%")
    print(f"  RMS deviation:        {results['rms_deviation']:.1f}%")

    # Show best and worst predictions
    sorted_galaxies = sorted(results['galaxies'], key=lambda x: abs(x['deviation']))

    print("\n" + "-" * 80)
    print("BEST 10 PREDICTIONS:")
    print("-" * 80)
    print(f"{'Galaxy':<12} {'M_baryon':<12} {'v_obs':<10} {'v_pred':<10} {'Deviation':<10}")
    for g in sorted_galaxies[:10]:
        print(f"{g['name']:<12} {g['M_baryon']:.2e} {g['v_obs']:<10.1f} {g['v_pred']:<10.1f} {g['deviation']:+.1f}%")

    print("\n" + "-" * 80)
    print("WORST 10 PREDICTIONS:")
    print("-" * 80)
    print(f"{'Galaxy':<12} {'M_baryon':<12} {'v_obs':<10} {'v_pred':<10} {'Deviation':<10}")
    for g in sorted_galaxies[-10:]:
        print(f"{g['name']:<12} {g['M_baryon']:.2e} {g['v_obs']:<10.1f} {g['v_pred']:<10.1f} {g['deviation']:+.1f}%")

    # Analyze by mass range
    print("\n" + "-" * 80)
    print("ANALYSIS BY BARYONIC MASS:")
    print("-" * 80)

    mass_bins = [
        (1e8, 1e9, "Dwarf (10⁸-10⁹ M☉)"),
        (1e9, 1e10, "Small (10⁹-10¹⁰ M☉)"),
        (1e10, 1e11, "Medium (10¹⁰-10¹¹ M☉)"),
        (1e11, 1e13, "Large (>10¹¹ M☉)"),
    ]

    for m_low, m_high, label in mass_bins:
        bin_galaxies = [g for g in results['galaxies']
                       if m_low <= g['M_baryon'] < m_high]
        if bin_galaxies:
            devs = [g['deviation'] for g in bin_galaxies]
            abs_devs = [abs(d) for d in devs]
            n_good = sum(1 for d in abs_devs if d < 20)
            print(f"{label}: N={len(bin_galaxies)}, "
                  f"Mean Δ={np.mean(devs):+.1f}%, "
                  f"Median |Δ|={np.median(abs_devs):.1f}%, "
                  f"Good={100*n_good/len(bin_galaxies):.0f}%")

    print("\n" + "=" * 80)
    print("THEORETICAL PARAMETERS:")
    print(f"  Immirzi parameter γ₀ = {CONSTANTS['gamma_0']}")
    print(f"  Dark matter ratio π/(2γ₀) = {np.pi/(2*CONSTANTS['gamma_0']):.2f}")
    print("=" * 80)

    return results


if __name__ == "__main__":
    run_sparc_validation()
