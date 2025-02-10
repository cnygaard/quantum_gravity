import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytest
import numpy as np
from physics.models.stellar_dynamics import StellarDynamics
from physics.quantum_geometry import QuantumGeometry
from constants import CONSTANTS, SI_UNITS, CONVERSIONS, PLANCK_UNITS

def test_quantum_correction_scaling():
    """Test quantum corrections scale properly with mass and radius"""
    # small_galaxy = StellarDynamics(
    #     orbital_velocity=150,
    #     radius=1e3 * SI_UNITS['ly_si'],
    #     mass=1e8 * SI_UNITS['M_sun_si'],
    #     dark_mass=8e8 * SI_UNITS['M_sun_si'],
    #     total_mass=9e8 * SI_UNITS['M_sun_si'],
    #     visible_mass=1e8 * SI_UNITS['M_sun_si']
    # )
    
    # large_galaxy = StellarDynamics(
    #     orbital_velocity=300,
    #     radius=1e5,
    #     mass=1e12,
    #     dark_mass=8e12,
    #     total_mass=9e12,
    #     visible_mass=1e12
    # )
def test_quantum_correction_scaling():
    """Test quantum corrections scale properly with mass and radius"""
    # Dwarf galaxy (like Sculptor Dwarf)
    dwarf_galaxy = StellarDynamics(
        orbital_velocity=150,
        radius=1e3,  # Light years
        mass=1e8,    # Solar masses
        dark_mass=8e8,
        total_mass=9e8,
        visible_mass=1e8
    )

    # Intermediate galaxy (like LMC)
    intermediate_galaxy = StellarDynamics(
        orbital_velocity=200,
        radius=5e4,  # Light years
        mass=1e10,   # Solar masses
        dark_mass=8e10,
        total_mass=9e10,
        visible_mass=1e10
    )

    # Medium galaxy (like Milky Way)
    medium_galaxy = StellarDynamics(
        orbital_velocity=250,
        radius=8e4,
        mass=1e11,
        dark_mass=8e11,
        total_mass=9e11,
        visible_mass=1e11
    )

    # Large galaxy (like Andromeda)
    large_galaxy = StellarDynamics(
        orbital_velocity=300,
        radius=1e5,
        mass=1e12,
        dark_mass=8e12,
        total_mass=9e12,
        visible_mass=1e12
    )


    # Calculate quantum corrections
    dwarf_correction = dwarf_galaxy.compute_quantum_factor() - 1
    intermediate_correction = intermediate_galaxy.compute_quantum_factor() - 1
    medium_correction = medium_galaxy.compute_quantum_factor() - 1
    large_correction = large_galaxy.compute_quantum_factor() - 1
    print(f"dwarf_correction, intermediate_correction, medium_correction, large_correction: {dwarf_correction, intermediate_correction, medium_correction, large_correction}")
    
    #assert dwarf_correction > large_correction
        # Verify quantum corrections decrease with increasing galaxy size
    assert dwarf_correction > intermediate_correction > medium_correction > large_correction

def test_velocity_curve_flatness():
    """Test rotation curve flatness"""
    galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000,
        mass=1e11,
        dark_mass=8e11,
        total_mass=9e11,
        visible_mass=1e11
    )
    
    radii = np.logspace(3, 5, 10)
    velocities = []
    for r in radii:
        galaxy.radius = r
        velocities.append(galaxy.compute_rotation_curve())
    
    velocity_variation = np.std(velocities) / np.mean(velocities)
    assert velocity_variation < 0.1

@pytest.mark.parametrize("velocity,radius,mass", [
    (150, 10000, 1e9),   # Dwarf galaxy
    (220, 50000, 1e11),  # Milky Way-like
    (350, 100000, 1e12), # Giant elliptical
])
def test_universal_scaling(velocity, radius, mass):
    """Test universal scaling across galaxy types"""
    dark_mass = mass * 8
    total_mass = mass + dark_mass
    visible_mass = mass
    
    galaxy = StellarDynamics(
        orbital_velocity=velocity,
        radius=radius,
        mass=mass,
        dark_mass=dark_mass,
        total_mass=total_mass,
        visible_mass=visible_mass
    )
    
    quantum_factor = galaxy.compute_quantum_factor()
    assert 1.0 < quantum_factor < 1.1

if __name__ == '__main__':
    pytest.main()