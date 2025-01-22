import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytest
import numpy as np
from physics.models.stellar_dynamics import StellarDynamics
from constants import CONSTANTS, GALAXY_DATA

def test_quantum_correction_scaling():
    """
    Test if quantum corrections scale properly with mass and radius
    Should decrease with larger scales
    """
    # Test galaxies at different scales
    small_galaxy = StellarDynamics(
        orbital_velocity=150,
        radius=1e3,    # 1000 ly
        mass=1e8       # 100M solar masses
    )
    
    large_galaxy = StellarDynamics(
        orbital_velocity=300,
        radius=1e5,    # 100,000 ly
        mass=1e12      # 1T solar masses
    )
    
    # Quantum effects should be stronger at smaller scales
    small_correction = small_galaxy.compute_quantum_factor() - 1
    large_correction = large_galaxy.compute_quantum_factor() - 1
    
    assert small_correction > large_correction
    assert small_correction > 0
    assert large_correction > 0

def test_velocity_curve_flatness():
    """
    Test if quantum corrections produce flat rotation curves
    without explicit dark matter
    """
    galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000,
        mass=1e11
    )
    
    # Test velocities at different radii
    radii = np.logspace(3, 5, 10)  # 1 kly to 100 kly
    velocities = []
    
    for r in radii:
        galaxy.radius = r
        v = galaxy.compute_rotation_curve()
        velocities.append(v)
    
    velocities = np.array(velocities)
    
    # Calculate velocity variation
    velocity_variation = np.std(velocities) / np.mean(velocities)
    
    # Should be nearly flat (within 10%)
    assert velocity_variation < 0.1

def test_geometric_origin():
    """
    Test if force enhancement comes from geometric effects
    not just parameter fitting
    """
    galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000,
        mass=1e11
    )
    
    # Calculate force enhancement
    enhancement = galaxy.compute_geometric_enhancement()
    
    # Test scaling with lattice dimension
    def calc_with_dim(dim):
        CONSTANTS['LEECH_LATTICE_DIMENSION'] = dim
        return galaxy.compute_geometric_enhancement()
    
    # Should scale with geometric factors
    enh_24d = calc_with_dim(24)  # Leech lattice
    enh_8d = calc_with_dim(8)    # E8 lattice
    
    # Higher dimensional lattices should give stronger enhancement
    assert enh_24d > enh_8d

def test_quantum_classical_transition():
    """
    Test proper transition to classical behavior at large scales
    """
    galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000,
        mass=1e11
    )
    
    # Get baseline quantum factor
    base_factor = galaxy.compute_quantum_factor()
    
    # Scale up the galaxy
    scaled_galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000 * 1000,  # 1000x larger
        mass=1e11 * 1000
    )
    
    scaled_factor = scaled_galaxy.compute_quantum_factor()
    
    # Quantum effects should decrease with scale
    assert scaled_factor < base_factor
    # Should approach classical (1.0)
    assert abs(scaled_factor - 1.0) < abs(base_factor - 1.0)

def test_virial_scaling():
    """
    Test if quantum corrections maintain virial theorem
    while explaining dark matter effects
    """
    galaxy = StellarDynamics(
        orbital_velocity=220,
        radius=50000,
        mass=1e11
    )
    
    # Calculate kinetic and potential energies
    T = galaxy.kinetic_energy()
    V = galaxy.potential_energy()
    
    # Virial theorem: 2T + V = 0 should still hold
    # with quantum corrections
    virial_ratio = -2 * T / V
    
    # Should be close to 1.0 (within 5%)
    assert abs(virial_ratio - 1.0) < 0.05

@pytest.mark.parametrize("velocity,radius,mass", [
    (150, 10000, 1e9),   # Dwarf galaxy
    (220, 50000, 1e11),  # Milky Way-like
    (350, 100000, 1e12), # Giant elliptical
])
def test_universal_scaling(velocity, radius, mass):
    """
    Test if quantum corrections show universal scaling
    across different galaxy types
    """
    galaxy = StellarDynamics(velocity, radius, mass)
    
    # Calculate dimensionless parameters
    quantum_factor = galaxy.compute_quantum_factor()
    beta = galaxy._compute_beta()
    
    # Should scale consistently with mass/radius
    expected_beta = 2.32e-44 * (radius/CONSTANTS['R_sun'])
    
    # Test scaling relation (within order of magnitude)
    assert abs(np.log10(beta/expected_beta)) < 1

if __name__ == '__main__':
    pytest.main()