import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytest
import numpy as np
from physics.models.stellar_structure import StarSimulation
from constants import CONSTANTS, SI_UNITS

def test_star_quantum_corrections():
    """Test quantum geometric effects on stellar structure"""
    star = StarSimulation(
        mass=1.0,  # Solar mass
        radius=1.0  # Solar radius
    )
    
    # Test quantum factor scaling
    quantum_factor = star.compute_quantum_factor()
    print(f"quantum_factor {quantum_factor}")
    assert 1.0 < quantum_factor < 1.1  # Should be close to 1 with small correction
    
    # Test density enhancement
    density_ratio = star.compute_density_enhancement()
    assert density_ratio > 1.0  # Should increase core density

def test_star_thermodynamics():
    """Verify stellar thermodynamic properties"""
    star = StarSimulation(
        mass=2.0,  # 2 solar masses
        radius=1.5  # 1.5 solar radii
    )
    
    # Test temperature profile
    T_profile = star.compute_temperature_profile()
    assert T_profile.core > 1e7  # Core temp > 10M K
    assert T_profile.surface < 1e4  # Surface temp < 10k K
    
    # Test pressure balance
    P_total = star.compute_total_pressure()
    P_grav = star.compute_gravitational_pressure()
    assert np.abs(P_total - P_grav)/P_grav < 0.01  # 1% tolerance

def test_star_conservation():
    """Test conservation laws in stellar evolution"""
    star = StarSimulation(
        mass=1.0,
        radius=1.0
    )
    
    # Energy conservation
    E_initial = star.total_energy()
    star.evolve(timesteps=100)
    E_final = star.total_energy()
    assert np.abs(E_final - E_initial)/E_initial < 1e-6

def test_star_entanglement():
    """Test quantum entanglement effects"""
    star = StarSimulation(
        mass=1.0,
        radius=1.0
    )
    
    # Calculate entanglement entropy
    S_ent = star.compute_entanglement_entropy()
    assert S_ent > 0  # Should be positive
    assert S_ent < star.mass * CONSTANTS['k_B']  # Upper bound

@pytest.mark.parametrize("mass,radius", [
    (0.5, 0.5),   # Red dwarf
    (1.0, 1.0),   # Sun-like
    (2.0, 1.5),   # Blue star
])
def test_star_universal_scaling(mass, radius):
    """Test universal scaling relations"""
    star = StarSimulation(mass=mass, radius=radius)
    
    # Test mass-radius relation
    quantum_factor = star.compute_quantum_factor()
    assert 1.0 < quantum_factor < 1.2
    
    # Test temperature-mass relation
    T_surface = star.compute_surface_temperature()
    T_expected = 5778 * (mass**0.5) * (radius**-0.5)
    assert np.abs(T_surface - T_expected)/T_expected < 0.1
