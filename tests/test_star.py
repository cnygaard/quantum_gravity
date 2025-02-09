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

@pytest.mark.parametrize("mass,radius,star_type", [
    (0.08, 0.11, "Brown Dwarf"),      # Minimum stellar mass
    (0.21, 0.24, "Red Dwarf"),        # M-dwarf
    (1.0, 1.0, "Sun"),                # Solar type
    (2.1, 1.7, "Sirius A"),           # A-type
    (15.0, 6.8, "Blue Giant"),        # O-type
    (20.0, 800.0, "Red Supergiant"),  # Betelgeuse-like
    (1.4, 1e-5, "Neutron Star")       # Compact object
])
def test_stellar_structure(mass, radius, star_type):
    """Test quantum effects across stellar mass spectrum"""
    star = StarSimulation(mass=mass, radius=radius)
    
    # Verify quantum corrections
    quantum_factor = star.compute_quantum_factor()
    print(f"quantum_factor {quantum_factor}")
    assert 1.0 <= quantum_factor <= 1.1 + (1.4/mass)**0.5
    
    # Test temperature structure
    temp_profile = star.compute_temperature_profile()
    print(f"temp_profile {temp_profile}")
    assert temp_profile.core > temp_profile.surface
    
    # Verify pressure balance
    P_total = star.compute_total_pressure()
    P_grav = star.compute_gravitational_pressure()
    
    if star_type == "Neutron Star":
        # Enhanced quantum effects for compact objects
        assert quantum_factor > 1.05
        assert P_total/P_grav > 1.1
    else:
        # Standard stars follow classical limits more closely
        assert 0.95 <= P_total/P_grav <= 1.05

def test_neutron_star_physics():
    """Specific tests for neutron star quantum effects"""
    ns = StarSimulation(mass=1.4, radius=1e-5)
    
    # Test strong quantum corrections
    quantum_factor = ns.compute_quantum_factor()
    assert quantum_factor > 1.05
    
    # Verify extreme density effects
    density_enhancement = ns.compute_density_enhancement()
    assert density_enhancement > 1.1
    
    # Check quantum geometric coupling
    beta = ns.beta
    gamma_eff = ns.gamma_eff
    assert beta * gamma_eff * np.sqrt(196560/24) > 0.1

STELLAR_DATA = {
    'SUN': {
        'mass': 1.0,
        'radius': 1.0,
        'T_core': 1.57e7,
        'T_surface': 5778
    },
    'SIRIUS_A': {
        'mass': 2.063,
        'radius': 1.711,
        'T_core': 2.1e7,
        'T_surface': 9940
    },
    'PROXIMA_CENTAURI': {
        'mass': 0.122,
        'radius': 0.154,
        'T_core': 1.55e7,
        'T_surface': 3042
    },
    'BETELGEUSE': {
        'mass': 16.5,
        'radius': 764.0,
        'T_core': 3.5e7,
        'T_surface': 3600
    },
    'NEUTRON_STAR_GEMINGA': {
        'mass': 1.47,
        'radius': 1.6e-5,
        'T_core': 1.57e8,
        'T_surface': 1.6e6
    }
}

def test_stellar_temperatures():
    """Verify temperature calculations against observed stars"""
    for star_name, data in STELLAR_DATA.items():
        star = StarSimulation(mass=data['mass'], radius=data['radius'])
        
        # Test core temperature
        T_core = star.compute_temperature_profile().core
        T_core_error = abs(T_core - data['T_core'])/data['T_core']
        
        # Test surface temperature
        T_surface = star.compute_surface_temperature()
        T_surface_error = abs(T_surface - data['T_surface'])/data['T_surface']
        
        # Standard stars: 10% tolerance
        # Compact objects: 20% tolerance due to quantum effects
        tolerance = 0.2 if 'NEUTRON' in star_name else 0.1
        
        assert T_core_error < tolerance
        assert T_surface_error < tolerance
