import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import pytest
import numpy as np
from physics.models.stellar_dynamics import StellarDynamics
from constants import CONSTANTS, GALAXY_DATA, SI_UNITS

def test_dark_matter_ratio():
    """Verify dark matter ratios match observational data"""
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass']
        )
        dark_mass = galaxy.calculate_universal_dark_matter()
        calculated_ratio = dark_mass / data['visible_mass']
        # Adjust tolerance for large numbers
        relative_error = abs((calculated_ratio - data['dark_ratio'])/data['dark_ratio'])
        assert relative_error < 0.1  # 10% tolerance

def test_rotation_curves():
    """Verify rotation curves match observed velocities"""
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass']
        )
        predicted_velocity = galaxy.compute_rotation_curve()
        assert abs(predicted_velocity - data['velocity']) < 10

def test_quantum_factors():
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass']
        )
        quantum_factor = galaxy.compute_quantum_factor()
        # Use relative scaling comparison
        relative_diff = abs(quantum_factor - 1.0)  # Should be close to 1
        assert relative_diff < 0.1

def test_dark_matter_scaling():
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass']
        )
        dark_mass = galaxy.calculate_universal_dark_matter()
        # Use logarithmic comparison for large numbers
        log_ratio = np.log10(dark_mass/data['visible_mass'])
        assert abs(log_ratio - np.log10(data['dark_ratio'])) < 1.0
if __name__ == '__main__':
    pytest.main()
