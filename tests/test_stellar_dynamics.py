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
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],  # Add dark matter mass
            total_mass=data['mass'] ,
            visible_mass=data['visible_mass']  # Add total mass
        )
        print(f"\nGalaxy: {galaxy_name}")
        print(f"Visible mass: {data['visible_mass']:.2e} M☉")
        print(f"Dark matter mass: {data['dark_mass']:.2e} M☉")
        print(f"Dark matter ratio: {data['dark_ratio']:.1f}")
        #print(f"Calculated dark matter ratio: {galaxy.calculate_dark_matter_ratio():.1f}")
        dark_mass = galaxy.calculate_universal_dark_matter()
        calculated_ratio = dark_mass / data['visible_mass']
        # Adjust tolerance for large numbers
        relative_error = abs((calculated_ratio - data['dark_ratio'])/data['dark_ratio'])
        assert relative_error < 0.11  # 11% tolerance

# def test_rotation_curves():
#     """Verify rotation curves match observed velocities"""
#     for galaxy_name, data in GALAXY_DATA.items():
#         galaxy = StellarDynamics(
#             orbital_velocity=data['velocity'],
#             radius=data['radius'],
#             mass=data['visible_mass'],
#             dark_mass=data['dark_mass'],  # Add dark matter mass
#             total_mass=data['mass'] ,
#             visible_mass=data['visible_mass']  # Add total mass
#         )
#         predicted_velocity = galaxy.compute_rotation_curve()
#         assert abs(predicted_velocity - data['velocity']) 
def test_rotation_curves():
    """Verify rotation curves match observed velocities"""
    VELOCITY_TOLERANCE = 35  # km/s
    
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],
            total_mass=data['mass'],
            visible_mass=data['visible_mass']
        )
        predicted_velocity = galaxy.compute_rotation_curve()
        calculated_dark_mass = galaxy.calculate_universal_dark_matter()  # This gets the computed value
        velocity_diff = abs(predicted_velocity - data['velocity'])

        # Compare masses
        input_dark_mass = data['dark_mass']
        dark_mass_ratio = calculated_dark_mass / input_dark_mass
        
        print(f"\nGalaxy: {galaxy_name}")
        print(f"Predicted velocity: {predicted_velocity:.1f} km/s")
        print(f"Observed velocity: {data['velocity']} km/s")
        print(f"Difference: {velocity_diff:.1f} km/s")
        print(f"Mass ratio (dark/visible): {data['dark_mass']/data['visible_mass']:.1f}")
        print(f"Radius: {data['radius']} ly")
        print(f"Input dark mass: {input_dark_mass:.2e} M☉")
        print(f"Calculated dark mass: {calculated_dark_mass:.2e} M☉")
        print(f"Dark mass ratio (calc/input): {dark_mass_ratio:.2f}")
        print(f"Velocity (pred/obs): {predicted_velocity:.1f}/{data['velocity']} km/s")

        assert velocity_diff < VELOCITY_TOLERANCE, \
            f"Galaxy {galaxy_name}: predicted={predicted_velocity:.1f}, actual={data['velocity']}"


def test_quantum_factors():
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],  # Add dark matter mass
            total_mass=data['mass'] ,
            visible_mass=data['visible_mass']  # Add total mass
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
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],  # Add dark matter mass
            total_mass=data['mass'] ,
            visible_mass=data['visible_mass']  # Add total mass
        )
        dark_mass = galaxy.calculate_universal_dark_matter()
        # Use logarithmic comparison for large numbers
        log_ratio = np.log10(dark_mass/data['visible_mass'])
        assert abs(log_ratio - np.log10(data['dark_ratio'])) < 1.0
if __name__ == '__main__':
    pytest.main()
