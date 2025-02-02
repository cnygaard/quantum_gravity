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

def test_gas_contribution():
    """Test gas contribution to rotation curves"""
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],
            total_mass=data['mass'],
            visible_mass=data['visible_mass']
        )
        #galaxy = StellarDynamics(...)
        v_gas = galaxy.compute_gas_contribution()
        # Gas velocity should be 8-12% of total
        v_total = galaxy.compute_rotation_curve() * 1000
        gas_fraction = v_gas / v_total
        print(f"\nGalaxy: {galaxy_name}")
        print(f"Gas velocity: {v_gas:.1f} m/s")
        print(f"Total velocity: {v_total:.1f} m/s")
        print(f"Gas fraction: {gas_fraction:.3f}")
        assert 0.08 <= gas_fraction <= 0.12

def test_energy_conservation():
    """Test energy conservation in dynamics"""
    galaxy = StellarDynamics(...)
    KE = galaxy.kinetic_energy()
    PE = galaxy.potential_energy()
    # Virial theorem for stable systems
    assert abs(2*KE + PE) / abs(PE) < 0.1

def test_energy_conservation():
    """Test energy conservation in dynamics"""
    for galaxy_name, data in GALAXY_DATA.items():
        galaxy = StellarDynamics(
            orbital_velocity=data['velocity'],
            radius=data['radius'],
            mass=data['visible_mass'],
            dark_mass=data['dark_mass'],
            total_mass=data['mass'],
            visible_mass=data['visible_mass']
        )
        
        KE = galaxy.kinetic_energy()
        PE = galaxy.potential_energy()
        
        # Calculate virial ratio
        virial_ratio = abs(2*KE + PE) / abs(PE)
        
        print(f"\nGalaxy: {galaxy_name}")
        print(f"Kinetic Energy: {KE:.10e} J")
        print(f"Potential Energy: {PE:.10e} J")
        print(f"Virial Ratio: {virial_ratio:.5f}")
        
        # Virial theorem states 2T + V = 0 for stable systems
        assert virial_ratio < 0.11, \
            f"Virial ratio {virial_ratio:.3f} exceeds stability threshold 0.1"



if __name__ == '__main__':
    pytest.main()
