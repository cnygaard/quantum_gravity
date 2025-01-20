import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.stellar_dynamics import StellarDynamics
from constants import CONSTANTS, GALAXY_DATA
import numpy as np
# Example for a typical spiral galaxy rotation
galaxy = StellarDynamics(
    orbital_velocity=220,  # km/s
    radius=50000,         # light years
    mass=1e11            # solar masses
)

rotation_curve = galaxy.compute_rotation_curve()
print(f"Beta stellar factor: {galaxy.beta * (galaxy.radius/CONSTANTS['R_sun'])}")
print(f"Quantum enhancement: {np.sqrt(1 + galaxy.beta * (galaxy.radius/CONSTANTS['R_sun']))}")
print(f"Input orbital velocity: {galaxy.orbital_velocity} km/s")
print(f"Predicted rotation velocity: {rotation_curve} km/s")

# Test different velocities and radii
# test_cases = [
#     (200, 40000, 1e11),  # Lower velocity
#     (250, 60000, 1e11),  # Higher velocity
#     (220, 30000, 1e11),  # Different radius
# ]
# def calculate_orbital_velocity(radius_ly, mass_solar):
#     """Calculate orbital velocity in km/s"""
#     G = CONSTANTS['G']
#     r = radius_ly * CONSTANTS['light_year']
#     M = mass_solar * CONSTANTS['M_sun']
    
#     # Classical calculation
#     v = np.sqrt(G * M / r)
#     v_kms = v / 1000
    
#     # Add galactic scale normalization using proton radius scaling
#     scale_factor = 1.2e-15  # Proton radius in meters
#     return v_kms * scale_factor






# test_cases = [
#     (220, 50000, 1e11),    # Reference case
#     (250, 110000, 1.5e12), #  andromeda
#     (150, 45000, 7e10),    # ngc3198
#     (130,30000, 5e10),     #  triangulum
#     (350,25000, 8e11),     #  sombrero
# ]
test_cases = [(GALAXY_DATA[galaxy]['velocity'], 
               GALAXY_DATA[galaxy]['radius'],
               GALAXY_DATA[galaxy]['visible_mass'])
              for galaxy in GALAXY_DATA]

print("\nValidation tests:")
for v, r, m in test_cases:
    test_galaxy = StellarDynamics(orbital_velocity=v, radius=r, mass=m)
    test_curve = test_galaxy.compute_rotation_curve()
    print(f"v={v}, r={r}: predicted={test_curve:.1f} km/s")

# print("Calculated orbital velocities:")
# for v_obs, r, m in test_cases:
#     v_calc = calculate_orbital_velocity(r, m)
#     print(f"r={r} ly, M={m:.1e} M☉: v_calc={v_calc:.9f} km/s (observed: {v_obs} km/s)")