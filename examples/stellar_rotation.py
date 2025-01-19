# import sys
# from pathlib import Path
# sys.path.append(str(Path(__file__).parent.parent))
# from physics.models.stellar_dynamics import StellarDynamics


# # Example for a typical spiral galaxy rotation
# galaxy = StellarDynamics(
#     orbital_velocity=220,  # km/s
#     radius=50000,          # light years
#     mass=1e11              # solar masses
# )

# rotation_curve = galaxy.compute_rotation_curve()
# print(f"Predicted rotation velocity: {rotation_curve} km/s")
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.stellar_dynamics import StellarDynamics
from constants import CONSTANTS
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

test_cases = [
    (220, 50000, 1e11),    # Reference case
    (250, 110000, 1.5e12), #  andromeda
    (150, 45000, 7e10),    # ngc3198
    (130,30000, 5e10),     #  triangulum
    (350,25000, 8e11),     #  sombrero
]


print("\nValidation tests:")
for v, r, m in test_cases:
    test_galaxy = StellarDynamics(orbital_velocity=v, radius=r, mass=m)
    test_curve = test_galaxy.compute_rotation_curve()
    print(f"v={v}, r={r}: predicted={test_curve:.1f} km/s")
