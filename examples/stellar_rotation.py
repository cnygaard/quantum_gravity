import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.stellar_dynamics import StellarDynamics


# Example for a typical spiral galaxy rotation
galaxy = StellarDynamics(
    orbital_velocity=220,  # km/s
    radius=50000,          # light years
    mass=1e11              # solar masses
)

rotation_curve = galaxy.compute_rotation_curve()
print(f"Predicted rotation velocity: {rotation_curve} km/s")
