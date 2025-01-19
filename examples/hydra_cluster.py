import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis


hydra = DarkMatterAnalysis(
    observed_mass=2.1e14,  # Solar masses
    total_mass=1.9e15,     # Solar masses
    radius=2.8e6,          # Light years 
    velocity_dispersion=784  # km/s
)

results = hydra.compare_with_observations()

print(results)