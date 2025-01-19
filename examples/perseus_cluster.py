import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis


perseus = DarkMatterAnalysis(
    observed_mass=3.8e14,  # Solar masses
    total_mass=2.7e15,     # Solar masses
    radius=2.4e6,          # Light years 
    velocity_dispersion=1280  # km/s
)

results = perseus.compare_with_observations()

print(results)