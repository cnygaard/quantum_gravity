import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis


virgo = DarkMatterAnalysis(
    observed_mass=4.2e14,  # Solar masses
    total_mass=1.2e15,     # Solar masses
    radius=2.2e6,          # Light years 
    velocity_dispersion=800  # km/s
)

results = virgo.compare_with_observations()

print(results)