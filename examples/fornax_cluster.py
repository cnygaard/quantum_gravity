import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis


fornax = DarkMatterAnalysis(
    observed_mass=1.2e14,  # Solar masses
    total_mass=7.8e14,     # Solar masses
    radius=1.8e6,          # Light years 
    velocity_dispersion=374  # km/s
)

results = fornax.compare_with_observations()

print(results)