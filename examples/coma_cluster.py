import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from physics.models.dark_matter import DarkMatterAnalysis


coma = DarkMatterAnalysis(
    observed_mass=1e14,  # Solar masses
    total_mass=1e15,     # Solar masses
    radius=3e6,          # Light years 
    velocity_dispersion=1000  # km/s
)

results = coma.compare_with_observations()

print(results)