from __init__ import QuantumGravity
from examples.black_hole import BlackHoleSimulation

# Create simulation with 1000 Planck mass black hole
sim = BlackHoleSimulation(mass=1000.0)

# Run evolution
sim.run_simulation(t_final=1.0)

# Plot results
sim.plot_results()