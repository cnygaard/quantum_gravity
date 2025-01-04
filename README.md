# Quantum Gravity Framework

A high-performance numerical framework for quantum gravity simulations, focusing on black hole dynamics and quantum effects.

## Features

- Black hole evolution with quantum corrections
- Hawking radiation and temperature calculations
- Entanglement entropy measurements
- Parallel computation support via MPI
- Adaptive grid refinement
- Error tracking and conservation laws

## Installation

## Linux Install Debian/Ubuntu tools
```bash
apt install python3-pip build-essential openmpi-devel
```

```bash
git clone https://github.com/username/quantum_gravity.git
cd quantum_gravity
pip install -r requirements.txt
```

## Architecture

```
core/: Core quantum gravity implementation
physics/: Physical observables and measurements
numerics/: Numerical methods and parallel computing
examples/: Usage examples and demonstrations
```

## Running Demo simulation

You can run the demo simulation using the following command:

The following code will create a simulation with a black hole of 1000 Planck masses and run it for 100 time units: 

You can change the BlackHoleSimulation mass=1000 black hole mass and run_simulation t_final=100.0 number of simulations steps to change the simulation parameters.
### Create simulation with 1000 Planck mass black hole
sim = BlackHoleSimulation(mass=1000.0)

### Run evolution
sim.run_simulation(t_final=100.0)

```bash
ipython demo_blackhole.py
```


## Usage
### Running a simulation step by step 
```python
#from quantum_gravity import QuantumGravity
from __init__ import QuantumGravity
from examples.black_hole import BlackHoleSimulation

# Create simulation with 1000 Planck mass black hole
sim = BlackHoleSimulation(mass=1000.0)

# Run evolution
sim.run_simulation(t_final=1000.0)

# Plot results
sim.plot_results()
```

# Runtime Metrics Explanation

The simulation tracks several key physical quantities during black hole evolution:

## Trinity Verification Metrics

- `spacetime_interval`: Measures the proper spacetime distance (dS²) between events, decreasing from ~880 to ~57 as the black hole evaporates, indicating spacetime geometry changes
- `entanglement_measure`: Quantifies quantum entanglement (dE²) between horizon degrees of freedom, growing from ~5e5 to ~1.1e7, showing increasing quantum correlations
- `information_metric`: Represents information flow (dI²) during evaporation, increasing from ~8e-7 to ~8e-5, tracking information release through Hawking radiation

## Physical Observables

- `Temperature`: Hawking temperature in Planck units, inversely proportional to mass (T ∝ 1/M), increasing as the black hole evaporates
- `Entropy`: Bekenstein-Hawking entropy (S = A/4ℏG), proportional to horizon area, decreasing with mass loss
- `Flux`: Hawking radiation power output (F ∝ 1/M²), increasing as the black hole becomes smaller and hotter

These metrics verify the unified theory relationship dS² = dE² + γ²dI², connecting spacetime geometry, quantum entanglement, and information flow during black hole evolution.

