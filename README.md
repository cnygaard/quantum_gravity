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
apt install python3-pip pyton3-tk build-essential openmpi-devel
```

```bash
git clone https://github.com/username/quantum_gravity.git
cd quantum_gravity
pip install -r requirements.txt
```

## Architecture

```
core/      Core quantum gravity implementation
physics/   Physical observables and measurements
numerics/  Numerical methods and parallel computing
examples/  Usage examples and demonstrations
results/   Simulation results folder
```

## Running the Simulation

You can run the black hole simulation using the following command:

```bash
python examples/black_hole.py
```

## Simulation Outputs

The black hole simulation generates the following output files in the `results/black_hole/` directory:

### Visualization
- `evolution.png`: Plot grid showing:
  - Black hole mass evolution over time
  - Entropy changes
  - Temperature progression
  - Hawking radiation flux

### Data Files
- `measurements_M.json`: Contains time series data of:
  - Mass history
  - Temperature measurements
  - Entropy calculations
  - Radiation flux values
  - Measurement timestamps

### Logging
- Detailed progress updates
- Trinity verification metrics
- Physical parameter tracking
- Conservation law validation

The output files provide both visual and numerical data for analysis of the black hole evolution, quantum effects, and thermodynamic properties.


## Running in Container

You can run the black hole simulation in a container:

To build the container with Docker:
```bash
docker build -t quantum-gravity -f Containerfile .
```

To build the container with Podman:
```bash
podman build -t quantum-gravity -f Containerfile .
```


To run the container with Docker:
```bash
docker run -v $(pwd)/results:/app/results --name quantum-sim quantum-gravity
```

To run the container with Podman:
```bash
podman run -v $(pwd)/results:/app/results --replace --name quantum-sim quantum-gravity
```

This bind mounts the local results directory to the container's /app/results directory, making simulation outputs directly accessible on the host system. The visualization plots and measurement data will be available immediately after the simulation completes.

View simulation output logs:
```bash
docker logs -f quantum-sim
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

## Testing

The framework includes comprehensive unit tests covering black hole simulation functionality:

### Test Coverage

- Simulation initialization and parameters
- Mass evolution and conservation
- Temperature and entropy tracking 
- Radiation flux measurements
- Trinity verification metrics

### Running Tests

Run the full test suite:
```bash
python test_black_hole_simulation.py
