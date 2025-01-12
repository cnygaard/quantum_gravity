# Quantum Gravity Framework

Experimental Quantum Physics simulator
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


### Clone the simualtor code

```bash
git clone https://github.com/cnygaard/quantum_gravity.git
cd quantum_gravity
```

### The simulator runs in Linux, the simulator is intended to be run in a Python virtual environment 

```bash
virtualenv .venv
source .venv/bin/activate
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

The black hole simulation generates the following outputs in `results/black_hole/` for each mass configuration:

### Visualization Files
- `evolution_M{mass}.png`: Evolution plots showing:
  - Mass evolution over time
  - Entropy changes
  - Temperature progression 
  - Geometric-Entanglement verification
  - Beta parameter evolution
  - Gamma effective parameter
  - Radiation flux
- `black_hole_geometry_M{mass}.png`: 3D and 2D visualizations of:
  - Event horizon structure
  - Quantum effects distribution
  - Ergosphere
  - Quantum density field

### Data Files  
- `measurements_M{mass}.json`: Contains:
  - Time series data
  - Mass history
  - Entropy calculations
  - Temperature measurements
  - Radiation flux values
  - Geometric-Entanglement verification metrics
  - Beta and Gamma parameters
  - Measurement timestamps

### Simulation Logs
- `simulation_M{mass}.txt`: Detailed output including:
  - Initial simulation parameters
  - Quantum Black Hole Physics at time intervals
  - Classical Parameters (Mass, Horizon Radius, Temperature, Entropy)
  - Quantum Parameters (β, γ_eff)
  - Geometric-Entanglement Formula verification
  - Final equation verification summary

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

## Verification Metrics

- `Geometric-Entanglement Formula`: Verifies the relationship dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩
  - LHS: Represents the spacetime interval measure
  - RHS: Represents the quantum geometric-entanglement contribution

## Physical Parameters

### Classical Parameters
- `Mass`: Black hole mass in Planck masses (M_p)
- `Horizon Radius`: Event horizon size in Planck lengths (l_p)
- `Temperature`: Hawking temperature in Planck temperature units (T_p)
- `Entropy`: Bekenstein-Hawking entropy in Boltzmann constant units (k_B)

### Quantum Parameters
- `β (l_p/r_h)`: Quantum scale parameter relating Planck length to horizon radius
- `γ_eff`: Effective coupling parameter for quantum corrections

These metrics verify the geometric-entanglement relationship during black hole evolution, connecting classical geometry with quantum effects through the unified verification formula.


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
