# Quantum Gravity Framework

Experimental artifical intelligence cocreated Quantum Physics simulator
A high-performance numerical framework for quantum gravity simulations, focusing on black hole dynamics and quantum effects.

## Latest Results

### Star Simulation
- Quantum Parameters:
  * β (l_p/R): 2.32e-44
  * γ_eff: 8.15e-45
- Vacuum Energy: 2.00e-01
- Cosmological Constant: 2.51e+01
- Core Properties:
  * Mass: 1.00 M_sun
  * Radius: 1.00 R_sun
  * Central Density: 1.62e+05
  * Central Pressure: 1.32e+16
  * Core Temperature: 1.57e+08

### Cosmology Results
- Scale Factor Evolution: a = 7.39e+03
- Hubble Parameter: H = 1.01e-01
- Equation of State: w = -9.99e-01
- Acceleration: q = -7.54e+03
- Cosmic Entropy: S = 6.86e+08
- Power Spectrum: P(k) mean = 1.85e+16

## Enhanced Features

### Star Simulation
- Full stellar structure evolution
- Quantum-corrected central conditions
- Geometric-entanglement verification
- Leech lattice vacuum energy
- Visualization:
  * Core property evolution plots
  * 3D structure visualization
  * Quantum effects distribution
  * Leech lattice representation

### Cosmology Updates
- Quantum-corrected Friedmann equations
- Inflation dynamics tracking
- Enhanced scale factor evolution
- Power spectrum analysis
- Comprehensive visualization suite

## Verification Metrics
- Geometric-Entanglement Formula
- Quantum-Classical Transition
- Scale-dependent coupling
- Conservation law tracking

## Output Structure
/results
  /star
    - star_evolution.png
    - star_geometry.png
    - measurements.json
  /cosmology
    - evolution.png
    - measurements.json



## Features

- Black hole evolution with quantum corrections
- Hawking radiation and temperature calculations 
- EP=EPR correspondence tracking
- Geometric-Quantum verification
- Entanglement entropy measurements
- Parallel computation support via MPI
- Adaptive grid refinement
- Error tracking and conservation laws
- Near-horizon structure visualization
- Quantum parameter evolution tracking
- Leech lattice vacuum energy calculations [experimental]
- Dark energy emergence from quantum geometry

## Theoretical Foundations

This framework builds upon:

- Loop Quantum Gravity (Rovelli, 2004)
  - Discrete quantum geometry
  - Background independence
  - Spin networks and quantum states

- String Theory & Holography
  - AdS/CFT correspondence
  - EP=EPR implementation

- Black Hole Thermodynamics
  - Hawking radiation
  - Information preservation

### Vacuum Energy Implementation [Experimental]
The framework implements vacuum energy calculations using:
- 24-dimensional Leech lattice geometry
- M24 symmetry group effects
- Statistical convergence with lattice points
- Geometric suppression mechanisms
- Quantum-to-classical scale transitions

Key results:
- Stable vacuum energy: 2.01e-01 (Planck units)
- Cosmological constant: 2.51e+01
- Quantum coupling β: 2.32e-44
- Effective coupling γ: 8.15e-45

## Theory

The framework implements a experimental unified quantum gravity theory combining:
- Geometric-Entanglement correspondence
- EP=EPR implementation 
- Black hole evolution with quantum corrections
- Adaptive grid refinement near horizons

Key mathematical components:
- Modified Einstein equations with quantum terms
- Hawking radiation and temperature evolution
- Information preservation via EP=EPR
- Conservation law verification

### Star Simulation (Experimental)

The stellar evolution simulation (`examples/star.py`) models quantum gravitational effects in stellar objects:
- Tracks density, pressure, and temperature profiles
- Includes quantum corrections to metric components
- Handles stellar structure with proper scaling to Planck units
- Verifies geometric-entanglement relationships
- Currently under active development
- Quantum vacuum effects via Leech lattice
- Geometric-entanglement verification
- Vacuum energy evolution tracking
- Dark energy scale emergence

### Cosmology Simulation (Experimental) 
The cosmological simulation (`examples/cosmology.py`) explores quantum effects in early universe evolution:
- Models scale factor evolution with quantum corrections
- Tracks energy density and perturbation spectra
- Handles quantum bounce scenarios
- Includes inflation dynamics
- Under development with ongoing refinements

## Installation

## Linux Install Debian/Ubuntu tools

apt install python3-pip pyton3-tk build-essential openmpi-devel



### Clone the simualtor code


git clone https://github.com/cnygaard/quantum_gravity.git
cd quantum_gravity


### The simulator runs in Linux, the simulator is intended to be run in a Python virtual environment 


virtualenv .venv
source .venv/activate
pip install -r requirements.txt



## Architecture


core/      Core quantum gravity implementation
physics/   Physical observables and measurements
numerics/  Numerical methods and parallel computing
examples/  Usage examples and demonstrations
results/   Simulation results folder


## Running the Simulation

You can run the black hole simulation using the following command:


python examples/black_hole.py


You can run the cosmology simulation using the following command:


python examples/cosmology.py


You can run the star simulation using the following command:


python examples/star.py



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

## Visualization Outputs

The simulation generates comprehensive visualizations:
- 3D quantum density plots
- Event horizon evolution
- Ergosphere boundary
- Mass/temperature dynamics
- Entropy evolution
- Geometric-Entanglement verification
- Beta parameter tracking
- Gamma effective evolution
- Radiation flux measurements

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

docker build -t quantum-gravity -f Containerfile .


To build the container with Podman:

podman build -t quantum-gravity -f Containerfile .



To run the container with Docker:

docker run -v $(pwd)/results:/app/results --name quantum-sim quantum-gravity


To run the container with Podman:

podman run -v $(pwd)/results:/app/results --replace --name quantum-sim quantum-gravity


This bind mounts the local results directory to the container's /app/results directory, making simulation outputs directly accessible on the host system. The visualization plots and measurement data will be available immediately after the simulation completes.

View simulation output logs:

docker logs -f quantum-sim


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

python test_black_hole_simulation.py


## Physics Validation & Technical Details

### Geometric-Entanglement Formula
The framework implements the key relationship:
dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩

Where:
- dS² represents spacetime interval measure
- γ is the coupling constant (currently set to 0.55)
- êᵢ(x) are entanglement operators
- îᵢ(x) are information operators

### Quantum Effects Magnitude Ranges
Quantum corrections scale as:
- Black Holes: β = l_p/r_h ~ 10⁻(3-6)
- Stars: β = l_p/R ~ 10⁻(5-8) 
- Cosmology: β = l_p*H ~ 10⁻(2-4)

Effective coupling γ_eff = γβ√0.407 determines correction strength.

### Validation Metrics
The framework tracks:
1. Geometric-Entanglement Verification
   - LHS/RHS relative error < 10⁻6
   - Conservation law violations < 10⁻8
   - Quantum corrections magnitude: 0.1% - 1%

2. Physical Constraints
   - Energy conditions preserved
   - Horizon area quantization
   - Entropy bounds respected

3. Numerical Stability
   - Grid point scaling: 10³ - 10⁵ points
   - Adaptive timestep: 10⁻6 - 10⁻2 t_p
   - Error tolerances: 10⁻8 - 10⁻10

## Technical Requirements

### System Requirements
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- 8GB RAM minimum
- Multi-core CPU recommended tested on a x86 Linux Laptop

### Dependencies
- Matplotlib for visualization


### Quantum Effects as Dark Phenomena (Experimental non verified hypothesis)
The framework implements quantum gravity corrections that manifest as dark matter and dark energy effects:

Dark Matter as Quantum Gravity:
- β_galaxy = ℓ_P/R_galaxy (quantum/classical scale ratio)
- γ_eff = 2β√0.407 (effective coupling)
- Force enhancement: F_eff/F_classical ≈ 1 + γ_eff

Dark Energy as Modified Vacuum:
- ρ_vacuum = ℏ/(cℓ_P⁴) (base vacuum energy)
- ρ_modified = ρ_vacuum(1 + γ_eff) (quantum corrected)
- Λ ∝ ρ_modified (cosmological constant)

These quantum corrections naturally produce:
- Galaxy rotation curve modifications
- Large-scale structure formation
- Cosmic acceleration
