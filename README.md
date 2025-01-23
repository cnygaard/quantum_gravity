# Quantum Gravity Framework

Experimental artifical intelligence cocreated Quantum Physics simulator
A numerical framework for quantum gravity simulations, focusing on black hole dynamics and quantum effects.

#### Core Features
Physics Simulations

1. Black Hole Evolution

* Hawking radiation tracking
* Horizon dynamics
* Information preservation
* Temperature evolution

2. Dark Matter Analysis

* Quantum geometric coupling
* Rotation curve predictions
* Galactic structure modeling
* Leech lattice integration

3. Cosmological Evolution

* Scale factor dynamics
* Quantum bounce handling
* Inflation modeling
* Structure formation

#### Technical Capabilities
* Adaptive grid refinement
* Conservation law tracking
* Error analysis system
* Visualization toolkit
* Experimental Mathematical Foundation


## Current Status & Features
- Black hole evolution: Working (known LHS/RHS discrepancy ~37%)
- Galaxy dynamics: 16/18 tests passing (quantum corrections need refinement)  
- Star/Cosmology: Early development stage

#### Key Relationships [Experimental]
1. Geometric-Entanglement Formula:
dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩

Where:
- dS² represents spacetime interval measure
- γ is the coupling constant (currently set to 0.55)
- êᵢ(x) are entanglement operators
- îᵢ(x) are information operators


2. Modified Friedmann Equation:
H² = (8πG/3)ρ(1 - ρ/ρc)


3. Quantum Corrections:
β = l_p/R  # Scale parameter
γ_eff = γβ√0.407  # Effective coupling


#### Usage Examples
#### Black Hole Simulation

```
from quantum_gravity import BlackHoleSimulation

# Initialize simulation
sim = BlackHoleSimulation(
    mass=1000,  # Solar masses
    grid_points=10000
)

# Run evolution
sim.evolve(t_final=1000)

# Analyze results
sim.plot_evolution()
sim.verify_conservation()
```

#### Dark Matter Analysis
```
from quantum_gravity import DarkMatterAnalysis

# Setup galaxy analysis
galaxy = DarkMatterAnalysis(
    visible_mass=1e11,  # Solar masses
    radius=50000,       # Light years
    velocity=220        # km/s
)

# Calculate quantum effects
results = galaxy.compute_rotation_curve()
galaxy.plot_results()
```

#### Installation

Requirements
Python 3.8+
Scientific Python stack

#### Setup
```
git clone https://github.com/username/quantum_gravity.git
cd quantum_gravity
pip install -r requirements.txt
```

### Framework Architecture 

```bash
quantum_gravity/
├── core/           # Core implementation
    grid.py         # Grid
    state.py        # State management
    operators.py    # Operators
    state.py        # State handling
├── physics/        # Physics modules
    conservation.py # Physical conservation laws
    entaglement.py  # Entaglement
    observables.py  # Physical observables
    verification.py # Physics verification
├── numerics/       # Numerical methods
    errors.py       # Error handling
    integrator.py   # Integrator
    parellel.py     # MPI implmentation not done
├── utils/  # Plotting tools
    visualizations  # Visualization helper
    io.py           # io handling
└── examples/       # Usage examples
    black_hole.py   # Black hole simulation 
    cosmology.py    # Cosmology simulation
    star.py         # Star simulation
└── results/        # Simulation output results files png and txt
    black_hole      # Black hole simulation 2d and 3d plots
    star            # Star simulation 
    cosmology       # Cosmology simulation plot
```

#### Verification Metrics

The framework tracks:

1. Conservation Laws

* Energy conservation
* Angular momentum
* Information preservation

2. Physical Constraints
* Subluminal velocities
* Positive energy conditions
* Entropy bounds

3. Numerical Stability
* Grid convergence
* Error propagation
* Conservation violations

#### Documentation
Detailed guides available in docs/:

Theory Guide: Mathematical foundation
API Reference: Function documentation
Examples: Additional scenarios
Validation: Framework verification


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


### Dark Matter Implementation
The framework implements dark matter effects through quantum geometric coupling:

- Scaling Parameters:
  * Dark matter ratio: 7.2 (typical for spiral galaxies 5:1 to 10:1)
  * Radius-dependent scaling: R/R_sun * 1e-15
  * Beta universal: β * lattice_factor * radius_scale


### Quantum-Geometric Coupling
- Beta parameter: 2.32e-44 * (R/R_sun)
- Gamma effective: 8.15e-45
- Total mass calculation: M_total = M * dark_matter_factor * (1 + β_universal)


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



### The simulator runs in Linux, the simulator is intended to be run in a Python virtual environment 


virtualenv .venv
source .venv/activate
pip install -r requirements.txt

## Running the Simulation

You can run the black hole simulation using the following command:

```bash
python examples/black_hole.py
```

You can run the cosmology simulation using the following command:

```bash
python examples/cosmology.py
```

You can run the star simulation using the following command:

```bash
python examples/star.py
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

The framework includes unit tests covering black hole simulation functionality:

### Test Coverage

- Simulation initialization and parameters
- Mass evolution and conservation
- Temperature and entropy tracking 
- Radiation flux measurements
- Trinity verification metrics

### Running Tests

Run the full test suite:

```
python test_black_hole_simulation.py
```

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

### Dark Matter Analysis [Highly Experimental]
- Quantum Parameters:
  * β (scaled): 2.32e-44 [quantum scale parameter]
  * γ_eff: 8.15e-45 [effective coupling]
- Dark Matter Ratio: 7.2 [apparent/visible mass]
- Radius-dependent scaling implemented through Leech lattice geometry
- Geometric coupling through 24-dimensional Leech lattice structure

Note: This is a highly experimental framework exploring a novel theoretical approach 
to dark matter through geometric effects rather than particles. All results are 
preliminary and require further validation. The framework suggests dark matter 
effects might emerge from spacetime warping into higher dimensions, but this 
remains an untested hypothesis.

This research was developed with AI assistance as an exploration of theoretical 
possibilities in quantum gravity.