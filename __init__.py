# quantum_gravity/__init__.py
"""
Efficient Quantum Gravity Framework
=================================

A framework for numerical simulations of quantum gravity with efficient
implementation of core algorithms and parallel computing capabilities.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Dict
import json

from physics.observables import (
    AreaObservable,
    ADMMassObservable,
    BlackHoleTemperatureObservable,
    HawkingFluxObservable,
    ScaleFactorObservable,
    EnergyDensityObservable,
    QuantumCorrectionsObservable,
    PerturbationSpectrumObservable,
    StellarTemperatureObservable,
    PressureObservable
)

# Make key components available at package level
from core.grid import AdaptiveGrid
from core.state import QuantumState
from core.operators import QuantumOperator
from core.evolution import TimeEvolution
#from core.evolution import TimeEvolution

from numerics.errors import ErrorTracker
from physics.conservation import ConservationLawTracker
from utils.io import QuantumGravityIO

# Package metadata
__version__ = '0.1.0'
__author__ = 'Christian Nygaard'
__license__ = 'MIT'

# Physical constants (in natural units)
CONSTANTS = {
    'hbar': 1.0,                    # ℏ = 1
    'c': 1.0,                       # c = 1
    'G': 1.0,                       # G = 1
    'l_p': 1.0,                     # Planck length
    't_p': 1.0,                     # Planck time
    'm_p': 1.0,                     # Planck mass
    'lambda': 1e-52                 # Cosmological constant
}

# Default configuration
DEFAULT_CONFIG = {
    'grid': {
        'points_min': 1000,
        'points_max': 10000,
        'adaptive_threshold': 1e-6,
        'refinement_factor': 2.0
    },
    'evolution': {
        'dt_min': 1e-6,
        'dt_max': 1e-2,
        'rtol': 1e-6,
        'atol': 1e-8,
        'method': 'adaptive'
    },
    'numerics': {
        'eps_cut': 1e-10,
        'max_iterations': 1000,
        'convergence_threshold': 1e-8
    },
    'parallel': {
        'load_balance': 'dynamic',
        'comm_pattern': 'neighbor',
        'chunk_size': 1000
    },
    'io': {
        'output_dir': './output',
        'checkpoint_interval': 100
    }
}


class QuantumGravityConfig:
    """Configuration manager for quantum gravity framework."""
    def __init__(self, config_path: str = None):
        self.config = {
            'grid': {
                'points_min': 1000,
                'points_max': 10000,
                'adaptive_threshold': 1e-6,
                'refinement_factor': 2.0
            },
            'numerics': {
                'eps_cut': 1e-10,
                'max_iterations': 1000,
                'convergence_threshold': 1e-8
            },
            'io': {
                'output_dir': './output',
                'checkpoint_interval': 100
            }
        }
        self.io = QuantumGravityIO(self.config['io']['output_dir'])

        if config_path:
            self.load_config(config_path)

        # Setup logging
        #self._setup_logging()

    def load_config(self, config_path: str) -> None:
        """Load configuration from JSON file."""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            self._update_recursive(self.config, user_config)
            logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.error(f"Error loading configuration: {str(e)}")
            raise

    def save_config(self, config_path: str) -> None:
        """Save current configuration to JSON file."""
        try:
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2)
            logging.info(f"Saved configuration to {config_path}")
        except Exception as e:
            logging.error(f"Error saving configuration: {str(e)}")
            raise

    def _update_recursive(self, d: Dict, u: Dict) -> None:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _setup_logging(self) -> None:
        """Configure logging system."""
        log_dir = Path(self.config['io']['output_dir'])
        log_dir.mkdir(exist_ok=True)

        logging.basicConfig(
            filename=str(log_dir / 'quantum_gravity.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Also log to console
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logging.getLogger('').addHandler(console)

def configure_logging(mass: float = None, simulation_type: str = 'black_hole'):
    """Configure unified logging for quantum gravity framework.
    
    Args:
        mass: Mass parameter for black hole simulations
        simulation_type: Type of simulation ('black_hole' or 'cosmology')
    """
    # Clear existing handlers
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
            
    # Create output directory based on simulation type
    output_dir = Path(f"results/{simulation_type}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging to both file and console
    handlers = [logging.StreamHandler()]
    
    # Add appropriate file handler based on simulation type
    if simulation_type == 'black_hole':
        log_file = f"simulation_M{mass:.0f}.txt"
    else:
        log_file = "simulation.txt"
        
    handlers.append(logging.FileHandler(str(output_dir / log_file), mode='w'))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=handlers
    )


class QuantumGravity:
    """Main interface for quantum gravity framework."""

    def __init__(self, config_path: str = None):
        """Initialize quantum gravity framework."""
        #configure_logging()  # Set up logging first
        # Load default configuration
        self.config = QuantumGravityConfig(config_path)
        
        # Override output directory to use results/
        self.config.config['io']['output_dir'] = ''
        
        # Initialize IO handler with correct path
        from utils.io import QuantumGravityIO
        self.io = QuantumGravityIO(self.config.config['io']['output_dir'])
        
        # Initialize grid first
        self.grid = AdaptiveGrid(
            eps_threshold=self.config.config['grid']['adaptive_threshold'],
            l_p=1.0
        )

        # Set initial points before creating state
        initial_points = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        self.grid.set_points(initial_points)

        # Initialize state with populated grid
        self.state = QuantumState(
           grid=self.grid,
           initial_mass=1.0,
           eps_cut=self.config.config['numerics']['eps_cut']
        )

        # Initialize quantum operators
        self.operators = {
            'hamiltonian': self._create_hamiltonian(),
            'momentum': self._create_momentum_operator(),
            'angular_momentum': self._create_angular_momentum(),
            'constraints': self._create_constraints()
        }

        # Add physics namespace
        self.physics = self._init_physics()

        logging.info("Quantum gravity framework initialized")

    def _create_hamiltonian(self):
        """Create Hamiltonian operator for energy evolution."""
        return QuantumOperator(
            self.grid,
            operator_type='hamiltonian',
            coupling_constant=CONSTANTS['G']
        )

    def _create_momentum_operator(self):
        """Create momentum operator for translations."""
        return QuantumOperator(
            self.grid,
            operator_type='momentum',
            dimensions=3
        )

    def _create_angular_momentum(self):
        """Create angular momentum operator for rotations."""
        return QuantumOperator(
            self.grid,
            operator_type='angular_momentum',
            dimensions=3
        )

    def _create_constraints(self):
        """Create constraint operators for gauge invariance."""
        return [
            QuantumOperator(
                self.grid,
                operator_type='constraint',
                constraint_index=i
            )
            for i in range(4)  # 4 constraints for diffeomorphism invariance
        ]

    def _init_physics(self):
        """Initialize physics components."""
        class Physics:
            pass

        physics = Physics()
        physics.AreaObservable = AreaObservable
        physics.ADMMassObservable = ADMMassObservable
        physics.BlackHoleTemperatureObservable = BlackHoleTemperatureObservable
        physics.HawkingFluxObservable = HawkingFluxObservable
        physics.ScaleFactorObservable = ScaleFactorObservable
        physics.EnergyDensityObservable = EnergyDensityObservable
        physics.QuantumCorrectionsObservable = QuantumCorrectionsObservable
        physics.PerturbationSpectrumObservable = PerturbationSpectrumObservable
        physics.StellarTemperatureObservable = StellarTemperatureObservable
        physics.PressureObservable = PressureObservable
        return physics

    def _load_config(self, config_path: str = None) -> Dict:
        """Load configuration with defaults."""
        default_config = {
            'grid': {
                'points_min': 1000,
                'points_max': 10000,
                'adaptive_threshold': 1e-6,
                'refinement_factor': 2.0
            },
            'numerics': {
                'eps_cut': 1e-10,
                'max_iterations': 1000,
                'convergence_threshold': 1e-8
            }
        }

        if config_path:
            try:
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                # Update defaults with user config
                self._update_recursive(default_config, user_config)
            except Exception as e:
                logging.warning(
                    f"Failed to load config from {config_path}: {e}"
                )
                logging.info(
                    "Using default configuration"
                )
        return default_config

    def _update_recursive(self, d: Dict, u: Dict) -> Dict:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = self._update_recursive(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def _setup_grid(self):
        logging.debug(f"R_star = {self.R_star:.3e}")
        
        # Generate and validate points
        points = self._generate_grid_points()
        logging.debug(f"Points shape: {points.shape}")
        logging.debug(f"Points stats: min={np.min(points):.3e}, max={np.max(points):.3e}")
        
        # Check for invalid values
        if not np.all(np.isfinite(points)):
            invalid_mask = ~np.isfinite(points)
            invalid_points = points[invalid_mask]
            invalid_indices = np.where(invalid_mask)
            logging.error(f"Invalid points at indices {invalid_indices}")
            logging.error(f"Invalid values: {invalid_points}")
            raise ValueError("NaN/Inf found in grid points")
            
        # Clamp to safe range if needed
        max_allowed = 1e308  # Max float64 ~1e308
        if np.any(np.abs(points) > max_allowed):
            points = np.clip(points, -max_allowed, max_allowed)
            logging.warning("Points clipped to safe range")
            
        self.qg.grid.set_points(points)

    def _generate_grid_points(self):
        num_points = 1000
        # Use smaller scale factor to avoid overflow
        scale = min(self.R_star, 1e100)
        points = np.random.rand(num_points, 3) * scale
        return points

    def run_simulation(self, t_final: float, callback=None) -> None:
        """Run simulation to specified time."""
        # Initialize evolution if not already done
        if not hasattr(self, 'evolution'):
            # Create error tracker
            error_tracker = ErrorTracker(
                grid=self.grid,
                base_tolerances=self.config.config['numerics']
            )

            # Create conservation tracker
            conservation_tracker = ConservationLawTracker(self.grid)

            self.evolution = TimeEvolution(
                grid=self.grid,
                config={
                    'dt': 0.01,
                    'error_tolerance': 1e-6
                },
                error_tracker=error_tracker,
                conservation_tracker=conservation_tracker
            )

        # Add progress tracking
        current_t = 0.0
        step = 0
        progress_interval = t_final / 100
        next_report = progress_interval

        while current_t < t_final:
            self.state = self.evolution.step(self.state)
            current_t += self.evolution.dt

            # Progress reporting
            if current_t >= next_report:
                progress = (current_t / t_final) * 100
                logging.info(
                    f"Simulation progress: {progress:.1f}% "
                    f"(t={current_t:.2f}/{t_final})"
                )
                next_report += progress_interval

            if callback:
                callback(self.state, current_t, step)
            step += 1

        logging.info(f"Simulation completed to time {t_final}")

__all__ = [
    'QuantumGravity',
    'QuantumGravityConfig',
    'CONSTANTS',
    'AdaptiveGrid',
    'QuantumState',
    'QuantumOperator',
    'TimeEvolution',
    'QuantumGravityIO'
]
