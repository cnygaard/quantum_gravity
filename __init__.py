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
    PerturbationSpectrumObservable
)

# Make key components available at package level
from core.grid import AdaptiveGrid
from core.state import QuantumState
from core.operators import QuantumOperator
from core.evolution import TimeEvolution
#from core.evolution import TimeEvolution

from numerics.errors import ErrorTracker
from physics.conservation import ConservationLawTracker


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

        if config_path:
            self.load_config(config_path)

        # Setup logging
        self._setup_logging()

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


class QuantumGravity:
    """Main interface for quantum gravity framework."""

    def __init__(self, config_path: str = None):
        """Initialize quantum gravity framework."""
        # Load default configuration
        self.config = QuantumGravityConfig(config_path)

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

# class QuantumGravity:
#     """Main interface for quantum gravity framework."""

#     def __init__(self, config: Dict[str, Any] = None):
#         """Initialize quantum gravity framework."""
#         # Load configuration
#         self.config = QuantumGravityConfig(config)

#         # Import core components
#         from .core.grid import AdaptiveGrid
#         from .core.state import QuantumState
#         from .core.operators import (
#             QuantumOperator, MetricOperator, MomentumOperator,
#             HamiltonianOperator, ConstraintOperator
#         )
#         from .core.evolution import TimeEvolution

#         # Import physics components
#         from .physics.conservation import ConservationLawTracker
#         from .physics.entanglement import EntanglementHandler
#         from .physics.observables import (
#             Observable, GeometricObservable, VolumeObservable,
#             AreaObservable, CurvatureObservable, EntanglementObservable
#         )

#         # Import numerical components
#         from .numerics.integrator import AdaptiveIntegrator
#         from .numerics.errors import ErrorTracker
#         from .numerics.parallel import ParallelManager

#         # Import utilities
#         from .utils.visualization import QuantumVisualization
#         from .utils.io import QuantumGravityIO

#         # Make components available
#         self.grid
#         self.state = None
#         self.evolution = None
#         self.parallel = None
#         self.io = QuantumGravityIO(self.config.config['io']['output_dir'])

#         logging.info("Quantum gravity framework initialized")

#     def setup_simulation(self) -> None:
#         """Setup simulation components."""
#         try:
#             # Create grid
#             self.grid = AdaptiveGrid(
#                 rho_0=1.0,
#                 eps_threshold=self.config.config['grid']['adaptive_threshold']
#             )

#             # Initialize state
#             self.state = QuantumState(
#                 self.grid,
#                 eps_cut=self.config.config['numerics']['eps_cut']
#             )

#             # Setup parallel environment
#             self.parallel = ParallelManager(
#                 self.grid,
#                 self.config.config['parallel']
#             )


__all__ = [
    'QuantumGravity',
    'QuantumGravityConfig',
    'CONSTANTS',
    'AdaptiveGrid',
    'QuantumState',
    'QuantumOperator',
    'TimeEvolution'
]
