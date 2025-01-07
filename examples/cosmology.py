#!/usr/bin/env python
# examples/cosmology.py

"""
Quantum Cosmology Simulation
===========================

This example demonstrates the simulation of quantum cosmological
scenarios, including universe expansion, inflation, and structure
formation with quantum corrections.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
# from quantum_gravity import QuantumGravity, CONSTANTS
from __init__ import QuantumGravity
from constants import CONSTANTS
from core.evolution import TimeEvolution
from utils.io import MeasurementResult


class CosmologySimulation:
    """Quantum cosmology simulation."""
    
    def __init__(self,
                 initial_scale: float,
                 hubble_parameter: float,
                 config_path: str = None):
        """Initialize cosmology simulation.
        
        Args:
            initial_scale: Initial scale factor
            hubble_parameter: Initial Hubble parameter in Planck units
            config_path: Optional path to configuration file
        """
        # Initialize framework
        self.qg = QuantumGravity(config_path)
        
        # Cosmological parameters
        self.initial_scale = initial_scale
        self.hubble_parameter = hubble_parameter
        self.lambda_cosm = CONSTANTS['lambda']  # Cosmological constant
        
        # Setup simulation
        self._setup_grid()
        self._setup_initial_state()
        self._setup_observables()
        
        # Results storage
        self.time_points = []
        self.scale_factor_history = []
        self.energy_density_history = []
        self.quantum_corrections_history = []
        self.perturbation_spectrum_history = []
        
    def _setup_grid(self) -> None:
        """Setup grid for cosmological simulation."""
        # Configure grid for large-scale structure
        grid_config = self.qg.config.config['grid']
        grid_config['refinement_factor'] = 2.0
        
        # Create periodic spatial grid
        L = 100.0  # Box size in Planck lengths
        N = grid_config['points_max']
        n = int(np.cbrt(N))  # Points per dimension
        
        x = np.linspace(-L/2, L/2, n)
        y = np.linspace(-L/2, L/2, n)
        z = np.linspace(-L/2, L/2, n)
        
        X, Y, Z = np.meshgrid(x, y, z)
        points = np.column_stack([X.flatten(), Y.flatten(), Z.flatten()])
        
        self.qg.grid.set_points(points)
        self.box_size = L
    def _setup_initial_state(self) -> None:
        """Setup initial quantum state for cosmology."""
        state = self.qg.state
        
        # Initialize cosmological parameters
        state.scale_factor = self.initial_scale
        state.energy_density = 3 * self.hubble_parameter**2 / (8 * np.pi * CONSTANTS['G'])
        
        # Set up FLRW metric with quantum corrections
        n_points = len(self.qg.grid.points)
        state._metric_array = np.zeros((4, 4, n_points))
        
        # Set metric components
        state._metric_array[0, 0, :] = -1  # Proper time components
        quantum_factor = 1 + (CONSTANTS['l_p']/state.scale_factor)**2
        
        for i in range(1, 4):
            state._metric_array[i, i, :] = state.scale_factor**2 * quantum_factor
            
        # Add initial perturbations
        self._add_quantum_fluctuations(state)
        
    def _add_quantum_fluctuations(self, state: 'QuantumState') -> None:
        """Add quantum fluctuations to initial state."""
        # Generate spectrum of fluctuations
        k_max = 2 * np.pi * np.cbrt(len(self.qg.grid.points)) / self.box_size
        
        for i, point in enumerate(self.qg.grid.points):
            # Compute quantum fluctuations in metric
            delta_g = self._compute_quantum_fluctuations(point, k_max)
            
            # Add fluctuations to metric components
            for mu in range(4):
                for nu in range(mu, 4):
                    current = state.get_metric_component((mu, nu), i)
                    state.set_metric_component((mu, nu), i, current + delta_g)
                    
    def _compute_quantum_fluctuations(self,
                                    point: np.ndarray,
                                    k_max: float) -> float:
        """Compute quantum fluctuations at a point."""
        # Simple model of quantum fluctuations
        amplitude = np.sqrt(CONSTANTS['hbar']/(2 * k_max))
        phase = np.random.uniform(0, 2*np.pi)
        
        return amplitude * np.cos(k_max * np.linalg.norm(point) + phase)
        
    def _setup_observables(self) -> None:
        """Setup observables for cosmological measurements."""
        # Scale factor observable
        self.scale_obs = self.qg.physics.ScaleFactorObservable(
            self.qg.grid
        )
        
        # Energy density observable
        self.density_obs = self.qg.physics.EnergyDensityObservable(
            self.qg.grid
        )
        
        # Quantum corrections observable
        self.quantum_obs = self.qg.physics.QuantumCorrectionsObservable(
            self.qg.grid
        )
        
        # Perturbation spectrum observable
        self.spectrum_obs = self.qg.physics.PerturbationSpectrumObservable(
            self.qg.grid
        )
    def _check_quantum_bounce(self, state: 'QuantumState') -> bool:
        """Detect quantum bounce conditions."""
        # Planck density threshold with proper scaling
        rho_planck = CONSTANTS['c']**5 / (CONSTANTS['hbar'] * CONSTANTS['G']**2)
    
        # Enhanced quantum correction terms
        quantum_factor = (CONSTANTS['l_p'] / state.scale_factor)**2
        bounce_threshold = rho_planck * quantum_factor
    
        # Add hysteresis to prevent rapid oscillations
        if not hasattr(self, '_last_bounce_time'):
            self._last_bounce_time = -float('inf')
        
        # Minimum time between bounces (in Planck times)
        bounce_cooldown = 1.0
    
        # Check bounce conditions with proper timing
        bounce_condition = (state.energy_density >= bounce_threshold and 
                       state.time - self._last_bounce_time > bounce_cooldown)
    
        if bounce_condition:
            self._last_bounce_time = state.time
        
        return bounce_condition

    def _handle_bounce(self, state):
        """Handle quantum bounce transition."""
        # Calculate critical density
        rho_crit = 0.41 * CONSTANTS['rho_planck']
        
        # Implement bounce dynamics when ρ > ρ_crit
        if state.energy_density >= rho_crit:
            # Reverse contraction to expansion
            self.hubble_parameter = abs(self.hubble_parameter)
            # Add quantum corrections
            quantum_factor = 1 - state.energy_density/rho_crit
            state.scale_factor *= quantum_factor
    def run_simulation(self, t_final: float, dt_save: float = None) -> None:
        """Run cosmological evolution with quantum bounce detection."""
        dt = 0.01
        t = 0.0
    
        # Add initialization logging
        logging.info(f"Starting simulation with initial scale factor: {self.qg.state.scale_factor}")
    
        step_count = 0
        while t < t_final:
            # Log progress every 100 steps
            if step_count % 100 == 0:
                logging.info(f"Step {step_count}: t={t:.2f}, a={self.qg.state.scale_factor:.6e}")
        
            # Check for bounce conditions
            if self._check_quantum_bounce(self.qg.state):
                self._handle_bounce(self.qg.state)
                logging.info(f"Quantum bounce detected at t={t:.2f}, a={self.qg.state.scale_factor:.6e}")
        
            # Evolution with quantum corrections
            old_scale = self.qg.state.scale_factor
            self.qg.state.scale_factor *= (1 + self.hubble_parameter * dt)
            self.qg.state.energy_density *= (1 - 3 * self.hubble_parameter * 
                                        (1 + self.qg.state.equation_of_state) * dt)
            # Verify update occurred
            if abs(old_scale - self.qg.state.scale_factor) < 1e-10:
                logging.warning(f"Scale factor not updating at t={t}")
        
            # Calculate quantum corrections
            quantum_factor = 1 + (CONSTANTS['l_p']/self.qg.state.scale_factor)**2
        
            # Update metric with quantum corrections
            for i in range(len(self.qg.grid.points)):
                for mu in range(1, 4):
                    current = self.qg.state.get_metric_component((mu, mu), i)
                    self.qg.state.set_metric_component((mu, mu), i, 
                        current * quantum_factor)
        
            # Record measurements periodically
            if dt_save is None or t % dt_save < dt:
                self._record_measurements(t)
        
            t += dt
            step_count += 1
    
        logging.info(f"Simulation completed: {step_count} steps")
    def _record_measurements(self, t: float) -> None:
        """Record measurements at current time."""
        # Measure observables
        scale = self.scale_obs.measure(self.qg.state)
        density = self.density_obs.measure(self.qg.state)
        quantum = self.quantum_obs.measure(self.qg.state)
        spectrum = self.spectrum_obs.measure(self.qg.state)
        
        # Store results
        self.time_points.append(t)
        self.scale_factor_history.append(scale.value)
        self.energy_density_history.append(density.value)
        self.quantum_corrections_history.append(quantum.value)
        self.perturbation_spectrum_history.append(spectrum.value)
        
    def plot_results(self, save_path: str = None) -> None:
        """Plot simulation results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Scale factor evolution
        ax1.plot(self.time_points, self.scale_factor_history)
        ax1.set_xlabel('Time [t_P]')
        ax1.set_ylabel('Scale Factor [l_P]')
        ax1.set_title('Universe Scale Factor Evolution')
        ax1.grid(True)
        
        # Energy density evolution
        ax2.plot(self.time_points, self.energy_density_history)
        ax2.set_xlabel('Time [t_P]')
        ax2.set_ylabel('Energy Density [ρ_P]')
        ax2.set_title('Energy Density Evolution')
        ax2.grid(True)
        
        # Quantum corrections
        ax3.plot(self.time_points, self.quantum_corrections_history)
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('Quantum Correction')
        ax3.set_title('Quantum Corrections Magnitude')
        ax3.grid(True)
        # Latest perturbation spectrum
        if self.perturbation_spectrum_history:
            k, Pk = self.perturbation_spectrum_history[-1]
            # Average over spatial components to get scalar power spectrum
            Pk_scalar = np.mean(np.mean(Pk, axis=0), axis=0)
            ax4.loglog(k, Pk_scalar)
            ax4.set_xlabel('Wavenumber k [1/l_P]')
            ax4.set_ylabel('Power Spectrum P(k)')
            ax4.set_title('Matter Power Spectrum')
            ax4.grid(True)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
    def compute_derived_quantities(self) -> Dict[str, np.ndarray]:
        """Compute derived cosmological quantities."""
        # Hubble parameter
        H = np.gradient(self.scale_factor_history, self.time_points)
        H = H / np.maximum(self.scale_factor_history, CONSTANTS['l_p'])
    
        # Deceleration parameter with regularization
        a = np.array(self.scale_factor_history)
        a_dot = np.gradient(a, self.time_points)
        a_dot = np.maximum(a_dot, CONSTANTS['l_p'])
        q = -a * np.gradient(a_dot, self.time_points) / a_dot**2
            
        return {
            'hubble_parameter': H,
            'deceleration_parameter': q
        }
        
def main():
    """Run cosmology simulation example."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directories with correct path
    output_dir = Path("results/cosmology")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial cosmological parameters
    initial_scale = 1000.0  # in Planck lengths
    hubble_parameter = 0.1  # in Planck units
    
    # Create and run simulation
    sim = CosmologySimulation(initial_scale, hubble_parameter)
    
    # Run until significant expansion
    t_final = 10000.0  # in Planck times
    sim.run_simulation(t_final)
    
    # Plot and save results
    sim.plot_results(str(output_dir / "evolution.png"))
    
    # Compute and save derived quantities
    derived = sim.compute_derived_quantities()
    
    # Create proper MeasurementResult objects
    measurements = [
        MeasurementResult(
            value=sim.time_points,
            uncertainty=None,
            metadata={'type': 'time'}
        ),
        MeasurementResult(
            value=sim.scale_factor_history,
            uncertainty=None,
            metadata={'type': 'scale_factor'}
        ),
        MeasurementResult(
            value=sim.energy_density_history,
            uncertainty=None,
            metadata={'type': 'energy_density'}
        ),
        MeasurementResult(
            value=sim.quantum_corrections_history,
            uncertainty=None,
            metadata={'type': 'quantum_corrections'}
        ),
        MeasurementResult(
            value=derived['hubble_parameter'].tolist(),
            uncertainty=None,
            metadata={'type': 'hubble_parameter'}
        ),
        MeasurementResult(
            value=derived['deceleration_parameter'].tolist(),
            uncertainty=None,
            metadata={'type': 'deceleration_parameter'}
        )
    ]

    sim.qg.io.save_measurements(measurements, str(output_dir / "measurements"))
    
if __name__ == "__main__":
    main()

