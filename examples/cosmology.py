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
from matplotlib.gridspec import GridSpec
# from quantum_gravity import QuantumGravity, CONSTANTS
from __init__ import QuantumGravity, configure_logging
from constants import CONSTANTS
from core.evolution import TimeEvolution
from utils.io import MeasurementResult
from numerics.errors import ErrorTracker
from physics.conservation import ConservationLawTracker
from physics.verification import CosmologicalVerification
from core.state import QuantumState, CosmologicalState
from core.grid import AdaptiveGrid, LeechLattice
from physics.observables import CosmicEvolutionObservable

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

        # Add Leech lattice structure
        self.leech_lattice = LeechLattice(points=100000)
        
        # Add vacuum energy from Leech lattice
        self.vacuum_energy = self.leech_lattice.compute_vacuum_energy()


        self.qg.state = CosmologicalState(
            grid=self.qg.grid,
            initial_scale=initial_scale,
            hubble_parameter=hubble_parameter
        )

        # Add trackers
        base_tolerances = {'truncation': 1e-10, 'constraint': 1e-8, 'conservation': 1e-8}
        self.error_tracker = ErrorTracker(self.qg.grid, base_tolerances)
        self.conservation_tracker = ConservationLawTracker(self.qg.grid)

        # Setup simulation
        self._setup_grid()
        self._setup_initial_state()
        self._setup_observables()

        # Add tracking lists for verification metrics
        self.verification_results = []
        self.hubble_squared_lhs = []
        self.hubble_squared_rhs = []

        # Results storage
        self.time_points = []
        self.scale_factor_history = []
        self.energy_density_history = []
        self.quantum_corrections_history = []
        self.perturbation_spectrum_history = []
        self.hubble_history = []
        self.eos_history = []
        self.acceleration_history = []
        self.entropy_history = []



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
        #state.energy_density = 3 * self.hubble_parameter**2 / (8 * np.pi * CONSTANTS['G'])
        base_energy_density = 3 * self.hubble_parameter**2 / (8 * np.pi * CONSTANTS['G'])
        state.energy_density = base_energy_density + self.vacuum_energy

        # Set up FLRW metric with quantum corrections
        n_points = len(self.qg.grid.points)
        state._metric_array = np.zeros((4, 4, n_points))
        
        # Set metric components
        state._metric_array[0, 0, :] = -1  # Proper time components
        quantum_factor = 1 + (CONSTANTS['l_p']/state.scale_factor)**2
        leech_factor = self.vacuum_energy * (CONSTANTS['l_p']/state.scale_factor)**24

        for i in range(1, 4):
            state._metric_array[i, i, :] = state.scale_factor**2 * quantum_factor
        #for i in range(1, 4):
        #    state._metric_array[i, i, :] = state.scale_factor**2 * (quantum_factor + leech_factor)    
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

        # Add cosmic evolution observable
        self.cosmic_obs = CosmicEvolutionObservable()
        
        # Track evolution history
        self.hubble_history = []
        self.eos_history = []
        self.acceleration_history = []
        self.entropy_history = []

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
        """Handle quantum bounce transition with proper dynamics."""
        rho_crit = 0.41 * CONSTANTS['rho_planck']
        
        if state.energy_density >= rho_crit:
            # Reverse contraction to expansion
            self.hubble_parameter = abs(self.hubble_parameter)
            # Update state parameters
            quantum_factor = 1 - state.energy_density/rho_crit
            state.scale_factor *= quantum_factor
            # Update hubble parameter in state
            state.hubble_parameter = self.hubble_parameter

    def run_simulation(self, t_final: float, dt_save: float = None) -> None:
        """Run simulation with synchronized data collection and full logging."""
        # Initialize simulation parameters
        dt = 0.01
        t = 0.0
        step_count = 0
        
        # Initialize all tracking arrays
        self.time_points = []
        self.verification_results = []
        self.hubble_squared_lhs = []
        self.hubble_squared_rhs = []
        
        logging.info(f"Starting simulation with initial scale factor: {self.qg.state.scale_factor}")
        self.verifier = CosmologicalVerification(self)
        
        # Record initial state
        self._record_measurements(t)
        initial_metrics = self.verifier.verify_geometric_entanglement(self.qg.state)
        initial_friedmann = self.verifier.verify_friedmann_equations(self.qg.state)
        
        self.verification_results.append({
            'time': t,
            'scale_factor': self.qg.state.scale_factor,
            'lhs': initial_metrics['lhs'],
            'rhs': initial_metrics['rhs']
        })
        self.hubble_squared_lhs.append(initial_friedmann['lhs'])
        self.hubble_squared_rhs.append(initial_friedmann['rhs'])
        
        # Base evolution configuration
        evolution_config = {
            'dt': dt,
            'error_tolerance': 1e-6
        }
        
        while t < t_final:
            # Evolution step
            evolution = TimeEvolution(
                grid=self.qg.grid,
                config=evolution_config,
                error_tracker=self.error_tracker,
                conservation_tracker=self.conservation_tracker,
                state=self.qg.state
            )
            evolution._evolve_state(dt)
            
            # Update time and collect metrics
            t += dt
            step_count += 1
            
            metrics = self.verifier.verify_geometric_entanglement(self.qg.state)
            friedmann = self.verifier.verify_friedmann_equations(self.qg.state)
            
            # Store verification results
            self.verification_results.append({
                'time': t,
                'scale_factor': self.qg.state.scale_factor,
                'lhs': metrics['lhs'],
                'rhs': metrics['rhs']
            })
            self.hubble_squared_lhs.append(friedmann['lhs'])
            self.hubble_squared_rhs.append(friedmann['rhs'])
            
            # Detailed logging every 100 steps
            if step_count % 100 == 0:
                # Log inflation dynamics
                inflation_metrics = self.verifier.verify_inflation_dynamics(self.qg.state)
                logging.info(
                    f"\nInflation Dynamics at t={t:.2f}:"
                    f"\nSlow-roll parameter ε = {inflation_metrics['slow_roll']:.6e}"
                    f"\nPerturbation spectrum = {inflation_metrics['spectrum']:.6e}"
                )
                
                # Log cosmic evolution
                cosmic = self.cosmic_obs.measure(self.qg.state)
                logging.info(
                    f"\nCosmic Evolution at t={t:.2f}:"
                    f"\nHubble Parameter H = {cosmic.value['hubble']:.6e}"
                    f"\nEquation of State w = {cosmic.value['eos']:.6e}"
                    f"\nAcceleration q = {cosmic.value['acceleration']:.6e}"
                    f"\nCosmic Entropy S = {cosmic.value['entropy']:.6e}"
                )
                
                # Log geometric entanglement
                logging.info(
                    f"\nGeometric-Entanglement Formula at t={t:.2f}:"
                    f"\nLHS = {metrics['lhs']:.6e}"
                    f"\nRHS = {metrics['rhs']:.6e}"
                    f"\nRelative Error = {metrics['relative_error']:.6e}"
                )
                
                # Log Friedmann equations
                logging.info(
                    f"\nQuantum-Corrected Friedmann Equations at t={t:.2f}:"
                    f"\nH² (LHS) = {friedmann['lhs']:.6e}"
                    f"\nH² (RHS) = {friedmann['rhs']:.6e}"
                    f"\nQuantum Correction = {friedmann['quantum_correction']:.6e}"
                )
                
                # Log observables
                scale = self.scale_obs.measure(self.qg.state)
                density = self.density_obs.measure(self.qg.state)
                quantum = self.quantum_obs.measure(self.qg.state)
                spectrum = self.spectrum_obs.measure(self.qg.state)
                
                # Calculate and log power spectrum metrics
                k, Pk = spectrum.value
                Pk_scalar = np.mean(np.mean(Pk, axis=0), axis=0)
                logging.info(
                    f"Time t={t:.2f}, a={self.qg.state.scale_factor:.6e}: "
                    f"Energy Density={density.value:.6e}, "
                    f"Quantum Corrections={quantum.value:.6e}"
                    f"\nScale Factor = {scale.value:.6e}"
                    f"\nPower Spectrum k_max = {np.max(k):.6e}"
                    f"\nPower Spectrum P(k) mean = {np.mean(Pk_scalar):.6e}"
                )
                
                logging.info(f"Simulation progress: {t/t_final*100:.1f}% (t={t:.2f}/{t_final})")
            
            # Store quantum state on grid
            self.qg.grid.quantum_state = self.qg.state
            
            # Check for bounce conditions
            if self._check_quantum_bounce(self.qg.state):
                self._handle_bounce(self.qg.state)
                logging.info(f"Quantum bounce detected at t={t:.2f}, a={self.qg.state.scale_factor:.6e}")
            
            # Update metric with quantum corrections
            quantum_factor = 1 + (CONSTANTS['l_p']/self.qg.state.scale_factor)**2
            for i in range(len(self.qg.grid.points)):
                for mu in range(1, 4):
                    current = self.qg.state.get_metric_component((mu, mu), i)
                    self.qg.state.set_metric_component((mu, mu), i, current * quantum_factor)
            
            # Record measurements
            self._record_measurements(t)
        
        logging.info(f"Simulation completed: {step_count} steps")


    def _record_measurements(self, t: float) -> None:
        """Record measurements at current time."""
        # Always measure when this method is called
        scale = self.scale_obs.measure(self.qg.state)
        density = self.density_obs.measure(self.qg.state)
        quantum = self.quantum_obs.measure(self.qg.state)
        spectrum = self.spectrum_obs.measure(self.qg.state)
        cosmic = self.cosmic_obs.measure(self.qg.state)
        
        # Initialize lists if they don't exist
        if not hasattr(self, 'time_points'):
            self.time_points = []
        if not hasattr(self, 'scale_factor_history'):
            self.scale_factor_history = []
        if not hasattr(self, 'energy_density_history'):
            self.energy_density_history = []
        if not hasattr(self, 'quantum_corrections_history'):
            self.quantum_corrections_history = []
        if not hasattr(self, 'perturbation_spectrum_history'):
            self.perturbation_spectrum_history = []
        if not hasattr(self, 'hubble_history'):
            self.hubble_history = []
        if not hasattr(self, 'eos_history'):
            self.eos_history = []
        if not hasattr(self, 'acceleration_history'):
            self.acceleration_history = []
        if not hasattr(self, 'entropy_history'):
            self.entropy_history = []
        
        # Store all results
        if t not in self.time_points:  # Only record if we haven't recorded this time point
            self.time_points.append(t)
            self.scale_factor_history.append(scale.value)
            self.energy_density_history.append(density.value)
            self.quantum_corrections_history.append(quantum.value)
            self.perturbation_spectrum_history.append(spectrum.value)
            self.hubble_history.append(cosmic.value['hubble'])
            self.eos_history.append(cosmic.value['eos'])
            self.acceleration_history.append(cosmic.value['acceleration'])
            self.entropy_history.append(cosmic.value['entropy'])
            
            # Log measurements periodically
            if len(self.time_points) % 100 == 0:
                logging.info(
                    f"Time t={t:.2f}, a={scale.value:.6e}: "
                    f"Energy Density={density.value:.6e}, "
                    f"Quantum Corrections={quantum.value:.6e}"
                    f"\nScale Factor = {scale.value:.6e}"
                    f"\nPower Spectrum k_max = {np.max(spectrum.value[0]):.6e}"
                    f"\nPower Spectrum P(k) mean = {np.mean(spectrum.value[1]):.6e}"
                )

    def plot_results(self, save_path: str = None) -> None:
        """Plot comprehensive cosmological evolution with synchronized data."""
        fig = plt.figure(figsize=(15, 24))
        gs = GridSpec(5, 2, figure=fig)
        
        # Ensure time_array matches the actual data length
        time_array = np.array(self.time_points)
        
        # Verification data arrays
        verification_times = np.array([v['time'] for v in self.verification_results])
        lhs_values = np.array([v['lhs'] for v in self.verification_results])
        rhs_values = np.array([v['rhs'] for v in self.verification_results])
        
        # Ensure all arrays have the same length
        min_length = min(len(time_array), len(verification_times))
        time_array = time_array[:min_length]
        
        # Basic evolution plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(time_array, self.scale_factor_history[:min_length])
        ax1.set_yscale('log')
        ax1.set_xlabel('Time [t_P]')
        ax1.set_ylabel('Scale Factor [l_P]')
        ax1.set_title('Universe Scale Factor Evolution')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(time_array, self.energy_density_history[:min_length])
        ax2.set_yscale('log')
        ax2.set_xlabel('Time [t_P]')
        ax2.set_ylabel('Energy Density [ρ_P]')
        ax2.set_title('Energy Density Evolution')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(time_array, self.quantum_corrections_history[:min_length])
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('Quantum Correction')
        ax3.set_title('Quantum Corrections Magnitude')
        ax3.grid(True)
        
        # Power spectrum evolution
        ax4 = fig.add_subplot(gs[1, 1])
        if self.perturbation_spectrum_history:
            times = [0, min_length//2, -1]
            for t_idx in times:
                if t_idx < len(self.perturbation_spectrum_history):
                    k, Pk = self.perturbation_spectrum_history[t_idx]
                    Pk_scalar = np.mean(np.mean(Pk, axis=0), axis=0)
                    ax4.loglog(k, Pk_scalar, label=f't={time_array[t_idx]:.1f}')
        ax4.set_xlabel('Wavenumber k [1/l_P]')
        ax4.set_ylabel('Power Spectrum P(k)')
        ax4.set_title('Matter Power Spectrum Evolution')
        ax4.legend()
        ax4.grid(True)
        
        # Cosmic evolution plots with synchronized lengths
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(time_array, self.hubble_history[:min_length])
        ax5.set_title('Hubble Parameter Evolution')
        ax5.set_xlabel('Time [t_P]')
        ax5.set_ylabel('H [1/t_P]')
        ax5.grid(True)
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(time_array, self.eos_history[:min_length])
        ax6.set_title('Equation of State Evolution')
        ax6.set_xlabel('Time [t_P]')
        ax6.set_ylabel('w(t)')
        ax6.grid(True)
        
        ax7 = fig.add_subplot(gs[3, 0])
        ax7.plot(time_array, self.acceleration_history[:min_length])
        ax7.set_title('Cosmic Acceleration')
        ax7.set_xlabel('Time [t_P]')
        ax7.set_ylabel('q(t)')
        ax7.grid(True)
        
        ax8 = fig.add_subplot(gs[3, 1])
        ax8.plot(time_array, self.entropy_history[:min_length])
        ax8.set_title('Cosmic Entropy Evolution')
        ax8.set_xlabel('Time [t_P]')
        ax8.set_ylabel('S [k_B]')
        ax8.grid(True)
        
        # Verification plots with synchronized data
        ax9 = fig.add_subplot(gs[4, 0])
        ax9.plot(verification_times[:min_length], lhs_values[:min_length], 
                label='LHS', color='blue')
        ax9.plot(verification_times[:min_length], rhs_values[:min_length], 
                label='RHS', color='red')
        ax9.set_yscale('log')
        ax9.set_xlabel('Time [t_P]')
        ax9.set_ylabel('Geometric-Entanglement Terms')
        ax9.set_title('Geometric-Entanglement Evolution')
        ax9.legend()
        ax9.grid(True)
        
        # Friedmann equations error plot
        ax10 = fig.add_subplot(gs[4, 1])
        hubble_lhs = np.array(self.hubble_squared_lhs[:min_length])
        hubble_rhs = np.array(self.hubble_squared_rhs[:min_length])
        friedmann_error = np.abs(hubble_lhs - hubble_rhs) / hubble_lhs
        ax10.plot(time_array, friedmann_error)
        ax10.set_yscale('log')
        ax10.set_xlabel('Time [t_P]')
        ax10.set_ylabel('Relative Error')
        ax10.set_title('Friedmann Equations Error Evolution')
        ax10.grid(True)
        
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
    #logging.basicConfig(level=logging.INFO)
    configure_logging(simulation_type='cosmology')
    
    # Create output directories with correct path
    output_dir = Path("results/cosmology")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initial cosmological parameters
    initial_scale = 1000.0  # in Planck lengths
    hubble_parameter = 0.1  # in Planck units
    
    # Create and run simulation
    sim = CosmologySimulation(initial_scale, hubble_parameter)
    
    # Run until significant expansion
    t_final = 20.0  # in Planck times
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

