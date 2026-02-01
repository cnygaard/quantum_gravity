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
from core.grid import AdaptiveGrid
from physics.quantum_geometry import QuantumGeometry
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

        # v7: Use QuantumGeometry instead of LeechLattice
        self.quantum_geometry = QuantumGeometry()
        gamma_0 = CONSTANTS['gamma_0']  # Immirzi parameter

        # v7 vacuum energy using cosmic factor
        cosmic_factor = np.pi / gamma_0  # ≈ 11.46
        self.vacuum_energy = (CONSTANTS['l_p'] / initial_scale)**4 * cosmic_factor

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

        # v7: Enhanced vacuum energy with Immirzi-based scaling
        gamma_0 = CONSTANTS['gamma_0']
        cosmic_factor = np.pi / gamma_0
        vacuum_energy = (CONSTANTS['l_p'] / self.initial_scale)**4 * cosmic_factor

        state.energy_density = base_energy_density + self.vacuum_energy

        # Set up FLRW metric with quantum corrections
        n_points = len(self.qg.grid.points)
        state._metric_array = np.zeros((4, 4, n_points))
        
        # Set metric components
        state._metric_array[0, 0, :] = -1  # Proper time components
        # v7: Quantum factor using Immirzi parameter
        gamma_0 = CONSTANTS['gamma_0']
        quantum_factor = 1 + gamma_0 * (CONSTANTS['l_p']/state.scale_factor)**2

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

    # def _check_quantum_bounce(self, state: 'QuantumState') -> bool:
    #     """Detect quantum bounce conditions."""
    #     # Planck density threshold with proper scaling
    #     rho_planck = CONSTANTS['c']**5 / (CONSTANTS['hbar'] * CONSTANTS['G']**2)
    
    #     # Enhanced quantum correction terms
    #     quantum_factor = (CONSTANTS['l_p'] / state.scale_factor)**2
    #     bounce_threshold = rho_planck * quantum_factor
    
    #     # Add hysteresis to prevent rapid oscillations
    #     if not hasattr(self, '_last_bounce_time'):
    #         self._last_bounce_time = -float('inf')
        
    #     # Minimum time between bounces (in Planck times)
    #     bounce_cooldown = 10.0
    
    #     # Check bounce conditions with proper timing
    #     bounce_condition = (state.energy_density >= bounce_threshold and 
    #                    state.time - self._last_bounce_time > bounce_cooldown)
    
    #     if bounce_condition:
    #         self._last_bounce_time = state.time
        
    #     return bounce_condition

    def _check_quantum_bounce(self, state: 'QuantumState') -> bool:
        """Check for quantum bounce following paper criteria."""
        rho_crit = 0.41 * CONSTANTS['rho_planck']
        rho = state.energy_density
        
        # Smooth transition function near bounce
        def transition(x):
            x0 = 0.95 * rho_crit  # Transition point
            width = 0.1 * rho_crit  # Transition width
            return 0.5 * (1 + np.tanh((x - x0)/width))
        
        bounce_factor = transition(rho)
        return bounce_factor > 0.5

    # def _handle_bounce(self, state):
    #     """Handle quantum bounce transition with proper dynamics."""
    #     rho_crit = 0.41 * CONSTANTS['rho_planck']
        
    #     if state.energy_density >= rho_crit:
    #         # Reverse contraction to expansion
    #         self.hubble_parameter = abs(self.hubble_parameter)
    #         # Update state parameters
    #         quantum_factor = 1 - state.energy_density/rho_crit
    #         state.scale_factor *= quantum_factor
    #         # Update hubble parameter in state
    #         state.hubble_parameter = self.hubble_parameter

    def _handle_bounce(self, state):
        """Handle quantum bounce with smooth transition following Ashtekar et al."""
        rho_crit = 0.41 * CONSTANTS['rho_planck']
        
        # Smooth transition function
        def transition(x):
            return 0.5 * (1 + np.tanh((rho_crit - x)/(0.1 * rho_crit)))
        
        if state.energy_density >= 0.1 * rho_crit:
            # Smooth reversal of Hubble parameter
            quantum_factor = transition(state.energy_density)
            self.hubble_parameter = (
                quantum_factor * abs(self.hubble_parameter) - 
                (1 - quantum_factor) * self.hubble_parameter
            )
            
            # Update state parameters smoothly
            state.scale_factor *= (1 - state.energy_density/rho_crit * quantum_factor)
            state.hubble_parameter = self.hubble_parameter

    def _validate_hubble(self, H: float, state: 'QuantumState') -> Dict[str, float]:
        """Validate Hubble parameter with detailed diagnostics"""
        # 1. Unit consistency check
        H_planck = H * CONSTANTS['l_p']/CONSTANTS['t_p']  # Convert to Planck units
        
        # 2. Calculate derivative - handle both single points and history
        a = state.scale_factor
        if len(self.time_points) > 1:
            # Use history if available
            da_dt = np.gradient(self.scale_factor_history, self.time_points)[-1]
            H_direct = da_dt/a
        else:
            # For single timestep, use current H
            H_direct = H
        
        # 3. Physical bounds validation
        H_max = np.sqrt(0.41 * CONSTANTS['rho_planck']/3)  # Maximum allowed H
        H_classical = self.hubble_parameter * (self.initial_scale/a)**(3/2)
        
        # 4. Classical vs quantum monitoring
        quantum_correction = 1 + (CONSTANTS['l_p']/a)**2
        H_quantum = H_classical * quantum_correction
        
        return {
            'H_planck': H_planck,
            'H_direct': H_direct,
            'H_max': H_max, 
            'H_classical': H_classical,
            'H_quantum': H_quantum,
            'quantum_correction': quantum_correction
        }


    def run_simulation(self, t_final: float, dt_save: float = None) -> None:
        """Run simulation with logarithmic time stepping for cosmological scales.

        Cosmological simulations span vast timescales (potentially 60+ orders of
        magnitude from Planck time to present). Logarithmic stepping efficiently
        samples this range while maintaining numerical accuracy.
        """
        # Initialize all tracking arrays
        self.time_points = []
        self.verification_results = []
        self.hubble_squared_lhs = []
        self.hubble_squared_rhs = []

        # Calculate characteristic cosmological timescales
        t_hubble = 1.0 / max(self.hubble_parameter, 1e-10)  # Hubble time

        # For inflation, use e-folding time; otherwise use Hubble time
        n_efolds = 60  # Typical number of e-folds for inflation
        t_inflation = n_efolds * t_hubble

        # Determine appropriate simulation timescale
        t_max = min(t_final, t_inflation * 0.99)

        # Logarithmic time stepping parameters
        n_steps = 1000
        dt_min = 0.001 * t_hubble  # Start with small fraction of Hubble time
        dt_min = max(dt_min, 1e-6)  # Ensure minimum timestep

        # Generate log-spaced time points
        if t_max > dt_min:
            log_times = np.logspace(np.log10(dt_min), np.log10(t_max), n_steps)
        else:
            # Fallback to linear if timescales are too close
            log_times = np.linspace(dt_min, t_max, n_steps)
        log_times = np.insert(log_times, 0, 0.0)  # Start from t=0

        logging.info(f"Starting Cosmology Simulation:")
        logging.info(f"Initial scale factor: {self.qg.state.scale_factor:.3e}")
        logging.info(f"Initial Hubble parameter: {self.hubble_parameter:.3e}")
        logging.info(f"Hubble time: {t_hubble:.3e} t_P")
        logging.info(f"\nUsing logarithmic time stepping:")
        logging.info(f"  Simulation covers: 0 to {t_max:.3e} t_P")
        logging.info(f"  Number of steps: {n_steps}")
        logging.info(f"  dt_min: {dt_min:.3e}, dt_max: {log_times[-1] - log_times[-2]:.3e}")

        self.verifier = CosmologicalVerification(self)

        # Record initial state at t=0
        self._record_measurements(0.0)
        initial_metrics = self.verifier.verify_geometric_entanglement(self.qg.state)
        initial_friedmann = self.verifier.verify_friedmann_equations(self.qg.state)

        self.hubble_squared_lhs.append(initial_friedmann['lhs'])
        self.hubble_squared_rhs.append(initial_friedmann['rhs'])

        # Base evolution configuration (dt will be updated each step)
        evolution_config = {
            'dt': dt_min,
            'error_tolerance': 1e-6
        }

        # Get reference to quantum state
        state = self.qg.state

        # Main evolution loop with logarithmic time stepping
        for step_idx in range(1, len(log_times)):
            t = log_times[step_idx]
            t_prev = log_times[step_idx - 1]
            dt = t - t_prev  # Variable timestep

            # Update evolution config with current dt
            evolution_config['dt'] = dt
            base_energy_density = 3 * self.hubble_parameter**2 / (8 * np.pi * CONSTANTS['G'])
            # v7: Update vacuum energy using cosmic factor
            gamma_0 = CONSTANTS['gamma_0']
            cosmic_factor = np.pi / gamma_0
            self.vacuum_energy = (CONSTANTS['l_p'] / self.qg.state.scale_factor)**4 * cosmic_factor

            # Update state
            state.energy_density = base_energy_density + self.vacuum_energy

            # Evolution step with variable dt
            evolution = TimeEvolution(
                grid=self.qg.grid,
                config=evolution_config,
                error_tracker=self.error_tracker,
                conservation_tracker=self.conservation_tracker,
                state=self.qg.state
            )
            evolution._evolve_state(dt)

            # Collect metrics at current time
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
            H = state.hubble_parameter
            diagnostics = self._validate_hubble(H, state)

            # Detailed logging at log-spaced intervals (every 10% of steps)
            if step_idx % (n_steps // 10) == 0:
                # Log inflation dynamics
                inflation_metrics = self.verifier.verify_inflation_dynamics(self.qg.state)
                logging.info(
                    f"\nInflation Dynamics at t={t:.2f}:"
                    f"\nSlow-roll parameter ε = {inflation_metrics['slow_roll']:.6e}"
                    f"\nPerturbation spectrum = {inflation_metrics['spectrum']:.6e}"
                )

                logging.info(f"""
                Hubble Diagnostics:
                H (computed): {H:.6e}
                H (direct): {diagnostics['H_direct']:.6e}
                H (classical): {diagnostics['H_classical']:.6e}
                Quantum correction: {diagnostics['quantum_correction']:.6e}
                """)

                # Log cosmic evolution
                cosmic = self.cosmic_obs.measure(self.qg.state)
                logging.info(
                    f"\nCosmic Evolution at t={t:.2f}:"
                    f"\nHubble Parameter H = {cosmic.value['hubble']:.6e}"
                    f"\nEquation of State w = {cosmic.value['eos']:.6e}"
                    f"\nAcceleration q = {cosmic.value['acceleration']:.6e}"
                    f"\nCosmic Entropy S = {cosmic.value['entropy']:.6e}"
                )
                
                # Log v7 master equation verification
                logging.info(
                    f"\nv7 Master Equation: g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν) at t={t:.2f}:"
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
                
                logging.info(f"Simulation progress: {step_idx/n_steps*100:.1f}% (t={t:.3e}/{t_max:.3e})")

            # Store quantum state on grid
            self.qg.grid.quantum_state = self.qg.state

            # Check for bounce conditions
            if self._check_quantum_bounce(self.qg.state):
                self._handle_bounce(self.qg.state)
                logging.info(f"Quantum bounce detected at t={t:.3e}, a={self.qg.state.scale_factor:.6e}")

            # Update metric with v7 quantum corrections
            gamma_0 = CONSTANTS['gamma_0']
            quantum_factor = 1 + gamma_0 * (CONSTANTS['l_p']/self.qg.state.scale_factor)**2
            for i in range(len(self.qg.grid.points)):
                for mu in range(1, 4):
                    current = self.qg.state.get_metric_component((mu, mu), i)
                    self.qg.state.set_metric_component((mu, mu), i, current * quantum_factor)

            # Record measurements at log-spaced intervals
            if step_idx % (n_steps // 100) == 0 or step_idx == len(log_times) - 1:
                self._record_measurements(t)

        # Final verification summary
        lhs_history = [v['lhs'] for v in self.verification_results]
        rhs_history = [v['rhs'] for v in self.verification_results]
        if lhs_history and rhs_history:
            errors = [abs(l - r) / max(abs(l), abs(r), 1e-30) for l, r in zip(lhs_history, rhs_history)]
            logging.info(f"\nFinal v7 Equation Verification Summary:")
            logging.info(f"g_μν = ℓ_P²(G_μν^Fisher + γ₀E_μν)  [γ₀ = {CONSTANTS['gamma_0']}]")
            logging.info(f"Final LHS = {lhs_history[-1]:.6e}")
            logging.info(f"Final RHS = {rhs_history[-1]:.6e}")
            logging.info(f"Final Error = {errors[-1]:.6e}")
            logging.info(f"Overall Mean Error = {np.mean(errors):.6e}")
            logging.info(f"Overall Max Error = {np.max(errors):.6e}")

        logging.info(f"\nSimulation completed: {n_steps} steps")


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
        #ax9.plot(verification_times[:min_length], lhs_values[:min_length], 
        #        label='LHS', color='blue')
        #ax9.plot(verification_times[:min_length], rhs_values[:min_length], 
        #        label='RHS', color='red')
        ax9.plot(time_array, lhs_values[:min_length], 
                label='LHS', color='blue')
        ax9.plot(time_array, rhs_values[:min_length], 
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
        #friedmann_error = np.abs(hubble_lhs - hubble_rhs) / hubble_lhs
        # In plot_results method
        friedmann_error = np.abs(hubble_lhs - hubble_rhs) / np.maximum(hubble_lhs, 1e-30)

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


    # def compute_derived_quantities(self) -> Dict[str, np.ndarray]:
    #     """Compute derived cosmological quantities."""
    #     # Hubble parameter
    #     H = np.gradient(self.scale_factor_history, self.time_points)
    #     H = H / np.maximum(self.scale_factor_history, CONSTANTS['l_p'])
    
    #     # Deceleration parameter with regularization
    #     a = np.array(self.scale_factor_history)
    #     a_dot = np.gradient(a, self.time_points)
    #     a_dot = np.maximum(a_dot, CONSTANTS['l_p'])
    #     q = -a * np.gradient(a_dot, self.time_points) / a_dot**2
            
    #     return {
    #         'hubble_parameter': H,
    #         'deceleration_parameter': q
    #     }

    # def compute_derived_quantities(self) -> Dict[str, np.ndarray]:
    #     # Use Savitzky-Golay filter for smooth derivatives
    #     from scipy.signal import savgol_filter
        
    #     # Smooth the scale factor data first
    #     window = 31  # Must be odd
    #     poly_order = 4
    #     a_smooth = savgol_filter(self.scale_factor_history, window, poly_order)
        
    #     # Calculate smoothed derivatives
    #     dt = np.mean(np.diff(self.time_points))
    #     a_dot = savgol_filter(a_smooth, window, poly_order, deriv=1, delta=dt)
    #     a_ddot = savgol_filter(a_smooth, window, poly_order, deriv=2, delta=dt)
        
    #     # Compute Hubble parameter
    #     H = a_dot / np.maximum(a_smooth, CONSTANTS['l_p'])
        
    #     # Compute deceleration parameter
    #     q = -a_smooth * a_ddot / (a_dot**2)
        
    #     # Apply additional smoothing to remove any remaining artifacts
    #     H = savgol_filter(H, window, poly_order)
    #     q = savgol_filter(q, window, poly_order)
        
    #     return {
    #         'hubble_parameter': H,
    #         'deceleration_parameter': q
    #     }

    def compute_derived_quantities(self) -> Dict[str, np.ndarray]:
        from scipy.signal import savgol_filter
        
        # Adjust window size based on data length
        n_points = len(self.scale_factor_history)
        window = min(15, n_points if n_points % 2 != 0 else n_points - 1)
        poly_order = min(3, window - 1)
        
        # Smooth the scale factor data
        a_smooth = savgol_filter(self.scale_factor_history, window, poly_order)
        
        # Calculate derivatives with proper timestep
        dt = self.time_points[1] - self.time_points[0]  # Use actual time intervals
        a_dot = savgol_filter(a_smooth, window, poly_order, deriv=1, delta=dt)
        a_ddot = savgol_filter(a_smooth, window, poly_order, deriv=2, delta=dt)
        
        # Compute parameters with smoothed derivatives
        H = a_dot / np.maximum(a_smooth, CONSTANTS['l_p'])
        q = -a_smooth * a_ddot / (a_dot**2)
        
        return {
            'hubble_parameter': H,
            'deceleration_parameter': q
        }


def main():
    """Run cosmology simulation example."""
    # Setup logging
    configure_logging(simulation_type='cosmology')

    # Create output directories with correct path
    output_dir = Path("results/cosmology")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial cosmological parameters
    initial_scale = 1000.0  # in Planck lengths
    hubble_parameter = 0.1  # in Planck units (H ~ 0.1 t_P^-1)

    # Create simulation
    sim = CosmologySimulation(initial_scale, hubble_parameter)

    # Calculate appropriate simulation time
    # Hubble time t_H = 1/H defines the characteristic expansion timescale
    t_hubble = 1.0 / hubble_parameter  # = 10 t_P for H=0.1
    n_efolds = 10  # Number of e-foldings to simulate

    # Simulate for multiple Hubble times to see significant expansion
    # For inflation: t_final ~ N_efolds * t_H
    t_final = n_efolds * t_hubble

    logging.info(f"Cosmological Timescales:")
    logging.info(f"  Hubble time t_H = {t_hubble:.3e} t_P")
    logging.info(f"  Simulating {n_efolds} e-folds")
    logging.info(f"  Total time: {t_final:.3e} t_P")

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

