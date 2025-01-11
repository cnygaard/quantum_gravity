#!/usr/bin/env python
# examples/black_hole.py

"""
Quantum Black Hole Simulation
============================

This example demonstrates the simulation of a quantum black hole,
including horizon dynamics, Hawking radiation, and entropy evolution.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from __init__ import QuantumGravity  # We haven't created QuantumGravity class yet!

import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Or 'Qt5Agg' if you prefer Qt
import matplotlib.pyplot as plt
from constants import CONSTANTS
from core.state import QuantumState
from physics.verification import UnifiedTheoryVerification
from utils.io import QuantumGravityIO, MeasurementResult


#from core.constants import CONSTANTS


#from quantum_gravity import QuantumGravity, CONSTANTS

class BlackHoleSimulation:
    """Quantum black hole simulation."""
    def plot_results(self, save_path: str = None) -> None:
        """Plot simulation results including geometric-entanglement verification."""
        import matplotlib.pyplot as plt

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Mass evolution
        ax1.plot(self.time_points, self.mass_history)
        ax1.set_xlabel('Time [t_P]')
        ax1.set_ylabel('Mass [m_P]')
        ax1.set_title('Black Hole Mass Evolution')
        ax1.grid(True)
        
        # Entropy evolution
        ax2.plot(self.time_points, self.entropy_history)
        ax2.set_xlabel('Time [t_P]')
        ax2.set_ylabel('Entropy [k_B]')
        ax2.set_title('Black Hole Entropy Evolution')
        ax2.grid(True)
        
        # Geometric-Entanglement Equation Verification
        if hasattr(self, 'verification_results'):
            times = [result['time'] for result in self.verification_results]
            geo_errors = [result['geometric_entanglement_error'] 
                         if 'geometric_entanglement_error' in result 
                         else 0.0 
                         for result in self.verification_results]
            
            ax3.plot(times, geo_errors, label='Geometric-Entanglement Error')
            ax3.set_xlabel('Time [t_P]')
            ax3.set_ylabel('Relative Error')
            ax3.set_title('Geometric-Entanglement Verification')
            ax3.set_yscale('log')
            ax3.legend()
            ax3.grid(True)
            
            # Field Equations Verification
            einstein_errors = [result['einstein_tensor_error'] 
                             if 'einstein_tensor_error' in result 
                             else 0.0 
                             for result in self.verification_results]
            quantum_errors = [result['quantum_tensor_error'] 
                            if 'quantum_tensor_error' in result 
                            else 0.0 
                            for result in self.verification_results]
            
            ax4.plot(times, einstein_errors, label='Einstein Tensor')
            ax4.plot(times, quantum_errors, label='Quantum Tensor')
            ax4.set_xlabel('Time [t_P]')
            ax4.set_ylabel('Relative Error')
            ax4.set_title('Field Equation Verification')
            ax4.set_yscale('log')
            ax4.legend()
            ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()


    def __init__(self, mass: float, quantum_gravity: 'QuantumGravity' = None, config_path: str = None):
        """Initialize black hole simulation."""
        if mass <= 0:
            raise ValueError("Black hole mass must be positive")
        # Initialize framework
        #self.qg = QuantumGravity(config_path)
        self.qg = quantum_gravity if quantum_gravity else QuantumGravity(config_path) 

        # Black hole parameters  
        self.initial_mass = mass
        self.horizon_radius = 2 * CONSTANTS['G'] * mass
        
        # Setup grid with proper points first
        self._setup_grid()
        
        # Now initialize quantum state
        self.qg.state = QuantumState(
            self.qg.grid,
            initial_mass=mass,  # Pass initial mass here
            eps_cut=self.qg.config.config['numerics']['eps_cut']
        )
        
        # Setup remaining components
        self._setup_initial_state()
        self._setup_observables()

        # Results storage
        self.time_points = []
        self.mass_history = []
        self.entropy_history = []  
        self.temperature_history = []
        self.radiation_flux_history = []

        # Add verification
        self.verifier = UnifiedTheoryVerification(self)
        self.verification_results = []
    def _setup_grid(self) -> None:
        """Setup grid for cosmological simulation."""
        # Configure grid for large-scale structure
        grid_config = self.qg.config.config['grid']
        
        # Create spatial grid with proper dimensionality
        L = 100.0  # Box size in Planck lengths
        N = grid_config['points_max']
        
        # Generate 3D grid points
        x = np.linspace(-L/2, L/2, int(np.cbrt(N)))
        points = np.array([[x_i, 0.0, 0.0] for x_i in x], dtype=np.float32)
        
        self.qg.grid.set_points(points)
        self.box_size = L
    def _setup_initial_state(self) -> None:
        """Setup initial state with time-dependent mass evolution."""
        state = self.qg.state

            # Add this line to initialize state mass
        state.mass = self.initial_mass  # Missing initialization
        state.initial_mass = self.initial_mass
        state.time = 0.0
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
    
        # Add small offset to prevent division by zero
        r = np.maximum(r, CONSTANTS['l_p'])  # Use Planck length as minimum
    
        # Store initial mass for evolution
        state.initial_mass = self.initial_mass
        state.time = 0.0
    
        # Calculate evaporation timescale
        evaporation_rate = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2)
        state.evaporation_timescale = (5120 * np.pi * CONSTANTS['G']**2 * state.initial_mass**3) / \
                                    (CONSTANTS['hbar'] * CONSTANTS['c']**4)
    
        # Calculate mass evolution
        def mass_at_time(t):
            """Calculate mass at time t using proper Hawking evaporation."""
            if t >= state.evaporation_timescale:
                return CONSTANTS['m_p']  # Return Planck mass as minimum
            return state.initial_mass * (1 - t/state.evaporation_timescale)**(1/3)
    
        # Update metric components with regularized radial coordinate
        g_tt = -(1 - 2*CONSTANTS['G']*mass_at_time(state.time)/r)
        g_rr = 1/(1 - 2*CONSTANTS['G']*mass_at_time(state.time)/r)
    
        # Set components efficiently
        state.set_metric_components_batch(
            [(0,0)]*len(r) + [(1,1)]*len(r),
            list(range(len(r)))*2,
            np.concatenate([g_tt, g_rr])
        )
     
            
    def _setup_observables(self) -> None:
        """Setup observables for black hole measurements."""
        # Horizon area observable
        self.area_obs = self.qg.physics.AreaObservable(
            self.qg.grid,
            normal=np.array([1.0, 0.0, 0.0])
        )
        
        # Mass observable (ADM mass)
        self.mass_obs = self.qg.physics.ADMMassObservable(
            self.qg.grid
        )
        
        # Temperature observable
        self.temp_obs = self.qg.physics.BlackHoleTemperatureObservable(
            self.qg.grid
        )
        
        # Radiation flux observable
        self.flux_obs = self.qg.physics.HawkingFluxObservable(
            self.qg.grid
        )

    def run_simulation(self, t_final: float) -> None:
        """Run black hole evolution simulation with geometric-entanglement verification."""
        # Add initial parameter logging
        logging.info(f"\nStarting Black Hole Simulation:")
        logging.info(f"Initial Mass: {self.initial_mass:.2e} Planck masses")
        logging.info(f"Simulation Time: {t_final:.2e} Planck times")
        logging.info(f"Initial Horizon Radius: {self.horizon_radius:.2e} Planck lengths\n")
    
        dt = 0.01  # Initial timestep
        t = 0.0
        
        # Track equation verification
        error_history = []
        lhs_history = []
        rhs_history = []

        # In run_simulation
        metrics = self.verifier._verify_geometric_entanglement(self.qg.state)
        if int(t/dt) % 100 == 0:  # Log every 100 steps
            self.verifier._log_verification_diagnostics(self.qg.state, metrics)

        while t < t_final:
            # Add verification step
            metrics = self.verifier.verify_unified_relations()
            
            # Store verification results
            self.verification_results.append({
                'time': t,
                'mass': self.qg.state.mass,
                **metrics
            })
            
            # Extract equation verification results
            lhs = metrics['geometric_entanglement_lhs']
            rhs = metrics['geometric_entanglement_rhs']
            error = metrics['geometric_entanglement_error']
            
            # Store for history
            error_history.append(error)
            lhs_history.append(lhs)
            rhs_history.append(rhs)
            
            # Update quantum state with mass loss
            #dm_dt = -(CONSTANTS['hbar'] * CONSTANTS['c']**6) / \
            #        (15360 * np.pi * CONSTANTS['G']**2 * self.qg.state.mass**2)
            #self.qg.state.mass += dm_dt * dt
            evaporation_rate = (CONSTANTS['hbar'] * CONSTANTS['c']**6) / \
                            (15360 * np.pi * CONSTANTS['G']**2 * self.qg.state.mass**2)
            
            # Compute mass loss
            #dm = evaporation_rate * dt * 1000.0
            original_dm = (self.qg.state.mass**2 * dt) / (15360 * np.pi)

            # Physical constants correction
            dm = original_dm * (CONSTANTS['c']**6 / CONSTANTS['G']**2)
            
            # logging.info(f"Time: {t}")
            # logging.info(f"Current Mass: {self.qg.state.mass}")
            # logging.info(f"Evaporation Rate: {evaporation_rate}")
            # logging.info(f"Mass Loss (dm): {dm:.16e}")

            # Update mass
            self.qg.state.mass = max(
                CONSTANTS['m_p'],  # Minimum mass is Planck mass
                self.qg.state.mass - dm
            )

            # Calculate derived quantities
            horizon_radius = 2 * CONSTANTS['G'] * self.qg.state.mass
            entropy = np.pi * horizon_radius**2 / (4 * CONSTANTS['l_p']**2)
            
            # Record measurements
            self.time_points.append(t)
            self.mass_history.append(self.qg.state.mass)
            self.entropy_history.append(entropy)
            
            # Log equation verification at intervals
            if int(t/t_final * 10) > int((t-dt)/t_final * 10):
                logging.info(f"\nGeometric-Entanglement Equation at t={t:.2f}:")
                logging.info(f"LHS (dS²)     = {lhs:.6e}")
                logging.info(f"RHS (integral) = {rhs:.6e}")
                logging.info(f"Relative Error = {error:.6e}")
                logging.info(f"Mass = {self.qg.state.mass:.9e}, Entropy = {entropy:.3e}")
                
                # Calculate running statistics
                mean_error = np.mean(error_history[-100:] if len(error_history) > 100 else error_history)
                max_error = np.max(error_history[-100:] if len(error_history) > 100 else error_history)
                logging.info(f"Recent Mean Error = {mean_error:.6e}, Max Error = {max_error:.6e}\n")
            
            t += dt
        
        # Final summary focused on equation verification
        logging.info("\nFinal Equation Verification Summary:")
        logging.info("dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩")
        logging.info(f"Final LHS = {lhs_history[-1]:.6e}")
        logging.info(f"Final RHS = {rhs_history[-1]:.6e}")
        logging.info(f"Final Error = {error_history[-1]:.6e}")
        logging.info(f"Overall Mean Error = {np.mean(error_history):.6e}")
        logging.info(f"Overall Max Error = {np.max(error_history):.6e}")
        
        # Store error history for later analysis
        self.equation_verification = {
            'errors': error_history,
            'lhs_values': lhs_history,
            'rhs_values': rhs_history,
            'times': self.time_points
        }

def main():
    """Run black hole simulation example."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize framework once
    config_path = None  # Use default config
    qg = QuantumGravity()
    #qg = QuantumGravity()
    test_masses = [100, 500, 1000, 2000, 5000]
    
    for initial_mass in test_masses:
        logging.info(f"\nRunning simulation for mass {initial_mass:.1f} Planck masses")
        
        # Pass existing QG instance to simulation
        sim = BlackHoleSimulation(initial_mass, quantum_gravity=qg)
        t_final = 1000.0
        sim.run_simulation(t_final)
        
        # Rest of the code remains the same
        
        output_dir = Path("results/black_hole")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sim.plot_results(str(output_dir / f"evolution_M{initial_mass:.0f}.png"))
        
        # Create measurement results for this mass
        measurements = [
            MeasurementResult(
                value=values,
                uncertainty=None,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'initial_mass': initial_mass
                }
            )
            for values in zip(sim.time_points, sim.mass_history,
                            sim.entropy_history, sim.temperature_history,
                            sim.radiation_flux_history)
        ]
        
        # Save measurements for this mass configuration
        io = QuantumGravityIO(str(output_dir))
        io.save_measurements(measurements, f"measurements_M{initial_mass:.0f}")

if __name__ == "__main__":
    main()

    
if __name__ == "__main__":
    main()