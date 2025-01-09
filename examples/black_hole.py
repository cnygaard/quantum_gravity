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
    def __init__(self, mass: float, config_path: str = None):
        """Initialize black hole simulation."""
        if mass <= 0:
            raise ValueError("Black hole mass must be positive")
        # Initialize framework
        self.qg = QuantumGravity(config_path)
        
        # Black hole parameters  
        self.initial_mass = mass
        self.horizon_radius = 2 * CONSTANTS['G'] * mass
        
        # Setup grid with proper points first
        self._setup_grid()
        
        # Now initialize quantum state
        self.qg.state = QuantumState(
            self.qg.grid,
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
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
    
        # Add small offset to prevent division by zero
        r = np.maximum(r, CONSTANTS['l_p'])  # Use Planck length as minimum
    
        # Store initial mass for evolution
        state.initial_mass = self.initial_mass
        state.time = 0.0
    
        # Calculate evaporation timescale
        evaporation_rate = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2)
        state.evaporation_timescale = state.initial_mass**3 / evaporation_rate
    
        # Calculate mass evolution
        def mass_at_time(t):
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
        dt = 0.01  # Initial timestep
        t = 0.0
        
        # Track equation verification
        error_history = []
        lhs_history = []
        rhs_history = []
        
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
            self.qg.state.mass -= (self.qg.state.mass**2 * dt) / (15360 * np.pi)
            
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
                logging.info(f"Mass = {self.qg.state.mass:.3e}, Entropy = {entropy:.3e}")
                
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

def plot_results(self, save_path: str = None) -> None:
    """Plot simulation results including geometric-entanglement verification."""
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
    if hasattr(self, 'equation_verification'):
        ax3.plot(self.equation_verification['times'], 
                np.abs(self.equation_verification['lhs_values']), 
                label='|LHS|')
        ax3.plot(self.equation_verification['times'], 
                np.abs(self.equation_verification['rhs_values']), 
                label='|RHS|')
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('|dS²| and |Integral|')
        ax3.set_title('Geometric-Entanglement Terms')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Error evolution
        ax4.plot(self.equation_verification['times'], 
                self.equation_verification['errors'])
        ax4.set_xlabel('Time [t_P]')
        ax4.set_ylabel('Relative Error')
        ax4.set_title('Geometric-Entanglement Error')
        ax4.set_yscale('log')
        ax4.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.close()            
        
def main():
    """Run black hole simulation example."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initial black hole parameters
    initial_mass = 1000.0  # in Planck masses
    
    # Create and run simulation
    sim = BlackHoleSimulation(initial_mass)
    
    # Run until significant mass loss
    t_final = 1000.0  # in Planck times
    sim.run_simulation(t_final)
    
    # Plot and save results
    output_dir = Path("results/black_hole")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim.plot_results(str(output_dir / "evolution.png"))
    
    # Create measurement results using the MeasurementResult dataclass
    measurements = [
        MeasurementResult(
            value=values,
            uncertainty=None,
            metadata={'timestamp': datetime.now().isoformat()}
        )
        for values in zip(sim.time_points, sim.mass_history, 
                         sim.entropy_history, sim.temperature_history,
                         sim.radiation_flux_history)
    ]
    
    # Use the IO utility class to save measurements
    io = QuantumGravityIO(str(output_dir))
    io.save_measurements(measurements, "measurements")
    
if __name__ == "__main__":
    main()