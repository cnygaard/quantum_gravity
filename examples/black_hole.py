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
        """Run black hole evolution simulation."""
        dt = 0.01  # Initial timestep
        t = 0.0
        
        while t < t_final:
            # Add verification step
            metrics = self.verifier.verify_unified_relations()
            self.verification_results.append({
                'time': t,
                'mass': self.qg.state.mass,
                **metrics
            })
            # Verify unified theory
            trinity_metrics = self.verifier.verify_spacetime_trinity()

            # Log verification results
            if int(t/t_final * 100) > int((t-dt)/t_final * 100):
                logging.info(f"Trinity verification at t={t:.2f}:")
                for key, value in trinity_metrics.items():
                    logging.info(f"  {key}: {value:.6e}")
    
            # Update quantum state with mass loss
            self.qg.state.mass -= (self.qg.state.mass**2 * dt) / (15360 * np.pi)
            
            # Calculate derived quantities
            horizon_radius = 2 * CONSTANTS['G'] * self.qg.state.mass
            temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.qg.state.mass)
            entropy = np.pi * horizon_radius**2 / (4 * CONSTANTS['l_p']**2)
            flux = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2 * self.qg.state.mass**2)
            
            # Record measurements (preserving history)
            self.time_points.append(t)
            self.mass_history.append(self.qg.state.mass)
            self.entropy_history.append(entropy)
            self.temperature_history.append(temperature)
            self.radiation_flux_history.append(flux)
            
            # Enhanced logging with both progress and parameters
            if int(t/t_final * 100) > int((t-dt)/t_final * 100):
                logging.info(f"Time t={t:.2f}: Mass={self.qg.state.mass:.6e}, "
                            f"Temperature={temperature:.6e}, Entropy={entropy:.6e}, "
                            f"Radiation Flux={flux:.6e}")
                logging.info(f"Simulation progress: {t/t_final*100:.1f}% (t={t:.2f}/{t_final})")
                
            t += dt
    def _record_measurements(self, t: float) -> None:
        """Record measurements at current time."""
        mass = self.mass_obs.measure(self.qg.state)
        area = self.area_obs.measure(self.qg.state)
        temp = self.temp_obs.measure(self.qg.state)
        flux = self.flux_obs.measure(self.qg.state)
    
        entropy = area.value / (4 * CONSTANTS['l_p']**2)
    
        # Store results with full precision
        self.time_points.append(t)
        self.mass_history.append(mass.value)
        self.entropy_history.append(entropy)
        self.temperature_history.append(temp.value)
        self.radiation_flux_history.append(flux.value)
    
        # Log with increased precision
        logging.info(f"Time t={t:.2f}: Mass={mass.value:.6e}, Temperature={temp.value:.6e}, " +
                    f"Entropy={entropy:.6e}, Radiation Flux={flux.value:.6e}")
        
    def plot_results(self, save_path: str = None) -> None:
        """Plot simulation results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
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
        
        # Temperature evolution
        ax3.plot(self.time_points, self.temperature_history)
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('Temperature [T_P]')
        ax3.set_title('Black Hole Temperature Evolution')
        ax3.grid(True)
        
        # Radiation flux
        ax4.plot(self.time_points, self.radiation_flux_history)
        ax4.set_xlabel('Time [t_P]')
        ax4.set_ylabel('Radiation Flux [P_P]')
        ax4.set_title('Hawking Radiation Flux')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.show()
        
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