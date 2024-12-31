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
import matplotlib
matplotlib.use('TkAgg')  # Or 'Qt5Agg' if you prefer Qt
import matplotlib.pyplot as plt
from constants import CONSTANTS
from core.state import QuantumState
#from core.constants import CONSTANTS


#from quantum_gravity import QuantumGravity, CONSTANTS

class BlackHoleSimulation:
    """Quantum black hole simulation."""
    def __init__(self, mass: float, config_path: str = None):
        """Initialize black hole simulation."""
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
    def _setup_grid(self):
        """Setup adaptive grid focused on horizon."""
        grid_config = self.qg.config.config['grid']
        
        # Reduced grid parameters for memory efficiency
        n_radial = 30
        n_theta = 8
        n_phi = 16
        
        # Generate grid points
        r_min = CONSTANTS['l_p']
        r_max = 20 * self.horizon_radius
        
        r_points = np.geomspace(r_min, r_max, n_radial, dtype=np.float32)
        theta = np.linspace(0, np.pi, n_theta, dtype=np.float32)
        phi = np.linspace(0, 2*np.pi, n_phi, dtype=np.float32)
        
        # Calculate points efficiently
        points_list = []
        for r in r_points:
            for t in theta:
                sin_t = np.sin(t)
                cos_t = np.cos(t)
                for p in phi:
                    points_list.append([
                        r * sin_t * np.cos(p),
                        r * sin_t * np.sin(p),
                        r * cos_t
                    ])
                    
        points = np.array(points_list, dtype=np.float32)
        self.qg.grid.set_points(points)
        
    def _setup_initial_state(self) -> None:
        """Setup initial state with time-dependent mass evolution."""
        state = self.qg.state
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
        
        # Store initial mass for evolution
        state.initial_mass = self.initial_mass
        state.time = 0.0
        
        # Calculate evaporation timescale
        evaporation_rate = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2)
        state.evaporation_timescale = state.initial_mass**3 / evaporation_rate
        
        # Set metric with mass evolution
        def mass_at_time(t):
            return state.initial_mass * (1 - t/state.evaporation_timescale)**(1/3)
        
        # Update metric components with time dependence
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
            # Update quantum state
            self.qg.state.mass -= (self.qg.state.mass**2 * dt) / (15360 * np.pi)  # Mass loss rate
            
            # Update derived quantities
            horizon_radius = 2 * CONSTANTS['G'] * self.qg.state.mass
            temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.qg.state.mass)
            entropy = np.pi * horizon_radius**2 / (4 * CONSTANTS['l_p']**2)
            flux = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2 * self.qg.state.mass**2)
            
            # Record measurements
            self.time_points.append(t)
            self.mass_history.append(self.qg.state.mass)
            self.entropy_history.append(entropy)
            self.temperature_history.append(temperature)
            self.radiation_flux_history.append(flux)
            
            # Log progress
            if int(t/t_final * 100) > int((t-dt)/t_final * 100):
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
    
    # Save data
    sim.qg.io.save_measurements({
        'time': sim.time_points,
        'mass': sim.mass_history,
        'entropy': sim.entropy_history,
        'temperature': sim.temperature_history,
        'radiation_flux': sim.radiation_flux_history
    }, str(output_dir / "measurements.json"))
    
if __name__ == "__main__":
    main()