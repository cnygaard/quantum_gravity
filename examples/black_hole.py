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
import matplotlib.pyplot as plt
from constants import CONSTANTS
#from quantum_gravity import QuantumGravity, CONSTANTS

class BlackHoleSimulation:
    """Quantum black hole simulation."""
    
    def __init__(self, 
                 mass: float,
                 config_path: str = None):
        """Initialize black hole simulation.
        
        Args:
            mass: Initial black hole mass in Planck units
            config_path: Optional path to configuration file
        """
        # Initialize framework
        self.qg = QuantumGravity(config_path)
        
        # Black hole parameters
        self.initial_mass = mass
        self.horizon_radius = 2 * CONSTANTS['G'] * mass
        
        # Setup simulation
        self._setup_grid()
        self._setup_initial_state()
        self._setup_observables()
        
        # Results storage
        self.time_points = []
        self.mass_history = []
        self.entropy_history = []
        self.temperature_history = []
        self.radiation_flux_history = []
        
    def _setup_grid(self) -> None:
        """Setup adaptive grid focused on horizon."""
        # Configure grid with higher resolution near horizon
        grid_config = self.qg.config.config['grid']
        grid_config['refinement_factor'] = 4.0  # Increased resolution
        
        # Create grid points with exponential spacing
        r_min = CONSTANTS['l_p']
        r_max = 100 * self.horizon_radius
        
        # Generate radial points
        r_points = np.geomspace(r_min, r_max, grid_config['points_max'])
        
        # Convert to 3D points with spherical symmetry
        theta = np.linspace(0, np.pi, 20)
        phi = np.linspace(0, 2*np.pi, 40)
        
        points = []
        for r in r_points:
            for t in theta:
                for p in phi:
                    x = r * np.sin(t) * np.cos(p)
                    y = r * np.sin(t) * np.sin(p)
                    z = r * np.cos(t)
                    points.append([x, y, z])
        
        self.qg.grid.set_points(np.array(points))
        
    def _setup_initial_state(self) -> None:
        """Setup initial quantum state for black hole."""
        # Create initial state representing classical black hole
        state = self.qg.state
        
        # Set up metric coefficients
        r = np.sqrt(np.sum(self.qg.grid.points**2, axis=1))
        
        for i, ri in enumerate(r):
            # Schwarzschild metric components
            if ri > self.horizon_radius:
                g_tt = -(1 - 2*CONSTANTS['G']*self.initial_mass/ri)
                g_rr = 1/(1 - 2*CONSTANTS['G']*self.initial_mass/ri)
            else:
                # Inside horizon regularity conditions
                g_tt = -1e-10  # Small negative value
                g_rr = 1e10    # Large positive value
                
            # Add quantum corrections
            quantum_factor = 1 + (CONSTANTS['l_p']/ri)**2
            g_tt *= quantum_factor
            g_rr *= quantum_factor
            
            # Set metric components in state
            state.set_metric_component((0, 0), i, g_tt)
            state.set_metric_component((1, 1), i, g_rr)
            state.set_metric_component((2, 2), i, ri**2)
            state.set_metric_component((3, 3), i, ri**2 * np.sin(np.arccos(self.qg.grid.points[i,2]/ri))**2)
            
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
        
    def run_simulation(self, 
                      t_final: float,
                      dt_save: float = None) -> None:
        """Run black hole evolution simulation.
        
        Args:
            t_final: Final time in Planck units
            dt_save: Time interval for saving results
        """
        if dt_save is None:
            dt_save = t_final / 100
            
        def callback(state, t, step):
            """Callback function for measurements."""
            if step % int(dt_save / self.qg.evolution.dt) == 0:
                self._record_measurements(t)
                
        # Run evolution
        self.qg.run_simulation(t_final, callback)
        
    def _record_measurements(self, t: float) -> None:
        """Record measurements at current time."""
        # Measure observables
        mass = self.mass_obs.measure(self.qg.state)
        area = self.area_obs.measure(self.qg.state)
        temp = self.temp_obs.measure(self.qg.state)
        flux = self.flux_obs.measure(self.qg.state)
        
        # Compute entropy
        entropy = area.value / (4 * CONSTANTS['l_p']**2)
        
        # Store results
        self.time_points.append(t)
        self.mass_history.append(mass.value)
        self.entropy_history.append(entropy)
        self.temperature_history.append(temp.value)
        self.radiation_flux_history.append(flux.value)
        
        # Log current measurements
        logging.info(f"Time t={t:.2f}: Mass={mass.value:.2f}, Temperature={temp.value:.2e}, " +
                    f"Entropy={entropy:.2e}, Radiation Flux={flux.value:.2e}")
        
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