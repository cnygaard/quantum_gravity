#!/usr/bin/env python
# examples/cosmology.py

"""
Quantum Cosmology Simulation
===========================

This example demonstrates the simulation of quantum cosmological
scenarios, including universe expansion, inflation, and structure
formation with quantum corrections.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from quantum_gravity import QuantumGravity, CONSTANTS

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
        
        # Set up FLRW metric with quantum corrections
        for i, point in enumerate(self.qg.grid.points):
            # Classical FLRW metric components
            a = self.initial_scale
            
            # Spatial metric components with quantum corrections
            quantum_factor = 1 + (CONSTANTS['l_p']/a)**2
            
            # Set metric components
            state.set_metric_component((0, 0), i, -1)  # Proper time
            for j in range(3):
                state.set_metric_component((j+1, j+1), i, a**2 * quantum_factor)
                
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
        
    def run_simulation(self,
                      t_final: float,
                      dt_save: float = None) -> None:
        """Run cosmological evolution simulation.
        
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
            ax4.loglog(k, Pk)
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
        H /= self.scale_factor_history
        
        # Deceleration parameter
        a = np.array(self.scale_factor_history)
        q = -a * np.gradient(np.gradient(a, self.time_points), self.time_points) / \
            np.gradient(a, self.time_points)**2
            
        return {
            'hubble_parameter': H,
            'deceleration_parameter': q
        }
        
def main():
    """Run cosmology simulation example."""
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initial cosmological parameters
    initial_scale = 1000.0  # in Planck lengths
    hubble_parameter = 0.1  # in Planck units
    
    # Create and run simulation
    sim = CosmologySimulation(initial_scale, hubble_parameter)
    
    # Run until significant expansion
    t_final = 1000.0  # in Planck times
    sim.run_simulation(t_final)
    
    # Plot and save results
    output_dir = Path("results/cosmology")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sim.plot_results(str(output_dir / "evolution.png"))
    
    # Compute and save derived quantities
    derived = sim.compute_derived_quantities()
    
    # Save all data
    sim.qg.io.save_measurements({
        'time': sim.time_points,
        'scale_factor': sim.scale_factor_history,
        'energy_density': sim.energy_density_history,
        'quantum_corrections': sim.quantum_corrections_history,
        'hubble_parameter': derived['hubble_parameter'].tolist(),
        'deceleration_parameter': derived['deceleration_parameter'].tolist()
    }, str(output_dir / "measurements.json"))
    
if __name__ == "__main__":
    main()