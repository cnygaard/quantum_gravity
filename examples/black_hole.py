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

from __init__ import QuantumGravity, configure_logging  # We haven't created QuantumGravity class yet!

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
        """Plot comprehensive simulation results."""
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))
        
        # Mass evolution with temperature
        ax1.plot(self.time_points, self.mass_history, 'b-', label='Mass')
        ax1_twin = ax1.twinx()
        ax1_twin.plot(self.time_points, self.temperature_history, 'r--', label='Temperature')
        ax1.set_xlabel('Time [t_P]')
        ax1.set_ylabel('Mass [m_P]', color='b')
        ax1_twin.set_ylabel('Temperature [T_P]', color='r')
        ax1.grid(True)
        
        # Entropy evolution
        ax2.plot(self.time_points, self.entropy_history)
        ax2.set_xlabel('Time [t_P]')
        ax2.set_ylabel('Entropy [k_B]')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        # Geometric-Entanglement Verification
        ax3.plot(self.time_points, [v['lhs'] for v in self.verification_results], label='LHS')
        ax3.plot(self.time_points, [v['rhs'] for v in self.verification_results], label='RHS')
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('ds² terms')
        ax3.set_yscale('log')
        ax3.legend()
        ax3.grid(True)
        
        # Beta parameter
        ax4.plot(self.time_points, [v['diagnostics']['beta'] for v in self.verification_results])
        ax4.set_xlabel('Time [t_P]')
        ax4.set_ylabel('β (l_p/r_h)')
        ax4.set_yscale('log')
        ax4.grid(True)
        
        # Gamma effective
        ax5.plot(self.time_points, [v['diagnostics']['gamma_eff'] for v in self.verification_results])
        ax5.set_xlabel('Time [t_P]')
        ax5.set_ylabel('γ_eff')
        ax5.set_yscale('log')
        ax5.grid(True)
        
        # Radiation flux
        ax6.plot(self.time_points, self.radiation_flux_history)
        ax6.set_xlabel('Time [t_P]')
        ax6.set_ylabel('Radiation Flux')
        ax6.set_yscale('log')
        ax6.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def plot_black_hole_geometry(self, save_path: str = None) -> None:
        """Plot black hole horizon and quantum properties in 3D."""
        # Increase figure size height for title and explanations
        fig = plt.figure(figsize=(15, 12))
        
        # Add main title
        fig.suptitle('Quantum Black Hole Visualization: Structure and Effects', 
                    fontsize=16, y=0.95)
        
        ax = fig.add_subplot(121, projection='3d')
        
        # Event horizon radius
        r = 2 * CONSTANTS['G'] * self.qg.state.mass
        
        # [Your existing grid and quantum effects calculations remain the same]
        n_radial = 25
        n_theta = 20
        n_phi = 40
        
        radial_factor = 2.5
        radii = r + (np.exp(np.linspace(0, np.log(2*r), n_radial)) - 1) / radial_factor
        thetas = np.linspace(0, np.pi, n_theta)
        phis = np.linspace(0, 2*np.pi, n_phi)
        
        R, T, P = np.meshgrid(radii, thetas, phis)
        
        X = R * np.sin(T) * np.cos(P)
        Y = R * np.sin(T) * np.sin(P)
        Z = R * np.cos(T)
        
        points = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        radial_distances = np.linalg.norm(points, axis=1)
        
        quantum_density = self.qg.state.entropy / (4 * np.pi * r**2)
        temp = self.temperature_history[-1]
        
        near_horizon = np.exp(-(radial_distances - r)/(1.5*self.qg.grid.l_p))
        hawking_rad = temp * np.exp(-(radial_distances - r)/(1.2*r))
        quantum_corrected = np.exp(-((radial_distances - r)/(4*self.qg.grid.l_p))**2)
        
        quantum_density = quantum_density * (
            near_horizon + 
            2.0 * hawking_rad + 
            0.5 * quantum_corrected
        )
        
        quantum_density = quantum_density / np.max(quantum_density)

        # Create two separate scatter plots - one for near horizon, one for outer region
        near_horizon_mask = radial_distances < (r * 1.2)  # Points very close to horizon
        outer_region_mask = ~near_horizon_mask

        # Plot outer (blue) points with more transparency
        scatter_outer = ax.scatter(points[outer_region_mask,0], 
                                points[outer_region_mask,1], 
                                points[outer_region_mask,2],
                                c=quantum_density[outer_region_mask],
                                cmap='plasma',
                                alpha=0.3,  # More transparent
                                s=6)        # Slightly smaller

        # Plot near-horizon (yellow) points with less transparency
        scatter_inner = ax.scatter(points[near_horizon_mask,0], 
                                points[near_horizon_mask,1], 
                                points[near_horizon_mask,2],
                                c=quantum_density[near_horizon_mask],
                                cmap='plasma',
                                alpha=0.8,  # Less transparent
                                s=10)       # Slightly

        
        # Horizon surface
        phi_surf = np.linspace(0, 2*np.pi, 100)
        theta_surf = np.linspace(0, np.pi, 50)
        phi_surf, theta_surf = np.meshgrid(phi_surf, theta_surf)
        
        x_h = r * np.sin(theta_surf) * np.cos(phi_surf)
        y_h = r * np.sin(theta_surf) * np.sin(phi_surf)
        z_h = r * np.cos(theta_surf)
        
        surf = ax.plot_surface(x_h, y_h, z_h,
                            color='black',
                            alpha=0.7,
                            antialiased=True)
        
        # Add 3D plot explanation
        ax.text2D(0.22, -0.35, 
'Quantum Effects Visualization:\n' +
                '• Yellow/bright: Strong quantum effects near horizon\n' +
                '• Blue: Weakening quantum effects with distance\n' +
                '• Black surface: Event horizon\n\n' +
                'Geometric-Entanglement Formula:\n' +
                r'dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩',
                transform=ax.transAxes,
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        #cbar = plt.colorbar(scatter, label='Quantum Effects Intensity')
        cbar = plt.colorbar(scatter_outer, label='Quantum Effects Intensity')
        cbar.set_label('Normalized Quantum Effects Intensity', size=10)
        
        ax.set_xlabel('x [l_p]')
        ax.set_ylabel('y [l_p]')
        ax.set_zlabel('z [l_p]')
        ax.set_box_aspect([1,1,1])
        
        # 2D projection
        ax2 = fig.add_subplot(122)
        
        horizon = plt.Circle((0, 0), r, fill=False, color='black', 
                            label='Event Horizon', linewidth=2)
        ergosphere = plt.Circle((0, 0), 2*r, fill=False, color='red',
                            linestyle='--', label='Ergosphere', linewidth=2)
        ax2.add_artist(horizon)
        ax2.add_artist(ergosphere)
        
        # 2D visualization
        xy_points = points[:,:2]
        mask_2d = (np.abs(points[:,2]) < r/5)
        scatter2d = ax2.scatter(xy_points[mask_2d,0], xy_points[mask_2d,1],
                            c=quantum_density[mask_2d],
                            cmap='plasma',
                            alpha=0.7,
                            s=12)
        
        # Add 2D plot explanation
        ax2.text(0.02, 0.98,
                'Structure Features:\n' +
                '• Black circle: Event horizon (r = 2GM)\n' +
                '• Red dashed: Ergosphere (r = 4GM)\n' +
                '• Points: Quantum density distribution\n' +
                '• Colors gradient: Quantum effects intensity',
                transform=ax2.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        ax2.set_xlabel('x [l_p]')
        ax2.set_ylabel('y [l_p]')
        ax2.set_xlim(-150, 150)
        ax2.set_ylim(-150, 150)
        ax2.axis('equal')
        ax2.legend(loc='upper right')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust for title space
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def __init__(self, mass: float, quantum_gravity: 'QuantumGravity' = None, config_path: str = None):
        """Initialize black hole simulation."""
        if mass <= 0:
            raise ValueError("Black hole mass must be positive")
        # Initialize framework
        #self.qg = QuantumGravity(config_path)
        self.qg = quantum_gravity if quantum_gravity else QuantumGravity() 

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

    def log_physics_output(self, t: float):
        """Log comprehensive physics parameters and formulas."""
        # Calculate physical parameters
        self.temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.qg.state.mass)
        self.beta = CONSTANTS['l_p'] / self.horizon_radius
        self.gamma_eff = self.verifier.gamma * self.beta * np.sqrt(0.407)
        
        # Get verification metrics
        geo_metrics = self.verifier._verify_geometric_entanglement(self.qg.state)
        self.ds2 = float(geo_metrics['lhs'])  # Convert to float for formatting
        self.integral = float(geo_metrics['rhs'])
        self.entropy = np.pi * self.horizon_radius**2 / (4 * CONSTANTS['l_p']**2)

        # Log physics output with proper formatting
        logging.info("\nQuantum Black Hole Physics at t = %.2f:", t)
        logging.info("Classical Parameters:")
        logging.info(f"Mass: {self.qg.state.mass:.2e} M_p")
        logging.info(f"Horizon Radius: {self.horizon_radius:.2e} l_p")
        logging.info(f"Temperature: {self.temperature:.2e} T_p")
        logging.info(f"Entropy: {self.entropy:.2e} k_B")
        
        logging.info("\nQuantum Parameters:")
        logging.info(f"β (l_p/r_h): {self.beta:.2e}")
        logging.info(f"γ_eff: {self.gamma_eff:.2e}")
        
        logging.info("\nGeometric-Entanglement Formula:")
        logging.info("dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩")
        logging.info(f"LHS = {self.ds2:.2e}")
        logging.info(f"RHS = {self.integral:.2e}")
    
            
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
            #metrics = self.verifier.verify_unified_relations()
            metrics = self.verifier._verify_geometric_entanglement(self.qg.state)
            # Store verification results
            self.verification_results.append({
                'time': t,
                'mass': self.qg.state.mass,
                **metrics
            })
            
            # Extract equation verification results
            #lhs = metrics['geometric_entanglement_lhs']
            #rhs = metrics['geometric_entanglement_rhs']
            #error = metrics['geometric_entanglement_error']
            lhs = metrics['lhs']
            rhs = metrics['rhs']
            error = metrics['relative_error']

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
            

            # Calculate temperature
            temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.qg.state.mass)

            # Record measurements
            self.time_points.append(t)
            self.mass_history.append(self.qg.state.mass)
            self.entropy_history.append(entropy)
            self.temperature_history.append(temperature)
            self.radiation_flux_history.append(evaporation_rate)
            
            # Log equation verification at intervals
            if int(t/t_final * 10) > int((t-dt)/t_final * 10):
                self.log_physics_output(t)
                # logging.info(f"\nGeometric-Entanglement Equation at t={t:.2f}:")
                # logging.info(f"LHS (dS²)     = {lhs:.6e}")
                # logging.info(f"RHS (integral) = {rhs:.6e}")
                # logging.info(f"Relative Error = {error:.6e}")
                # logging.info(f"Mass = {self.qg.state.mass:.9e}, Entropy = {entropy:.3e}")
                
                # # Calculate running statistics
                # mean_error = np.mean(error_history[-100:] if len(error_history) > 100 else error_history)
                # max_error = np.max(error_history[-100:] if len(error_history) > 100 else error_history)
                # logging.info(f"Recent Mean Error = {mean_error:.6e}, Max Error = {max_error:.6e}\n")
            
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
#    logging.basicConfig(level=logging.INFO)
    configure_logging()


    # # Configure logging
    # logging.basicConfig(
    #     level=logging.INFO,
    #     format='%(message)s',  # Simplified format
    #     force=True,  # Override any existing handlers
    #     handlers=[
    #         logging.StreamHandler(),  # Console handler
    #         #logging.FileHandler('simulation.log')  # File handler
    #     ]
    # )

    # Initialize framework once
    config_path = None  # Use default config
    quantum_gravity = QuantumGravity()
    #test_masses = [100, 500, 1000, 2000, 5000]
    test_masses = [1000]

    for initial_mass in test_masses:
        logging.info(f"\nRunning simulation for mass {initial_mass:.1f} Planck masses")
        
        # Pass existing QG instance to simulation
        sim = BlackHoleSimulation(initial_mass, quantum_gravity=quantum_gravity)
        t_final = 1000.0
        sim.run_simulation(t_final)
        
        # Rest of the code remains the same
        # Inside run_simulation() where other measurements are recorded
        #self.temperature_history.append(self.temperature)
        #self.radiation_flux_history.append(evaporation_rate)        

        output_dir = Path("results/black_hole")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        sim.plot_results(str(output_dir / f"evolution_M{initial_mass:.0f}.png"))

        # Plot black hole geometry
        sim.plot_black_hole_geometry(str(output_dir / f"black_hole_geometry_M{initial_mass:.0f}.png"))


        # Create measurement results for this mass
        measurements = [
            MeasurementResult(
                value={
                    'time': t,
                    'mass': m,
                    'entropy': s,
                    'temperature': temp,
                    'radiation_flux': flux,
                    'geometric_ds2_lhs': sim.verification_results[i]['lhs'],
                    'geometric_ds2_rhs': sim.verification_results[i]['rhs'],
                    'beta_lp_rh': sim.verification_results[i]['diagnostics']['beta'],
                    'gamma_eff': sim.verification_results[i]['diagnostics']['gamma_eff']
                },
                uncertainty=None,
                metadata={
                    'timestamp': datetime.now().isoformat(),
                    'initial_mass': initial_mass
                }
            )
            for i, (t, m, s, temp, flux) in enumerate(zip(
                sim.time_points, 
                sim.mass_history,
                sim.entropy_history, 
                sim.temperature_history,
                sim.radiation_flux_history
            ))
        ]
        
        # Save measurements for this mass configuration
        io = QuantumGravityIO(str(output_dir))
        io.save_measurements(measurements, f"measurements_M{initial_mass:.0f}")

if __name__ == "__main__":
    main()

    
if __name__ == "__main__":
    main()


