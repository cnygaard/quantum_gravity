#!/usr/bin/env python

# examples/galaxy.py

"""
Quantum Galaxy Simulation
========================

This example demonstrates the simulation of a galaxy with quantum gravity effects,
including dark matter distribution, rotation curves, and geometric-entanglement verification.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Rename the file for compatibility with how it's being called
current_file = Path(__file__)
if current_file.name == "galaxy-simulation.py":
    # Set the current file name for proper imports when run directly
    __file__ = os.path.join(current_file.parent, "galaxy.py")

from __init__ import QuantumGravity, configure_logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from constants import CONSTANTS, SI_UNITS
from core.state import QuantumState
from physics.verification import UnifiedTheoryVerification, DarkMatterVerification
from utils.io import QuantumGravityIO, MeasurementResult
from matplotlib.gridspec import GridSpec
from scipy.interpolate import interp1d
from physics.models.dark_matter import DarkMatterAnalysis
from physics.observables import (
    RobustEntanglementObservable, ADMMassObservable, 
    EnergyDensityObservable, QuantumCorrectionsObservable
)


class GalaxySimulation:
    """Quantum galaxy simulation with dark matter and geometric corrections."""
    
    def __init__(self, 
                 stellar_mass: float,  # in solar masses 
                 radius: float,        # in kiloparsecs
                 galaxy_type: str = 'spiral',
                 bulge_fraction: float = 0.2,
                 dark_matter_ratio: float = 5.0,
                 quantum_gravity: 'QuantumGravity' = None):
        """Initialize galaxy simulation.
        
        Args:
            stellar_mass: Visible/stellar mass in solar masses
            radius: Galaxy radius in kiloparsecs
            galaxy_type: Type of galaxy ('spiral', 'elliptical', 'dwarf')
            bulge_fraction: Fraction of mass in central bulge (0-1)
            dark_matter_ratio: Ratio of dark matter to visible matter
            quantum_gravity: Optional QuantumGravity instance to share
        """
        # Initialize logging
        configure_logging(simulation_type='galaxy', log_file=f"galaxy_{galaxy_type}_{stellar_mass:.1e}")
        
        # Initialize framework
        self.qg = quantum_gravity if quantum_gravity else QuantumGravity()
        
        # Galaxy parameters
        self.initial_stellar_mass = stellar_mass * CONSTANTS['M_sun']  # Convert to Planck units
        self.radius_kpc = radius
        self.radius = radius * 3.086e19  # Convert kpc to Planck length
        self.R_star = self.radius  # Add R_star for compatibility with QuantumState
        self.galaxy_type = galaxy_type

        self.bulge_fraction = bulge_fraction
        self.dark_matter_ratio = dark_matter_ratio
        
        # Derived parameters
        self.dark_mass = self.initial_stellar_mass * dark_matter_ratio
        self.total_mass = self.initial_stellar_mass + self.dark_mass
        self.scale_radius = self.radius * 0.2  # NFW scale radius (typically ~20% of virial radius)
        
        # Initialize verification
        self.verifier = UnifiedTheoryVerification(self)
        self.dm_verifier = DarkMatterVerification(self)

        # Initialize quantum geometric parameters
        self.beta = CONSTANTS['l_p'] / self.radius  # Quantum scale parameter
        self.gamma = 0.55  # Coupling constant
        self.gamma_eff = self.gamma * self.beta * np.sqrt(0.364840)  # Effective coupling
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio for quantum NFW profile


        # Setup grid with proper points
        self._setup_grid()
        
        # Initialize quantum state
        self._setup_initial_state()
        self._setup_observables()
        
        
        # Initialize dark matter analysis
        self.dm_analysis = DarkMatterAnalysis(
            observed_mass=stellar_mass,
            total_mass=stellar_mass+self.dark_mass,
            radius=radius*3.26, # Convert kpc to light years
            velocity_dispersion=220.0, # km/s
            visible_mass=stellar_mass
        )
        
        # Results storage
        self.time_points = []
        self.rotation_curves = []
        self.mass_profiles = []
        self.entropy_history = []
        self.verification_results = []
        self.dark_matter_density = []
        
        logging.info(f"Galaxy simulation initialized:")
        logging.info(f"  Type: {self.galaxy_type}")
        logging.info(f"  Stellar Mass: {stellar_mass:.2e} M☉")
        logging.info(f"  Radius: {radius:.2f} kpc")
        logging.info(f"  Dark Matter Ratio: {dark_matter_ratio:.2f}")
        logging.info(f"  Total Mass: {self.total_mass/CONSTANTS['M_sun']:.2e} M☉")
        logging.info(f"  β (l_p/R): {self.beta:.2e}")
        logging.info(f"  γ_eff: {self.gamma_eff:.2e}")
    
    def _setup_grid(self) -> None:
        """Setup grid for galaxy simulation with appropriate distribution."""
        # Configure grid
        grid_config = self.qg.config.config['grid']
        N = grid_config['points_max']
        
        # Generate galaxy-specific grid points
        points = self._generate_galaxy_grid_points(N)
        
        # Set grid points
        self.qg.grid.set_points(points)
        logging.info(f"Grid initialized with {len(points)} points")
    
    def _generate_galaxy_grid_points(self, n_points: int) -> np.ndarray:
        """Generate grid points with galaxy-like distribution.
        
        Creates points with higher density in galactic bulge and
        exponentially decreasing density in disk, with halo.
        
        Args:
            n_points: Maximum number of grid points
            
        Returns:
            Array of 3D coordinates
        """
        points = []
        
        # Allocate points between components
        n_bulge = int(n_points * self.bulge_fraction)
        n_disk = int(n_points * 0.7)  # 70% in disk
        n_halo = n_points - n_bulge - n_disk  # Remainder in halo
        
        # Generate bulge points (spherical distribution)
        bulge_radius = self.radius * 0.1  # 10% of galaxy radius
        for i in range(n_bulge):
            # Use Hernquist profile for bulge
            r = bulge_radius * np.cbrt(np.random.random()) / (1 - np.random.random())
            if r > bulge_radius:
                r = bulge_radius * np.random.random()  # Fallback
                
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            points.append([x, y, z])
        
        # Generate disk points
        disk_scale_length = self.radius * 0.3  # Scale length
        disk_scale_height = self.radius * 0.05  # Scale height
        
        for i in range(n_disk):
            # Exponential disk profile
            r = disk_scale_length * np.random.exponential()
            if r > self.radius:
                r = self.radius * np.random.random()  # Fallback
                
            phi = 2 * np.pi * np.random.random()
            z = disk_scale_height * np.random.exponential() * (1 if np.random.random() > 0.5 else -1)
            
            # Apply spiral arm perturbation for spiral galaxies
            if self.galaxy_type == 'spiral':
                # Simple logarithmic spiral arms
                n_arms = 2
                arm_phase = np.random.normal(0, 0.2)  # Random phase with perturbation
                arm_strength = 0.3 * np.exp(-r / disk_scale_length)  # Arm strength decreases with radius
                
                # Apply spiral perturbation
                phi += arm_strength * np.sin(n_arms * np.log(r/disk_scale_length) + arm_phase)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = z
            
            points.append([x, y, z])
        
        # Generate halo points
        halo_scale = self.radius * 2  # Halo extends beyond visible galaxy
        for i in range(n_halo):
            # NFW-like profile for halo
            r = halo_scale * np.random.random()**0.5  # More points near the center
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            points.append([x, y, z])
        
        return np.array(points)
    
    def _setup_initial_state(self) -> None:
        """Setup initial state for galaxy simulation."""
        # Initialize quantum state
        self.qg.state = QuantumState(
            self.qg.grid,
            initial_mass=self.total_mass,
            eps_cut=self.qg.config.config['numerics']['eps_cut'],
            simulation=self  # Pass simulation reference
        )
        
        # Set time to zero
        self.qg.state.time = 0.0
        
        # Initialize metric tensor
        self._update_metric()
        
        # Set evolution timescale (galactic rotation period)
        typical_velocity = 220.0  # km/s
        self.rotation_period = 2 * np.pi * self.radius / (typical_velocity * 1000)  # seconds
        self.qg.state.evolution_timescale = self.rotation_period
        
        # Add galaxy-specific properties
        self.qg.state.stellar_mass = self.initial_stellar_mass
        self.qg.state.dark_mass = self.dark_mass
        self.qg.state.radius = self.radius
        self.qg.state.scale_radius = self.scale_radius
        self.qg.state.galaxy_type = self.galaxy_type
        self.qg.state.galaxy_radius = self.radius  # Add galaxy_radius for DarkMatterVerification
        self.qg.state.gamma = self.gamma  # Also add gamma since DarkMatterVerification uses it
    
    def _update_metric(self) -> None:
        """Update metric with galaxy-specific effects."""
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
        
        # Prevent division by zero
        r = np.maximum(r, CONSTANTS['l_p'])
        
        # Compute Newtonian potential
        phi_N = -CONSTANTS['G'] * self.total_mass / r
        
        # Apply quantum corrections
        phi_quantum = phi_N * (1 + self.gamma_eff * np.exp(-r/self.scale_radius))
        
        # Update metric components with weak field approximation
        g_tt = -(1 + 2*phi_quantum/CONSTANTS['c']**2)
        g_rr = 1 - 2*phi_quantum/CONSTANTS['c']**2
        
        # Set components efficiently
        self.qg.state.set_metric_components_batch(
            [(0,0)]*len(r) + [(1,1)]*len(r),
            list(range(len(r)))*2,
            np.concatenate([g_tt, g_rr])
        )
    
    def _setup_observables(self) -> None:
        """Setup observables for galaxy measurements."""
        self.mass_obs = ADMMassObservable(self.qg.grid)
        self.energy_obs = EnergyDensityObservable(self.qg.grid)
        self.quantum_corrections_obs = QuantumCorrectionsObservable(self.qg.grid)
        
        # Use robust entanglement observable instead
        self.entanglement_obs = RobustEntanglementObservable(
            self.qg.grid, 
            region_A=list(range(int(len(self.qg.grid.points) * 0.5)))  # Use half of points
        )
        
        # Galaxy-specific observables
        self.rotation_curve_obs = GalaxyRotationCurveObservable(self.qg.grid)
        self.dark_matter_obs = DarkMatterDensityObservable(self.qg.grid)
        self.velocity_dispersion_obs = VelocityDispersionObservable(self.qg.grid)

        # self.mass_obs = ADMMassObservable(self.qg.grid)
        # self.entanglement_obs = EntanglementObservable(
        # self.qg.grid, 
        #     region_A=list(range(int(len(self.qg.grid.points) * 0.5)))  # Use half of points
        # )
        # self.energy_obs = EnergyDensityObservable(self.qg.grid)
        # self.quantum_corrections_obs = QuantumCorrectionsObservable(self.qg.grid)
    
        # # Galaxy-specific observables
        # self.rotation_curve_obs = GalaxyRotationCurveObservable(self.qg.grid)
        # self.dark_matter_obs = DarkMatterDensityObservable(self.qg.grid)
        # self.velocity_dispersion_obs = VelocityDispersionObservable(self.qg.grid)

        # Use existing observables
        # self.mass_obs = self.qg.physics.ADMMassObservable(self.qg.grid)
        # self.entanglement_obs = self.qg.physics.EntanglementObservable(
        #     self.qg.grid, 
        #     region_A=list(range(int(len(self.qg.grid.points) * 0.5)))  # Use half of points
        # )
        # self.energy_obs = self.qg.physics.EnergyDensityObservable(self.qg.grid)
        # self.quantum_corrections_obs = self.qg.physics.QuantumCorrectionsObservable(self.qg.grid)
        
        # # Galaxy-specific observables
        # self.rotation_curve_obs = GalaxyRotationCurveObservable(self.qg.grid)
        # self.dark_matter_obs = DarkMatterDensityObservable(self.qg.grid)
        # self.velocity_dispersion_obs = VelocityDispersionObservable(self.qg.grid)
    
    def quantum_nfw_profile(self, r_by_rs: np.ndarray) -> np.ndarray:
        """Compute quantum-corrected NFW density profile.
        
        Args:
            r_by_rs: Radius divided by scale radius (r/rs)
            
        Returns:
            Quantum correction factor
        """
        # Mass coupling coefficient based on total mass
        mass_coupling = 0.1 * np.log10(self.total_mass / CONSTANTS['M_sun'])
        
        # Quantum NFW profile: Q(r) = 1 + c(M)exp(-r/(rs×φ))√(Λ/(24×φ))
        quantum_term = mass_coupling * np.exp(-r_by_rs/(self.phi))
        geometric_factor = np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/(24*self.phi))
        
        return 1 + quantum_term * geometric_factor
    
    def compute_rotation_curve(self, radii: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute rotation curve with quantum corrections.
        
        Args:
            radii: Optional array of radii in kpc
            
        Returns:
            Tuple of (radii, velocities) arrays
        """
        if radii is None:
            # Generate logarithmically spaced radii from 0.1% to 120% of galaxy radius
            radii = np.geomspace(self.radius_kpc * 0.001, self.radius_kpc * 1.2, 100)
        
        # Convert to SI
        r_si = radii * 3.086e19  # kpc to meters
        
        # Classical NFW + bulge + disk rotation curve
        v_bulge = np.sqrt(CONSTANTS['G'] * self.initial_stellar_mass * self.bulge_fraction / 
                         (r_si + self.scale_radius * 0.1))  # Bulge contribution
        
        v_disk = np.sqrt(CONSTANTS['G'] * self.initial_stellar_mass * (1 - self.bulge_fraction) / 
                        r_si)  # Disk contribution
        
        # NFW halo contribution
        x = r_si / self.scale_radius
        v_halo = np.sqrt(CONSTANTS['G'] * self.dark_mass / self.scale_radius * 
                        (np.log(1 + x) - x/(1 + x)) / (x * np.log(1 + x)))
        
        # Total classical velocity
        v_classical = np.sqrt(v_bulge**2 + v_disk**2 + v_halo**2)
        
        # Apply quantum corrections
        beta_r = CONSTANTS['l_p'] / r_si
        gamma_eff_r = self.gamma * beta_r * np.sqrt(0.364840)
        quantum_factor = 1 + gamma_eff_r
        
        # Quantum-corrected velocity
        v_quantum = v_classical * np.sqrt(quantum_factor)
        
        # Store for later use
        self.last_rotation_curve = (radii, v_quantum)
        
        return radii, v_quantum
    
    def compute_dark_matter_profile(self, radii: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Compute dark matter density profile with quantum corrections.
        
        Args:
            radii: Optional array of radii in kpc
            
        Returns:
            Tuple of (radii, density) arrays in SI units
        """
        if radii is None:
            # Generate logarithmically spaced radii
            radii = np.geomspace(self.radius_kpc * 0.001, self.radius_kpc * 1.2, 100)
        
        # Convert to SI
        r_si = radii * 3.086e19  # kpc to meters
        
        # Scale radius
        r_s = self.scale_radius
        
        # Classical NFW profile
        rho_s = self.dark_mass / (4 * np.pi * r_s**3)  # Characteristic density
        x = r_si / r_s
        rho_nfw = rho_s / (x * (1 + x)**2)
        
        # Apply quantum corrections
        quantum_factor = self.quantum_nfw_profile(x)
        rho_quantum = rho_nfw * quantum_factor
        
        # Store for later use
        self.last_dm_profile = (radii, rho_quantum)
        
        return radii, rho_quantum
    
    def compute_entanglement_entropy(self) -> float:
        """Compute entanglement entropy with geometric coupling.
        
        Returns:
            Entanglement entropy value
        """
        # Measure entanglement using observable
        entropy_result = self.entanglement_obs.measure(self.qg.state)
        
        # Apply geometric scaling factor
        entropy = entropy_result.value * (1 + self.gamma_eff * np.log(self.radius/CONSTANTS['l_p']))
        
        return entropy
    
    def run_simulation(self, t_final: float, dt: float = None) -> None:
        """Run galaxy simulation for specified time.
        
        Args:
            t_final: Final simulation time (in galactic rotation periods)
            dt: Time step (optional, defaults to 0.01 rotation periods)
        """
        if dt is None:
            dt = 0.01 * self.rotation_period
        
        # Track evolution time
        current_time = 0.0
        step = 0
        next_checkpoint = 0.1 * self.rotation_period
        
        logging.info(f"Starting galaxy simulation for {t_final} rotation periods")
        
        while current_time < t_final * self.rotation_period:
            # Update time
            current_time += dt
            step += 1
            self.qg.state.time = current_time
            
            # Evolve system
            self._evolve_step(dt)
            
            # Check for checkpoint
            if current_time >= next_checkpoint:
                # Verify geometry-entanglement relation
                metrics = self.verifier._verify_geometric_entanglement(self.qg.state)
                self.verification_results.append({
                    'time': current_time / self.rotation_period,
                    'lhs': metrics['lhs'],
                    'rhs': metrics['rhs'],
                    'error': metrics['relative_error']
                })
                
                # Measure key observables
                r, v_rot = self.compute_rotation_curve()
                r_dm, rho_dm = self.compute_dark_matter_profile()
                entropy = self.compute_entanglement_entropy()
                
                # Store history
                self.time_points.append(current_time / self.rotation_period)
                self.rotation_curves.append((r, v_rot))
                self.mass_profiles.append((r_dm, rho_dm))
                self.entropy_history.append(entropy)
                
                # Detailed logging
                logging.info(f"\nTime: {current_time/self.rotation_period:.2f} rotation periods")
                logging.info(f"Geometry-Entanglement Error: {metrics['relative_error']:.6e}")
                logging.info(f"Peak Rotation Velocity: {np.max(v_rot):.1f} m/s")
                logging.info(f"Entanglement Entropy: {entropy:.2e}")
                
                # Verify dark matter relations
                dm_relation = self.dm_verifier.verify_rotation_curve(self.qg.state)
                logging.info(f"Dark Matter Enhancement: {np.mean(dm_relation['enhancement']):.4f}")
                
                # Update next checkpoint
                next_checkpoint += 0.1 * self.rotation_period
                
                # Log progress
                progress = min(100.0, 100.0 * current_time / (t_final * self.rotation_period))
                logging.info(f"Simulation progress: {progress:.1f}%")
    
    def _evolve_step(self, dt: float) -> None:
        """Evolve galaxy by one time step.
        
        Args:
            dt: Time step
        """
        # Update geometric parameters
        self._update_metric()
        
        # Evolve quantum state
        self.qg.state.evolve(dt)
    
    def compute_total_pressure(self) -> float:
        """Compute total effective pressure in galaxy from velocity dispersion.
        
        For galaxies, "pressure" is related to the random motions of stars,
        which is characterized by velocity dispersion.
        
        Returns:
            Effective pressure value in Planck units
        """
        # Use velocity dispersion as basis for pressure calculation
        # P = ρ * σ² where σ is velocity dispersion
        
        # Typical velocity dispersion for this galaxy type
        if self.galaxy_type == 'spiral':
            vel_dispersion = 150.0 * 1000  # 150 km/s in m/s
        elif self.galaxy_type == 'elliptical':
            vel_dispersion = 200.0 * 1000  # 200 km/s in m/s
        elif self.galaxy_type == 'dwarf':
            vel_dispersion = 50.0 * 1000   # 50 km/s in m/s
        else:
            vel_dispersion = 100.0 * 1000  # default
        
        # Average density
        avg_density = self.initial_stellar_mass / (4/3 * np.pi * self.radius**3)
        
        # Compute pressure from density and velocity dispersion
        pressure = avg_density * vel_dispersion**2
        
        # Apply quantum enhancement
        pressure *= self._compute_quantum_factor()
        
        return pressure
    
    def _compute_quantum_factor(self) -> float:
        """Compute overall quantum enhancement factor for galaxy."""
        return 1 + self.gamma_eff * np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/24)
    
    def plot_results(self, save_path: str = None) -> None:
        """Plot comprehensive results of galaxy simulation.
        
        Args:
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(4, 2, figure=fig)
        
        # Title with key parameters
        fig.suptitle(f"Quantum Galaxy Simulation: {self.galaxy_type.capitalize()} Galaxy\n"
                    f"M_stellar = {self.initial_stellar_mass/CONSTANTS['M_sun']:.2e} M☉, "
                    f"R = {self.radius_kpc:.1f} kpc, "
                    f"DM Ratio = {self.dark_matter_ratio:.1f}",
                    fontsize=16, y=0.98)
        
        # Rotation curve
        ax1 = fig.add_subplot(gs[0, 0])
        last_idx = len(self.rotation_curves) - 1
        r, v = self.rotation_curves[last_idx]
        
        # Also plot classical curve for comparison
        x = r / (self.scale_radius/3.086e19)
        v_nfw = np.sqrt(CONSTANTS['G'] * self.dark_mass / self.scale_radius * 
                      (np.log(1 + x) - x/(1 + x)) / (x * np.log(1 + x)))
        v_disk = np.sqrt(CONSTANTS['G'] * self.initial_stellar_mass * (1 - self.bulge_fraction) / 
                        (r * 3.086e19))
        v_class = np.sqrt(v_nfw**2 + v_disk**2)
        
        ax1.plot(r, v/1000, 'b-', label='Quantum-Corrected')
        ax1.plot(r, v_class/1000, 'r--', label='Classical')
        ax1.set_xlabel('Radius [kpc]')
        ax1.set_ylabel('Rotation Velocity [km/s]')
        ax1.set_title('Galaxy Rotation Curve')
        ax1.grid(True)
        ax1.legend()
        
        # Dark matter density profile
        ax2 = fig.add_subplot(gs[0, 1])
        r_dm, rho = self.mass_profiles[last_idx]
        
        # Classical NFW for comparison
        r_s = self.scale_radius
        rho_s = self.dark_mass / (4 * np.pi * r_s**3)
        x = r_dm * 3.086e19 / r_s
        rho_nfw = rho_s / (x * (1 + x)**2)
        
        ax2.loglog(r_dm, rho, 'b-', label='Quantum-Corrected')
        ax2.loglog(r_dm, rho_nfw, 'r--', label='Classical NFW')
        ax2.set_xlabel('Radius [kpc]')
        ax2.set_ylabel('Dark Matter Density [kg/m³]')
        ax2.set_title('Dark Matter Density Profile')
        ax2.grid(True)
        ax2.legend()
        
        # Quantum enhancement factor
        ax3 = fig.add_subplot(gs[1, 0])
        enhancement = v / v_class
        ax3.semilogx(r, enhancement, 'g-')
        ax3.set_xlabel('Radius [kpc]')
        ax3.set_ylabel('Enhancement Factor')
        ax3.set_title('Quantum Gravity Enhancement')
        ax3.grid(True)
        
        # Entanglement entropy evolution
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.time_points, self.entropy_history, 'b-')
        ax4.set_xlabel('Time [rotation periods]')
        ax4.set_ylabel('Entanglement Entropy [k_B]')
        ax4.set_title('Entanglement Entropy Evolution')
        ax4.grid(True)
        
        # Geometry-Entanglement verification
        ax5 = fig.add_subplot(gs[2, 0])
        times = [v['time'] for v in self.verification_results]
        lhs = [v['lhs'] for v in self.verification_results]
        rhs = [v['rhs'] for v in self.verification_results]
        ax5.plot(times, lhs, 'b-', label='LHS (Classical Geometry)')
        ax5.plot(times, rhs, 'r--', label='RHS (Quantum Geometry)')
        ax5.set_xlabel('Time [rotation periods]')
        ax5.set_ylabel('Geometric-Entanglement Terms')
        ax5.set_title('Geometric-Entanglement Verification')
        ax5.grid(True)
        ax5.legend()
        
        # Error evolution
        ax6 = fig.add_subplot(gs[2, 1])
        errors = [v['error'] for v in self.verification_results]
        ax6.semilogy(times, errors, 'k-')
        ax6.set_xlabel('Time [rotation periods]')
        ax6.set_ylabel('Relative Error')
        ax6.set_title('Geometric-Entanglement Error')
        ax6.grid(True)
        
        # Dark matter ratio vs radius
        ax7 = fig.add_subplot(gs[3, 0])
        r_dm, rho_quantum = self.mass_profiles[last_idx]
        
        # Compute enclosed mass
        m_quantum = np.zeros_like(r_dm)
        m_classical = np.zeros_like(r_dm)
        # Replace the mass calculation loop with this improved version
        for i in range(len(r_dm)):
            if i == 0:
                # For the first point, use a simple approximation
                vol = (4/3) * np.pi * (r_dm[0] * 3.086e19)**3
                m_quantum[i] = rho[0] * vol
                m_classical[i] = rho_nfw[0] * vol
            else:
                # For other points, use proper integration
                shells = np.arange(i+1)
                r_shells = r_dm[shells] * 3.086e19
                r_squared = r_shells**2
                
                # Quantum mass
                m_quantum[i] = 4 * np.pi * np.trapezoid(
                    rho[shells] * r_squared,
                    r_shells
                )
                
                # Classical mass
                m_classical[i] = 4 * np.pi * np.trapezoid(
                    rho_nfw[shells] * r_squared,
                    r_shells
                )

        # for i in range(len(r_dm)):
        #     # Quantum mass
        #     m_quantum[i] = 4 * np.pi * np.trapz(
        #         rho * (r_dm[:i+1] * 3.086e19)**2, 
        #         r_dm[:i+1] * 3.086e19
        #     )
            
        #     # Classical mass
        #     m_classical[i] = 4 * np.pi * np.trapz(
        #         rho_nfw[:i+1] * (r_dm[:i+1] * 3.086e19)**2, 
        #         r_dm[:i+1] * 3.086e19
        #     )
        
        # Compute visible mass profile (simple exponential disk)
        r_visible = np.linspace(0, self.radius_kpc, 100)
        disk_scale = self.radius_kpc * 0.2
        rho_visible = self.initial_stellar_mass / (2*np.pi*disk_scale**2) * np.exp(-r_visible/disk_scale)
        
        # Compute dark matter fraction
        ax7.plot(r_dm, m_quantum/m_classical, 'b-', label='Mass Enhancement')
        ax7.plot(r_dm, np.ones_like(r_dm) * self.dark_matter_ratio, 'r--', label='Classical DM Ratio')
        ax7.set_xlabel('Radius [kpc]')
        ax7.set_ylabel('Dark Matter Enhancement')
        ax7.set_title('Quantum Enhanced Dark Matter')
        ax7.grid(True)
        ax7.legend()
        
        # Total mass distribution
        ax8 = fig.add_subplot(gs[3, 1])
        r_tot = np.geomspace(0.01, self.radius_kpc, 100)
        
        # Compute total mass profiles
        m_tot_classical = np.zeros_like(r_tot)
        m_tot_quantum = np.zeros_like(r_tot)
        m_visible = np.zeros_like(r_tot)
        
        # Simple models for illustration
        for i, r in enumerate(r_tot):
            # Visible mass (exponential disk)
            m_visible[i] = self.initial_stellar_mass * (1 - np.exp(-r/disk_scale)*(1 + r/disk_scale))
            
            # Dark matter (NFW)
            x = r * 3.086e19 / self.scale_radius
            m_dm_class = self.dark_mass * (np.log(1 + x) - x/(1 + x)) / (np.log(1 + 10) - 10/11)
            
            # Total classical mass
            m_tot_classical[i] = m_visible[i] + m_dm_class
            
            # Quantum enhancement
            enhancement = 1 + self.gamma_eff * np.exp(-x)
            m_tot_quantum[i] = m_visible[i] + m_dm_class * enhancement
        
        ax8.loglog(r_tot, m_visible/CONSTANTS['M_sun'], 'g-', label='Visible Mass')
        ax8.loglog(r_tot, m_tot_classical/CONSTANTS['M_sun'], 'r--', label='Classical Total')
        ax8.loglog(r_tot, m_tot_quantum/CONSTANTS['M_sun'], 'b-', label='Quantum Total')
        ax8.set_xlabel('Radius [kpc]')
        ax8.set_ylabel('Enclosed Mass [M☉]')
        ax8.set_title('Mass Distribution')
        ax8.grid(True)
        ax8.legend()
        
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logging.info(f"Results saved to {save_path}")
        plt.close()
    
    def plot_galaxy_structure(self, save_path: str = None) -> None:
        """Plot 3D visualization of galaxy structure with quantum effects.
        
        Args:
            save_path: Optional path to save figure
        """
        fig = plt.figure(figsize=(16, 8))
        
        # 3D plot of galaxy structure
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Generate points for visualization
        n_points = 5000
        points = self._generate_galaxy_grid_points(n_points)
        
        # Normalize points to proper scale
        max_extent = np.max(np.abs(points))
        points_scaled = points / max_extent
        
        # Compute quantum effects
        r = np.linalg.norm(points, axis=1)
        beta_r = CONSTANTS['l_p'] / r
        gamma_eff_r = self.gamma * beta_r * np.sqrt(0.364840)
        quantum_factor = 1 + gamma_eff_r * np.exp(-r/self.scale_radius)
        
        # Normalize for colormap
        colors = quantum_factor / np.max(quantum_factor)
        
        # Plot galaxy structure
        scatter = ax1.scatter(
            points_scaled[:, 0], 
            points_scaled[:, 1], 
            points_scaled[:, 2],
            c=colors,
            cmap='viridis',
            alpha=0.8,
            marker='o',
            s=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Quantum Enhancement Factor')
        
        # Set labels
        ax1.set_xlabel('X [normalized]')
        ax1.set_ylabel('Y [normalized]')
        ax1.set_zlabel('Z [normalized]')
        ax1.set_title('Galaxy Structure with Quantum Effects')
        
        # Add 2D projection
        ax2 = fig.add_subplot(122)
        
        # Filter to thin slice in z
        z_cut = 0.1
        disk_slice = (np.abs(points_scaled[:, 2]) < z_cut)
        disk_points = points_scaled[disk_slice]
        disk_colors = colors[disk_slice]
        
        # Plot disk visualization
        scatter2 = ax2.scatter(
            disk_points[:, 0],
            disk_points[:, 1],
            c=disk_colors,
            cmap='viridis',
            alpha=0.8,
            marker='o',
            s=1.0
        )
        
        # Draw reference circles
        theta = np.linspace(0, 2*np.pi, 100)
        # Visible radius
        visible_radius = 0.5
        ax2.plot(visible_radius*np.cos(theta), visible_radius*np.sin(theta), 'r--', 
                label='Visible Radius')
        # Dark matter halo
        halo_radius = 1.0
        ax2.plot(halo_radius*np.cos(theta), halo_radius*np.sin(theta), 'b--', 
                label='Dark Matter Halo')
        
        ax2.set_xlabel('X [normalized]')
        ax2.set_ylabel('Y [normalized]')
        ax2.set_title('Galaxy Disk View (Z-slice)')
        ax2.set_aspect('equal')
        ax2.grid(True)
        ax2.legend()
        
        # Add parameter annotations
        fig.text(0.1, 0.01, f"Galaxy Type: {self.galaxy_type.capitalize()}\n"
                           f"Stellar Mass: {self.initial_stellar_mass/CONSTANTS['M_sun']:.2e} M☉\n"
                           f"Dark Matter Ratio: {self.dark_matter_ratio:.1f}\n"
                           f"β: {self.beta:.2e}, γ_eff: {self.gamma_eff:.2e}")
        
        fig.text(0.6, 0.01, f"Quantum Formulae:\n"
                          f"β = l_p/R\n"
                          f"γ_eff = γβ√(196560/24)\n"
                          f"ρ_DM(r) = ρ_NFW(r) × (1 + γ_eff·e^(-r/rs))")
        
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        
        if save_path:
            plt.savefig(save_path, dpi=150)
            logging.info(f"Galaxy structure plot saved to {save_path}")
        plt.close()


class GalaxyRotationCurveObservable:
    """Observable for measuring galactic rotation curves."""
    
    def __init__(self, grid):
        self.grid = grid
    
    def measure(self, state):
        """Measure rotation curve from current state."""
        # Extract galaxy parameters
        mass = state.total_mass if hasattr(state, 'total_mass') else state.mass
        radius = state.radius if hasattr(state, 'radius') else 10e3 * CONSTANTS['l_p']
        scale_radius = state.scale_radius if hasattr(state, 'scale_radius') else radius * 0.2
        
        # Generate radial points
        radii = np.geomspace(0.001 * radius, 1.2 * radius, 100)
        
        # Compute rotation velocities
        velocities = np.zeros_like(radii)
        
        # NFW + disk model with quantum corrections
        for i, r in enumerate(radii):
            # NFW contribution
            x = r / scale_radius
            v_nfw = np.sqrt(CONSTANTS['G'] * mass * 0.8 / scale_radius * 
                          (np.log(1 + x) - x/(1 + x)) / (x * np.log(1 + x)))
            
            # Disk contribution
            v_disk = np.sqrt(CONSTANTS['G'] * mass * 0.2 / r)
            
            # Combined classical velocity
            v_classical = np.sqrt(v_nfw**2 + v_disk**2)
            
            # Apply quantum corrections
            beta_r = CONSTANTS['l_p'] / r
            gamma_eff_r = 0.55 * beta_r * np.sqrt(0.364840)
            quantum_factor = 1 + gamma_eff_r
            
            # Final velocity
            velocities[i] = v_classical * np.sqrt(quantum_factor)
        
        return MeasurementResult(
            value=(radii, velocities),
            uncertainty=0.05 * velocities,  # 5% uncertainty
            metadata={
                'mass': mass,
                'radius': radius,
                'scale_radius': scale_radius
            }
        )


class DarkMatterDensityObservable:
    """Observable for measuring dark matter density profile."""
    
    def __init__(self, grid):
        self.grid = grid
    
    def measure(self, state):
        """Measure dark matter density profile from current state."""
        # Extract galaxy parameters
        dark_mass = state.dark_mass if hasattr(state, 'dark_mass') else state.mass * 5.0
        radius = state.radius if hasattr(state, 'radius') else 10e3 * CONSTANTS['l_p']
        scale_radius = state.scale_radius if hasattr(state, 'scale_radius') else radius * 0.2
        
        # Generate radial points
        radii = np.geomspace(0.001 * radius, 1.2 * radius, 100)
        
        # Compute densities with NFW profile
        rho_s = dark_mass / (4 * np.pi * scale_radius**3)
        x = radii / scale_radius
        densities = rho_s / (x * (1 + x)**2)
        
        # Apply quantum corrections
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        beta = CONSTANTS['l_p'] / radius
        gamma = 0.55
        gamma_eff = gamma * beta * np.sqrt(0.364840)
        
        # Mass coupling coefficient
        mass_coupling = 0.1 * np.log10(dark_mass / CONSTANTS['M_sun'])
        
        # Quantum correction factor
        quantum_term = mass_coupling * np.exp(-x/phi)
        geometric_factor = np.sqrt(CONSTANTS['LEECH_LATTICE_POINTS']/(24*phi))
        quantum_factor = 1 + quantum_term * geometric_factor
        
        # Apply corrections
        quantum_densities = densities * quantum_factor
        
        return MeasurementResult(
            value=(radii, quantum_densities),
            uncertainty=0.1 * quantum_densities,  # 10% uncertainty
            metadata={
                'dark_mass': dark_mass,
                'scale_radius': scale_radius,
                'gamma_eff': gamma_eff
            }
        )


class VelocityDispersionObservable:
    """Observable for measuring velocity dispersion in galaxies."""
    
    def __init__(self, grid):
        self.grid = grid
    
    def measure(self, state):
        """Measure velocity dispersion from current state."""
        # Extract galaxy parameters
        mass = state.total_mass if hasattr(state, 'total_mass') else state.mass
        radius = state.radius if hasattr(state, 'radius') else 10e3 * CONSTANTS['l_p']
        
        # Generate radial points
        radii = np.geomspace(0.001 * radius, 1.2 * radius, 100)
        
        # Compute velocity dispersion (assuming isotropic)
        # Dispersion is related to circular velocity: σ² ≈ v²/2
        dispersion = np.zeros_like(radii)
        
        # Base dispersion from mass profile
        for i, r in enumerate(radii):
            # Simple model: v² = GM(<r)/r
            v_squared = CONSTANTS['G'] * mass * (r/radius)**0.8 / r
            dispersion[i] = np.sqrt(v_squared / 2)
        
        # Apply quantum corrections
        beta = CONSTANTS['l_p'] / radius
        gamma = 0.55
        gamma_eff = gamma * beta * np.sqrt(0.364840)
        quantum_factor = 1 + gamma_eff * np.exp(-radii/radius)
        
        # Final dispersion
        quantum_dispersion = dispersion * np.sqrt(quantum_factor)
        
        return MeasurementResult(
            value=(radii, quantum_dispersion),
            uncertainty=0.1 * quantum_dispersion,  # 10% uncertainty
            metadata={
                'mass': mass,
                'radius': radius,
                'gamma_eff': gamma_eff
            }
        )


def main():
    """Run galaxy simulation example."""
    # Configure logging
    configure_logging(simulation_type='galaxy')
    
    # Create output directory
    output_dir = Path("results/galaxy")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run simulations for different galaxy types
    galaxies = [
        {'type': 'spiral', 'mass': 5e10, 'radius': 15.0, 'dm_ratio': 5.0},
        {'type': 'elliptical', 'mass': 1e11, 'radius': 20.0, 'dm_ratio': 7.0},
        {'type': 'dwarf', 'mass': 1e9, 'radius': 5.0, 'dm_ratio': 10.0}
    ]
    
    # Create single quantum gravity instance to share
    qg = QuantumGravity()
    
    for galaxy in galaxies:
        logging.info(f"\nSimulating {galaxy['type']} galaxy")
        
        # Create galaxy simulation
        sim = GalaxySimulation(
            stellar_mass=galaxy['mass'],
            radius=galaxy['radius'],
            galaxy_type=galaxy['type'],
            dark_matter_ratio=galaxy['dm_ratio'],
            quantum_gravity=qg
        )
        
        # Run simulation
        t_final = 1.0  # one rotation period
        sim.run_simulation(t_final)
        
        # Generate plots
        sim.plot_results(str(output_dir / f"galaxy_{galaxy['type']}_evolution.png"))
        sim.plot_galaxy_structure(str(output_dir / f"galaxy_{galaxy['type']}_structure.png"))
        
        # Create measurement results
        measurements = []
        
        # Rotation curves
        for i, (r, v) in enumerate(sim.rotation_curves):
            measurements.append(
                MeasurementResult(
                    value={'radii': r.tolist(), 'velocities': v.tolist()},
                    uncertainty=None,
                    metadata={
                        'time': sim.time_points[i],
                        'type': galaxy['type'],
                        'stellar_mass': galaxy['mass'],
                        'radius': galaxy['radius'],
                        'dark_matter_ratio': galaxy['dm_ratio']
                    }
                )
            )
        
        # Save measurements
        io = QuantumGravityIO(str(output_dir))
        io.save_measurements(measurements, f"galaxy_{galaxy['type']}_measurements")
    
    logging.info("\nAll galaxy simulations completed")


if __name__ == "__main__":
    main()
