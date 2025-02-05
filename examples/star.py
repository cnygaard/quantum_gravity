#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from constants import CONSTANTS, SI_UNITS
from core.state import QuantumState
from core.grid import LeechLattice
from physics.verification import UnifiedTheoryVerification
from __init__ import QuantumGravity, configure_logging
import logging
from utils.io import MeasurementResult  # Add this import
from concurrent.futures import ThreadPoolExecutor
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

class StarSimulation:
    """Quantum star simulation extending black hole framework."""
    def __init__(self, mass: float, radius: float, galaxy_radius: float = None, quantum_gravity=None, debug=False):
        """Initialize star simulation with galaxy-scale effects."""
        # Initialize verification tracking
        self.verification_results = []

        self.mass = mass  # In solar masses
        self.radius = radius  # In solar radii
    
        # Convert to Planck units with proper scaling
        self.M_star = mass * CONSTANTS['M_sun']  
        self.R_star = radius * CONSTANTS['R_sun']

        # Add Hubble parameter initialization
        self.hubble_parameter = 70.0 * 1000 / (3.086e22)  # H0 in Planck units
    
        self.qg = quantum_gravity or QuantumGravity()
        self.debug = debug

        # Initialize quantum parameters with proper scaling
        self.gamma = 0.55  # Coupling constant
        self.beta = CONSTANTS['l_p'] / self.R_star  # Quantum scale parameter
        self.gamma_eff = self.gamma * self.beta * np.sqrt(0.407)  # Effective coupling

        # Galaxy scale quantum parameters
        self.galaxy_radius = galaxy_radius or 50000 * CONSTANTS['R_sun']  # Default ~50 kpc
        self.beta_galaxy = CONSTANTS['l_p'] / self.galaxy_radius
        self.gamma_eff_galaxy = self.gamma * self.beta_galaxy * np.sqrt(0.407)
    
        # Quantum vacuum parameters
        self.rho_vacuum = CONSTANTS['hbar'] / (CONSTANTS['c'] * CONSTANTS['l_p']**4)
        self.beta_universe = CONSTANTS['l_p'] / CONSTANTS['c'] * self.hubble_parameter
        self.rho_vacuum_modified = self.rho_vacuum * (1 + self.gamma_eff * self.beta_universe)

        # Add Leech lattice vacuum energy calculations
        self.leech = LeechLattice(points=CONSTANTS['LEECH_LATTICE_POINTS'])

        # Compute initial vacuum energy and lambda
        self.vacuum_energy = self._compute_leech_vacuum_energy()
        self.cosmological_constant = self._compute_modified_lambda()

        self.verifier = UnifiedTheoryVerification(self)


        # Setup grid and state first
        self._setup_grid()
        self._setup_observables()
        self._initialize_profile_arrays()
        self._setup_initial_state()

        # Initialize arrays 
        self._initialize_profile_arrays()

        # Initialize tracking variables
        self.current_size = 0
        self.time_points = []
        self.quantum_corrections = []

        self.next_checkpoint = 0.0

        # Add history tracking
        self.central_density_history = []
        self.central_pressure_history = []
        self.core_temperature_history = []
        self.quantum_corrections_history = []
        self.vacuum_energy_history = []

    def _setup_grid(self):
        points = self._generate_grid_points()
        logging.debug(f"Points stats: min={np.min(points):.3e}, max={np.max(points):.3e}")
        if not np.all(np.isfinite(points)):
            invalid_points = points[~np.isfinite(points)]
            logging.error(f"Invalid points: {invalid_points}")
            raise ValueError("NaN/Inf found in grid points.")
        self.qg.grid.set_points(points)

    def _initialize_profile_arrays(self):
        """Initialize profile arrays with proper dimensions."""
        n_points = len(self.qg.grid.points)
        initial_capacity = 1000  # Initial capacity for time steps

    def _setup_observables(self) -> None:
        """Setup observables including galaxy-scale effects."""
        # Existing observable setup...
    
        # Add galaxy-scale measurements
        self.effective_force = self._compute_effective_force()
        self.dark_matter_ratio = self.effective_force / self._compute_classical_force()
    
        # Track vacuum energy
        self.vacuum_energy_density = self.rho_vacuum_modified
        self.cosmological_constant = 8 * np.pi * CONSTANTS['G'] * self.rho_vacuum_modified

        """Setup observables for stellar measurements."""
        self.mass_obs = self.qg.physics.ADMMassObservable(self.qg.grid)
        self.radius_obs = self.qg.physics.AreaObservable(self.qg.grid)
        self.temp_obs = self.qq.physics.StellarTemperatureObservable(self.qg.grid)  # Direct initialization
        selfg.density_obs = self.qg.physics.EnergyDensityObservable(self.qg.grid)
        self.pressure_obs = self.qg.physics.PressureObservable(self.qg.grid)


    def _compute_effective_force(self) -> float:
        """Compute force with quantum gravity corrections."""
        return CONSTANTS['G'] * self.M_star * (1 + self.gamma_eff_galaxy * self.beta_galaxy) / self.galaxy_radius**2

    def _compute_classical_force(self) -> float:
        """Compute classical gravitational force."""
        return CONSTANTS['G'] * self.M_star / self.galaxy_radius**2

    def _compute_central_pressure(self):
        """Compute realistic central pressure"""
        G = CONSTANTS['G']
        M = self.M_star
        R = self.R_star
        
        # Basic stellar pressure with quantum corrections
        P_classical = (3 * G * M**2) / (8 * np.pi * R**4)
        P_quantum = P_classical * (1 + self.gamma_eff)
        
        return P_quantum

    def _compute_leech_vacuum_energy(self) -> float:
            """Compute vacuum energy with Leech lattice corrections"""
            base_energy = CONSTANTS['hbar']/(CONSTANTS['c'] * CONSTANTS['l_p']**4)
            leech_correction = self.leech.compute_vacuum_energy()
            quantum_factor = self._compute_quantum_factor()
            # Combine quantum corrections with Leech lattice effects
            return base_energy * quantum_factor * leech_correction
            #return base_energy * (1 + self.gamma_eff * self.beta_universe) * leech_correction

    def _compute_modified_lambda(self) -> float:
        """Calculate modified cosmological constant"""
        return 8 * np.pi * CONSTANTS['G'] * self.rho_vacuum / CONSTANTS['c']**2

    def _track_energy_conservation(self):
        """Track total energy conservation"""
        kinetic = self.qg.physics.compute_kinetic_energy(self.qg.state)
        potential = self.qg.physics.compute_potential_energy(self.qg.state)
        quantum = self.qg.physics.compute_quantum_energy(self.qg.state)
        return kinetic + potential + quantum

    def _initialize_profile_arrays(self):
        """Initialize profile arrays with proper dimensions."""
        n_points = len(self.qg.grid.points)
        initial_capacity = 1000  # Initial capacity for time steps
        
        # Initialize profile arrays with proper shape
        self.density_profile = np.zeros((initial_capacity, n_points))
        self.pressure_profile = np.zeros((initial_capacity, n_points))
        self.temperature_profile = np.zeros((initial_capacity, n_points))
        self.array_capacity = initial_capacity

    def _generate_grid_points(self):
        """Generate grid points for stellar simulation with proper scaling."""
        max_points = 50000
        points_per_dim = int(np.cbrt(max_points/4))
        
        # Convert solar radius to proper units and set scale
        #r_min = max(CONSTANTS['l_p'], self.R_star * 1e-6)  # Start from max of Planck length or small fraction of radius
        #r_max = self.R_star / CONSTANTS['l_p']  # Normalize to Planck length
        scale_factor = CONSTANTS['l_p']
        r_min = max(1.0, (self.R_star/scale_factor) * 1e-6)
        r_max = self.R_star/scale_factor 

        # Rescale to manageable numbers
        scale = np.log10(r_max)
        if scale > 30:
            r_min /= 10**scale
            r_max /= 10**scale
    
        r = np.geomspace(r_min, r_max, points_per_dim) 
                
        points = []
        total_points = 0
        
        # Generate points with spherical symmetry
        for r_i in r:
            # More angular points near equator
            n_theta = max(4, int(points_per_dim * np.sin(np.pi/2)))
            theta_samples = np.linspace(0, np.pi, n_theta)
            
            for theta_i in theta_samples:
                # Adjust phi points based on theta
                n_phi = max(4, int(n_theta * np.sin(theta_i)))
                phi_samples = np.linspace(0, 2*np.pi, n_phi)
                
                sin_theta = np.sin(theta_i)
                cos_theta = np.cos(theta_i)
                
                for phi_i in phi_samples:
                    if total_points >= max_points:
                        break
                    
                    # Convert to Cartesian coordinates
                    x = r_i * sin_theta * np.cos(phi_i)
                    y = r_i * sin_theta * np.sin(phi_i)
                    z = r_i * cos_theta
                    
                    # Add point if within proper range
                    if np.all(np.abs([x, y, z]) < 1e30):
                        points.append([x, y, z])
                        total_points += 1
                    
            if total_points >= max_points:
                break
        
        points = np.array(points)
        
        if len(points) < 100:
            raise ValueError(f"Too few valid points generated ({len(points)}). Minimum required: 100")
        
        # Final validation
        if not np.all(np.isfinite(points)):
            raise ValueError("Invalid points generated")
            
        if np.any(np.abs(points) > 1e30):
            raise ValueError("Points exceed maximum allowed magnitude")
        
        if self.debug:
            logging.debug(f"Generated {len(points)} grid points")
            logging.debug(f"Radial range: [{np.min(np.linalg.norm(points, axis=1)):.2e}, {np.max(np.linalg.norm(points, axis=1)):.2e}]")
        
        return points

    def _setup_stellar_structure(self) -> None:
        """Setup stellar structure equations with quantum corrections."""
        def stellar_structure(r, y):
            m, P = y
            
            # Prevent division by zero
            r = np.maximum(r, CONSTANTS['l_p'])
            m = np.maximum(m, CONSTANTS['m_p'])
            

            # Enhanced core density with original EOS
            core_density = self.M_star / (4/3 * np.pi * (0.1*self.R_star)**3)
            base_rho = np.maximum(self._equation_of_state(P), CONSTANTS['m_p']/(4*np.pi*r**3))
            rho = np.maximum(base_rho, core_density * np.exp(-r/self.R_star))
            
            # Set minimum central pressure based on virial theorem
            P_min = CONSTANTS['G'] * self.M_star**2 / (8 * np.pi * self.R_star**4)
            P = np.maximum(P, P_min)
            rho_quantum = rho * (1 + self.gamma_eff*self.beta)
            
            # Mass evolution
            dm_dr = 4*np.pi*r**2 * rho_quantum
            
            # Full relativistic pressure evolution
            dP_dr = -CONSTANTS['G']*m*rho_quantum/(r**2) * \
                    (1 + P_quantum/(rho_quantum*CONSTANTS['c']**2)) * \
                    (1 + 4*np.pi*r**3*P_quantum/(m*CONSTANTS['c']**2)) * \
                    (1 - 2*CONSTANTS['G']*m/(r*CONSTANTS['c']**2))**(-1)
            
            return [dm_dr, dP_dr]

    def _equation_of_state(self, P: float) -> float:
        """Simple polytropic equation of state."""
        gamma = 5/3  # Ideal gas
        K = 1e-5  # EOS constant
        return (P/K)**(1/gamma)

    def _setup_initial_state(self) -> None:
        """Setup initial quantum state."""
        state = QuantumState(
            self.qg.grid,
            initial_mass=self.M_star,
            eps_cut=self.qg.config.config['numerics']['eps_cut']
        )

        # Add stellar properties
        state.R_star = self.R_star
        state.T_surface = 5778  # K
        state.T_core = 1.57e7  # K

        # Initialize metric
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
        r = np.maximum(r, CONSTANTS['l_p'])
        
        # Schwarzschild metric outside star
        g_tt = -(1 - 2*CONSTANTS['G']*self.M_star/r)
        g_rr = 1/(1 - 2*CONSTANTS['G']*self.M_star/r)
        
        state.set_metric_components_batch(
            [(0,0)]*len(r) + [(1,1)]*len(r),
            list(range(len(r)))*2,
            np.concatenate([g_tt, g_rr])
        )
        
        self.qg.state = state

    def _setup_observables(self) -> None:
        """Setup observables for stellar measurements."""
        self.mass_obs = self.qg.physics.ADMMassObservable(self.qg.grid)
        self.radius_obs = self.qg.physics.AreaObservable(self.qg.grid)
        self.temp_obs = self.qg.physics.StellarTemperatureObservable(self.qg.grid)
        self.density_obs = self.qg.physics.EnergyDensityObservable(self.qg.grid)
        self.pressure_obs = self.qg.physics.PressureObservable(self.qg.grid)
        
    def run_simulation(self, t_final: float) -> None:
        """Run simulation with controlled time evolution and detailed logging"""
        self.next_checkpoint = 0.1
        checkpoint_interval = 0.1
        
        while self.qg.state.time < t_final:
            dt = self._compute_timestep()
            self.qg.state.evolve(dt)
            
            # Get measurements
            density_result = self._measure_density_profile()
            pressure_result = self._measure_pressure_profile()
            temp_result = self._measure_temperature_profile()

            # Track time points along with measurements
            self.time_points.append(self.qg.state.time)  # Add this line

            # Extract numeric values from MeasurementResults
            density_value = np.asarray(density_result.value)
            pressure_value = pressure_result.value[0].value if isinstance(pressure_result.value, np.ndarray) else pressure_result.value
            temp_value = np.asarray(temp_result.value)
            
            # Store numeric values in profile arrays
            if self.current_size >= len(self.density_profile):
                self._resize_profile_arrays()
                
            self.density_profile[self.current_size] = density_value
            self.pressure_profile[self.current_size] = pressure_value
            self.temperature_profile[self.current_size] = temp_value
            
            self.current_size += 1

            # Record data
            self.central_density_history.append(np.max(density_value))
            self.central_pressure_history.append(np.max(pressure_value))
            self.core_temperature_history.append(np.mean(temp_value))
            self.quantum_corrections_history.append(self.gamma_eff)
            self.vacuum_energy_history.append(self.vacuum_energy)

            # Check if we've reached checkpoint
            if self.qg.state.time >= self.next_checkpoint:
                # Update vacuum energy calculations
                self.vacuum_energy = self._compute_leech_vacuum_energy()
                self.cosmological_constant = self._compute_modified_lambda()


                if self.qg.state.time >= self.next_checkpoint:
                    metrics = self.verifier._verify_geometric_entanglement(self.qg.state)
                    # Pass metrics to normalization function
                    normalized_scales = self._normalize_geometric_scales(metrics)
                self.verification_results.append(metrics)


                # Log vacuum energy metrics
                logging.info(f"\nVacuum Energy: {self.vacuum_energy:.2e}")
                logging.info(f"Cosmological Constant: {self.cosmological_constant:.2e}")

                # Log stellar structure
                logging.info(f"\nStellar Structure at t={self.qg.state.time:.2f}:")
                logging.info(f"Mass: {self.M_star/CONSTANTS['M_sun']:.2e} M_sun")
                logging.info(f"Radius: {self.R_star/CONSTANTS['R_sun']:.2e} R_sun")
                logging.info(f"Central Density: {np.max(density_value):.26e}")
                logging.info(f"Central Pressure: {np.max(pressure_value):.26e}")
                logging.info(f"Core Temperature: {np.mean(temp_value):.26e}")

                # Log geometric verification
                logging.info(f"\nGeometric-Entanglement Formula:")
                logging.info(f"LHS = {metrics['lhs']:.44e}")
                logging.info(f"RHS = {metrics['rhs']:.44e}")
                logging.info(f"LHS (normalized) = {normalized_scales['lhs_normalized']:.44e}")
                logging.info(f"RHS (normalized) = {normalized_scales['rhs_normalized']:.44e}")
                logging.info(f"Relative Error = {metrics['relative_error']:.6e}")

                # Log quantum parameters
                logging.info(f"\nQuantum Parameters:")
                logging.info(f"β (l_p/R): {self.beta:.2e}")
                logging.info(f"γ_eff: {self.gamma_eff:.2e}")

                # Log progress
                progress = min((self.qg.state.time/t_final) * 100, 100.0)
                logging.info(f"\nSimulation progress: {progress:.1f}%")

                # Update checkpoint
                self.next_checkpoint += checkpoint_interval

            self.current_size += 1


    def _measure_profile(self, observable, name: str) -> MeasurementResult:
        """Base method for measuring physical profiles with consistent handling.
        
        Args:
            observable: Physics observable instance
            name: Name of the measurement for logging
        """
        try:
            result = observable.measure(self.qg.state)
            
            # Ensure we have array output
            if np.isscalar(result.value):
                value = np.full(len(self.qg.grid.points), result.value)
                uncertainty = np.full(len(self.qg.grid.points), result.uncertainty)
            else:
                value = np.asarray(result.value)
                uncertainty = np.asarray(result.uncertainty)
            
            return MeasurementResult(
                value=value,
                uncertainty=uncertainty,
                metadata={'time': self.qg.state.time}
            )
                
        except Exception as e:
            logging.error(f"Error measuring {name}: {e}")
            n_points = len(self.qg.grid.points)
            return MeasurementResult(
                value=np.zeros(n_points),
                uncertainty=np.zeros(n_points),
                metadata={'time': self.qg.state.time, 'error': str(e)}
            )

    def _measure_density_profile(self) -> MeasurementResult:
        result = self._measure_profile(self.density_obs, "density")
        
        # Use solar core density as minimum
        min_density = 1.62e5  # kg/m³
        density_value = np.maximum(result.value, min_density)
        
        return MeasurementResult(
            value=density_value,
            uncertainty=result.uncertainty,
            metadata=result.metadata
        )

        
    def _measure_pressure_profile(self) -> MeasurementResult:
        return self._measure_profile(self.pressure_obs, "pressure")
        
    def _measure_temperature_profile(self) -> MeasurementResult:
        return self._measure_profile(self.temp_obs, "temperature")

    def _compute_surface_temperature(self):
        """Calculate surface temperature with proper scaling"""
        # Include radiation pressure and quantum effects
        T_classical = (G * M * m_p / (k_B * R))**(1/4)
        T_quantum = T_classical * (1 + self.beta * self.gamma_eff)
        return T_quantum




    def _parallel_profile_measurement(self):
        """Measure profiles in parallel"""
        with ThreadPoolExecutor() as executor:
            density_future = executor.submit(self._measure_density_profile)
            pressure_future = executor.submit(self._measure_pressure_profile)
            temp_future = executor.submit(self._measure_temperature_profile)
            return (density_future.result(), pressure_future.result(), 
                    temp_future.result())

    def _normalize_geometric_scales(self, metrics):
        # Use physical scale normalization
        planck_scale = CONSTANTS['l_p']
        stellar_scale = self.R_star
        
        scale_factor = np.sqrt(planck_scale * stellar_scale)
        
        normalized = {
            'lhs_normalized': metrics['lhs'] / scale_factor,
            'rhs_normalized': metrics['rhs'] / scale_factor,
            'scale_factor': scale_factor
        }
        
        return normalized

    def _evolve_step(self, dt: float) -> None:
        """Evolve system one timestep with full parameter updates."""
        # Setup and evolve stellar structure
        self._setup_stellar_structure()
        
        # Track quantum geometric evolution
        self.geometric_coupling = self._compute_geometric_coupling()
        self.vacuum_fluctuations = self._compute_vacuum_fluctuations()
        
        # Update metric and spacetime geometry
        self._update_metric()
        
        # Evolve entanglement parameters
        self.entanglement_entropy = self._compute_entanglement()
        
        # Update time
        self.qg.state.time += dt

    def _compute_timestep(self):
        """Compute adaptive timestep considering all relevant scales"""
        # Update velocity field
        self.qg.state.compute_velocity()

        # Get geometric coupling from verifier
        geometric_coupling = self.verifier._compute_geometric_coupling(self.qg.state)
        
        # Calculate characteristic timescales
        t_dynamic = CONSTANTS['c'] * self.R_star / (CONSTANTS['G'] * self.M_star)
        t_quantum = CONSTANTS['hbar'] / (self.gamma_eff * CONSTANTS['c']**2)
        #t_entangle = CONSTANTS['hbar'] / self.geometric_coupling
        
        # Use smallest timescale with safety factor
        #dt = min(t_dynamic, t_quantum, t_entangle) * 0.01
        dt = min(t_dynamic, t_quantum) * 0.01

        # Enforce maximum timestep
        max_dt = 0.01  # Maximum timestep of 0.01 time units
        return min(dt, max_dt)

    def _update_metric(self) -> None:
        """Update metric with quantum corrections."""
        points = self.qg.grid.points
        r = np.maximum(np.linalg.norm(points, axis=1), CONSTANTS['l_p'])
        
        g_tt = -(1 - 2*CONSTANTS['G']*self.M_star/r) * (1 + self.gamma_eff*self.beta)
        g_rr = 1/(1 - 2*CONSTANTS['G']*self.M_star/r) * (1 + self.gamma_eff*self.beta)
        
        n_points = len(points)
        metric_indices = np.array([(0,0)] * n_points + [(1,1)] * n_points)
        point_indices = np.arange(n_points).repeat(2)
        values = np.concatenate([g_tt, g_rr])
        
        if self.qg.state._metric_array.shape[2] != n_points:
            self.qg.state._metric_array = np.zeros((4, 4, n_points))
        
        self.qg.state.set_metric_components_batch(
            metric_indices,
            point_indices,
            values
        )

    def plot_results(self, save_path: str = None) -> None:
        """Plot comprehensive stellar evolution."""
        from matplotlib.gridspec import GridSpec
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(15, 20))
        gs = GridSpec(4, 2, figure=fig)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.r_points/self.R_star, self.density_profile[-1])
        ax1.set_xlabel('r/R_star')
        ax1.set_ylabel('Density [ρ_P]')
        ax1.set_yscale('log')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.r_points/self.R_star, self.pressure_profile[-1])
        ax2.set_xlabel('r/R_star')
        ax2.set_ylabel('Pressure [P_P]')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.r_points/self.R_star, self.temperature_profile[-1])
        ax3.set_xlabel('r/R_star')
        ax3.set_ylabel('Temperature [T_P]')
        ax3.grid(True)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.time_points, [qc['gamma_eff'] for qc in self.quantum_corrections])
        ax4.set_xlabel('Time [t_P]')
        ax4.set_ylabel('γ_eff')
        ax4.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        plt.close()

    def _resize_profile_arrays(self):
        """Resize profile arrays when capacity is reached."""
        new_capacity = self.array_capacity * 2
        self.density_profile.resize((new_capacity, self.density_profile.shape[1]), refcheck=False)
        self.pressure_profile.resize((new_capacity, self.pressure_profile.shape[1]), refcheck=False)
        self.temperature_profile.resize((new_capacity, self.temperature_profile.shape[1]), refcheck=False)
        self.array_capacity = new_capacity

    def _trim_profile_arrays(self):
        """Trim profile arrays to actual size used."""
        self.density_profile = self.density_profile[:self.current_size]
        self.pressure_profile = self.pressure_profile[:self.current_size]
        self.temperature_profile = self.temperature_profile[:self.current_size]

    def _log_simulation_status(self, density):
        """Log current simulation status."""
        logging.info(f"\nStellar Structure at t={self.qg.state.time:.2f}:")
        logging.info(f"Mass: {self.M_star/CONSTANTS['M_sun']:.2e} M_sun")
        logging.info(f"Radius: {self.R_star/CONSTANTS['R_sun']:.2e} R_sun")
        logging.info(f"Central Density: {np.max(density):.2e}")
        logging.info(f"Quantum Parameter γ_eff: {self.gamma_eff:.2e}")

    def _measure_pressure_profile(self) -> MeasurementResult:
        """Measure current pressure profile with proper error handling."""
        try:
            result = self.pressure_obs.measure(self.qg.state)
            
            if isinstance(result, MeasurementResult):
                if isinstance(result.value, (float, int)):
                    # If scalar, expand to array
                    value = np.full(len(self.qg.grid.points), result.value)
                    uncertainty = np.full(len(self.qg.grid.points), result.uncertainty)
                    return MeasurementResult(value=value, uncertainty=uncertainty, metadata=result.metadata)
                return result
            else:
                # Handle raw value
                value = result if isinstance(result, np.ndarray) else np.full(len(self.qg.grid.points), result)
                return MeasurementResult(
                    value=value,
                    uncertainty=np.zeros_like(value),
                    metadata={'time': self.qg.state.time}
                )
                
        except Exception as e:
            logging.error(f"Error measuring pressure: {e}")
            n_points = len(self.qg.grid.points)
            return MeasurementResult(
                value=np.zeros(n_points),
                uncertainty=np.zeros(n_points),
                metadata={'time': self.qg.state.time, 'error': str(e)}
            )

    def plot_results(self, save_path: str = None) -> None:
        fig = plt.figure(figsize=(15, 24))
        gs = GridSpec(5, 2, figure=fig)

        # Get verification time points
        #verification_times = [t for t in self.time_points if t % checkpoint_interval == 0]

        # Core Properties Evolution
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.time_points, self.central_density_history)
        ax1.set_yscale('log')
        ax1.set_xlabel('Time [t_P]')
        ax1.set_ylabel('Central Density [ρ_P]')
        ax1.grid(True)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.time_points, self.central_pressure_history)
        ax2.set_yscale('log')
        ax2.set_xlabel('Time [t_P]')
        ax2.set_ylabel('Central Pressure [P_P]')
        ax2.grid(True)
        
        # Temperature and Quantum Effects
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(self.time_points, self.core_temperature_history)
        ax3.set_xlabel('Time [t_P]')
        ax3.set_ylabel('Core Temperature [T_P]')
        ax3.grid(True)
        
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(self.time_points, self.quantum_corrections_history)
        ax4.set_xlabel('Time [t_P]')
        ax4.set_ylabel('Quantum Correction Factor')
        ax4.grid(True)
        
        # Geometric-Entanglement Verification
        # ax5 = fig.add_subplot(gs[2, 0])
        # ax5.plot(verification_times, [v['lhs'] for v in self.verification_results], label='LHS')
        # ax5.plot(verification_times, [v['rhs'] for v in self.verification_results], label='RHS')
        # #ax5.plot(self.time_points, [v['lhs'] for v in self.verification_results], label='LHS')
        # #ax5.plot(self.time_points, [v['rhs'] for v in self.verification_results], label='RHS')
        # ax5.set_yscale('log')
        # ax5.set_xlabel('Time [t_P]')
        # ax5.set_ylabel('Geometric-Entanglement Terms')
        # ax5.legend()
        # ax5.grid(True)
        
        # Leech Lattice Energy
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(self.time_points, self.vacuum_energy_history)
        ax6.set_xlabel('Time [t_P]')
        ax6.set_ylabel('Vacuum Energy [E_P]')
        ax6.grid(True)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_star_geometry(self, save_path: str = None):
        fig = plt.figure(figsize=(15, 12))

    # Add main title with key parameters
        #central_pressure = np.max(self.pressure_profile[-1])
        #central_pressure = np.max(self.pressure_profile[-1])
        valid_pressures = self.pressure_profile[self.pressure_profile != 0]
        central_pressure = valid_pressures[0] if len(valid_pressures) > 0 else 0
        core_temperature = np.mean(self.temperature_profile[-1])
        fig.suptitle(f'Quantum Star Structure (M={self.M_star/CONSTANTS["M_sun"]:.1f} M☉, R={self.R_star/CONSTANTS["R_sun"]:.1f} R☉)\n' + 
                    f'T_core={self.qg.state.temperature:.2e} K, P_c={central_pressure:.2e} Pa', 
                    fontsize=14)
        


        ax1 = fig.add_subplot(121, projection='3d')

        # Add quantum parameters text box
        ax1.text2D(0.05, 0.95, 
                f'Quantum Parameters:\n' +
                f'β (l_p/R): {self.beta:.2e}\n' +
                f'γ_eff: {self.gamma_eff:.2e}\n' +
                f'Vacuum Energy: {self.vacuum_energy:.2e}',
                transform=ax1.transAxes,
                bbox=dict(facecolor='white', alpha=0.8))

        # Generate proper grid size
        n_points = 50  # Define grid resolution
        theta = np.linspace(0, np.pi, n_points)
        phi = np.linspace(0, 2*np.pi, n_points)
        
        # Normalize radius to solar radius
        R_normalized = self.R_star/CONSTANTS["R_sun"]
    
        X, Y, Z = self._generate_surface_grid(R_normalized, theta, phi)
        #X, Y, Z = self._generate_surface_grid(self.R_star, theta, phi)
        quantum_density = self._compute_quantum_density()
        
        # Interpolate quantum density onto visualization grid
        r_grid = np.sqrt(X**2 + Y**2 + Z**2)
        density_grid = np.interp(r_grid.flatten(), 
                                np.linalg.norm(self.qg.grid.points, axis=1),
                                quantum_density).reshape(X.shape)
        
        surf = ax1.plot_surface(X, Y, Z, 
                            facecolors=plt.cm.plasma(density_grid),
                            alpha=0.8)
        
        if save_path:
            plt.savefig(save_path)


    def _compute_quantum_density(self) -> np.ndarray:
        """Compute quantum density distribution using existing central density."""
        
        # Use existing central density
        central_density = 1.62e5  # Known solar core density
        
        # Get normalized radial coordinates
        points = self.qg.grid.points
        r = np.linalg.norm(points, axis=1)
        r_norm = r / self.R_star
        
        # Density profile with quantum corrections
        density = central_density * np.exp(-r_norm**2)
        
        # Add quantum effects
        quantum_factor = 1 + self.gamma_eff * (CONSTANTS['l_p']/r)**CONSTANTS['LEECH_LATTICE_DIMENSION']
        density *= quantum_factor
        
        # Normalize for visualization
        return density / central_density

    def _compute_quantum_factor(self):
        """Compute quantum geometric enhancement factor"""
        r_natural = self.radius * SI_UNITS['ly_si'] / (CONSTANTS['R_sun'] * SI_UNITS['R_sun_si'])
        m_natural = self.mass / CONSTANTS['M_sun']
        
        # Leech lattice geometric factors
        dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        points = CONSTANTS['LEECH_LATTICE_POINTS']
        lattice_factor = np.sqrt(points/dimension)
        
        # Normalized quantum enhancement
        scale_factor = np.exp(-r_natural/1e4)
        quantum_enhancement = scale_factor * lattice_factor * (m_natural)**0.25
        
        return 1.0 + 0.1 * np.tanh(quantum_enhancement * 1e-6)


    def _generate_surface_grid(self, r: float, theta: np.ndarray, phi: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate 3D surface grid for stellar visualization.
        
        Args:
            r: Stellar radius
            theta: Array of theta angles
            phi: Array of phi angles
            
        Returns:
            Tuple of (X, Y, Z) coordinate meshgrids
        """
        # Create meshgrid for spherical coordinates
        THETA, PHI = np.meshgrid(theta, phi)
        
        # Convert to Cartesian coordinates
        X = r * np.sin(THETA) * np.cos(PHI)
        Y = r * np.sin(THETA) * np.sin(PHI)
        Z = r * np.cos(THETA)
        
        # Apply quantum corrections to surface
        quantum_factor = 1 + self.gamma_eff * self.beta
        X *= quantum_factor
        Y *= quantum_factor
        Z *= quantum_factor
        
        return X, Y, Z


def main():
    """Run star simulation example."""
    configure_logging(simulation_type='star_simulation')
    sim = StarSimulation(mass=1.0, radius=1.0)
    sim.run_simulation(t_final=5.0)

    # Add plotting with proper output paths
    output_dir = Path("results/star")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot evolution results
    sim.plot_results(str(output_dir / "star_evolution.png"))
    sim.plot_star_geometry(str(output_dir / "star_geometry.png"))

if __name__ == "__main__":
    main()