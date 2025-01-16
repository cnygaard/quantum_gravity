#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from typing import Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp
from constants import CONSTANTS
from core.state import QuantumState
from core.grid import LeechLattice
from physics.verification import UnifiedTheoryVerification
from __init__ import QuantumGravity, configure_logging
import logging
from utils.io import MeasurementResult  # Add this import

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
        self.leech = LeechLattice(points=100000)

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

    def _compute_effective_force(self) -> float:
        """Compute force with quantum gravity corrections."""
        return CONSTANTS['G'] * self.M_star * (1 + self.gamma_eff_galaxy * self.beta_galaxy) / self.galaxy_radius**2

    def _compute_classical_force(self) -> float:
        """Compute classical gravitational force."""
        return CONSTANTS['G'] * self.M_star / self.galaxy_radius**2

    # def _compute_leech_vacuum_energy(self) -> float:
    #     """Compute vacuum energy with Leech lattice corrections"""
    #     # Get base vacuum energy from Leech lattice
    #     leech_energy = self.leech.compute_vacuum_energy()
        
    #     # Apply quantum corrections
    #     quantum_factor = 1 + self.gamma_eff * self.beta_universe
        
    #     # Combine effects
    #     return leech_energy * quantum_factor
    def _compute_leech_vacuum_energy(self) -> float:
            """Compute vacuum energy with Leech lattice corrections"""
            base_energy = CONSTANTS['hbar']/(CONSTANTS['c'] * CONSTANTS['l_p']**4)
            leech_correction = self.leech.compute_vacuum_energy()
            
            # Combine quantum corrections with Leech lattice effects
            return base_energy * (1 + self.gamma_eff * self.beta_universe) * leech_correction

    def _compute_modified_lambda(self) -> float:
        """Calculate modified cosmological constant"""
        return 8 * np.pi * CONSTANTS['G'] * self.rho_vacuum / CONSTANTS['c']**2

    # def _compute_modified_lambda(self) -> float:
    #     """Calculate modified cosmological constant"""
    #     vacuum_energy = self._compute_leech_vacuum_energy()
        
    #     # Convert to cosmological constant
    #     lambda_modified = 8 * np.pi * CONSTANTS['G'] * vacuum_energy / CONSTANTS['c']**2
        
    #     # Track for verification
    #     self.verification_results.append({
    #         'time': self.qg.state.time,
    #         'lambda': lambda_modified,
    #         'vacuum_energy': vacuum_energy
    #     })
        
    #     return lambda_modified

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
        r_min = max(CONSTANTS['l_p'], self.R_star * 1e-6)  # Start from max of Planck length or small fraction of radius
        r_max = self.R_star / CONSTANTS['l_p']  # Normalize to Planck length
        
        # Use log spacing with more points near center
        r = np.geomspace(r_min, r_max, points_per_dim)
        
        # Scale check to prevent numerical issues
        if np.any(r > 1e30) or np.any(r < 1e-30):
            logging.warning(f"Radial coordinates may be poorly scaled: range [{r_min:.2e}, {r_max:.2e}]")
            # Rescale if necessary
            scale_factor = r_max / 1e10  # Target a reasonable maximum
            r = r / scale_factor
        
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
            rho = np.maximum(self._equation_of_state(P), CONSTANTS['m_p']/(4*np.pi*r**3))
            
            # Quantum corrections
            P_quantum = P * (1 + self.gamma_eff*self.beta)
            rho_quantum = rho * (1 + self.gamma_eff*self.beta)
            
            # Mass evolution
            dm_dr = 4*np.pi*r**2 * rho_quantum
            
            # Pressure evolution with regularization
            dP_dr = -CONSTANTS['G']*m*rho_quantum/(r**2) * \
                    (1 + P_quantum/(rho_quantum*CONSTANTS['c']**2)) * \
                    (1 + 4*np.pi*r**3*P_quantum/(m*CONSTANTS['c']**2)) * \
                    (1 - 2*CONSTANTS['G']*m/(r*CONSTANTS['c']**2))**(-1)
            
            return [dm_dr, dP_dr]
        
        # Initial conditions at center
        P_c = 1e15
        r_span = [1e-10, self.R_star]
        y0 = [0, P_c]
        
        # Solve structure equations
        sol = solve_ivp(stellar_structure, r_span, y0, 
                       method='RK45', rtol=1e-8, atol=1e-8)
        
        # Store profiles
        self.r_points = sol.t
        self.mass_profile = sol.y[0]
        self.pressure_profile = sol.y[1]
        self.density_profile = self._equation_of_state(self.pressure_profile)

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
        """Run simulation until t_final."""
        try:
            while self.qg.state.time < t_final:
                self.qg.state.evolve(0.01)
                metrics = self.verifier._verify_geometric_entanglement(self.qg.state)

                # Get measurements directly
                density_result = self._measure_density_profile()
                pressure_result = self._measure_pressure_profile()
                temp_result = self._measure_temperature_profile()

                # Extract numerical values
                density_value = np.asarray(density_result.value)
                pressure_value = pressure_result.value
                pressure_value = np.array([result.value for result in pressure_result.value])
                temp_value = np.asarray(temp_result.value)
                
                # Check and resize arrays if needed BEFORE storing new values
                if self.current_size >= len(self.density_profile):
                    self._resize_profile_arrays()
                
                # Store in profile arrays
                self.density_profile[self.current_size] = density_value
                self.pressure_profile[self.current_size] = pressure_value
                self.temperature_profile[self.current_size] = temp_value

                # Update vacuum energy less frequently
                if int(self.qg.state.time * 100) % 10 == 0:  # Every 0.1 time units
                    self.vacuum_energy = self._compute_leech_vacuum_energy()
                    self.cosmological_constant = self._compute_modified_lambda()

                # Update vacuum energy and lambda each timestep
                #self.vacuum_energy = self._compute_leech_vacuum_energy()
                #self.cosmological_constant = self._compute_modified_lambda()
                
                # Log vacuum energy evolution
                logging.info(f"\nVacuum Energy: {self.vacuum_energy:.2e}")
                logging.info(f"Cosmological Constant: {self.cosmological_constant:.2e}")            

                # Update counters and check array size
                self.current_size += 1



                # Log comprehensive physics output
                logging.info(f"\nStellar Structure at t={self.qg.state.time:.2f}:")
                logging.info(f"Mass: {self.M_star/CONSTANTS['M_sun']:.2e} M_sun")
                logging.info(f"Radius: {self.R_star/CONSTANTS['R_sun']:.2e} R_sun")
                logging.info(f"Central Density: {np.max(density_result.value):.2e}")
                logging.info(f"Central Pressure: {np.max(pressure_value):.2e}")
                logging.info(f"Surface Temperature: {np.mean(temp_result.value):.2e}")

                # Log geometric verification
                logging.info(f"\nGeometric-Entanglement Formula:")
                logging.info(f"LHS = {metrics['lhs']:.44e}")
                logging.info(f"RHS = {metrics['rhs']:.44e}")
                logging.info(f"Relative Error = {metrics['relative_error']:.6e}")
                
                # Log quantum parameters
                logging.info(f"\nQuantum Parameters:")
                logging.info(f"β (l_p/R): {self.beta:.2e}")
                logging.info(f"γ_eff: {self.gamma_eff:.2e}")
                
                # Log simulation progress
                logging.info(f"\nSimulation progress: {(self.qg.state.time/t_final)*100:.1f}%")

        except Exception as e:
            print(f"Error in simulation step: {str(e)}")
            print(f"Current state: time={self.qg.state.time}, size={self.current_size}")
            raise

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

    # Then use it for specific measurements:
    def _measure_density_profile(self) -> MeasurementResult:
        return self._measure_profile(self.density_obs, "density")
        
    def _measure_pressure_profile(self) -> MeasurementResult:
        return self._measure_profile(self.pressure_obs, "pressure")
        
    def _measure_temperature_profile(self) -> MeasurementResult:
        return self._measure_profile(self.temp_obs, "temperature")


    # def _measure_density_profile(self) -> MeasurementResult:
    #     """Measure current density profile with single MeasurementResult."""
    #     try:
    #         result = self.density_obs.measure(self.qg.state)
        
    #         # Ensure we have array output
    #         if np.isscalar(result.value):
    #             value = np.full(len(self.qg.grid.points), result.value)
    #             uncertainty = np.full(len(self.qg.grid.points), result.uncertainty)
    #         else:
    #             value = np.asarray(result.value)
    #             uncertainty = np.asarray(result.uncertainty)
            
    #         return MeasurementResult(
    #             value=value,
    #             uncertainty=uncertainty,
    #             metadata={'time': self.qg.state.time}
    #         )
                
    #     except Exception as e:
    #         logging.error(f"Error measuring density: {e}")
    #         n_points = len(self.qg.grid.points)
    #         return MeasurementResult(
    #             value=np.zeros(n_points),
    #             uncertainty=np.zeros(n_points),
    #             metadata={'time': self.qg.state.time, 'error': str(e)}
    #         )

    # def _measure_pressure_profile(self) -> MeasurementResult:
    #     """Measure current pressure profile with proper error handling."""
    #     try:
    #         result = self.pressure_obs.measure(self.qg.state)
            
    #         if isinstance(result, MeasurementResult):
    #             if isinstance(result.value, (float, int)):
    #                 # If scalar, expand to array
    #                 value = np.full(len(self.qg.grid.points), result.value)
    #                 uncertainty = np.full(len(self.qg.grid.points), result.uncertainty)
    #                 return MeasurementResult(value=value, uncertainty=uncertainty, metadata=result.metadata)
    #             return result
    #         else:
    #             # Handle raw value
    #             value = result if isinstance(result, np.ndarray) else np.full(len(self.qg.grid.points), result)
    #             return MeasurementResult(
    #                 value=value,
    #                 uncertainty=np.zeros_like(value),
    #                 metadata={'time': self.qg.state.time}
    #             )
                
    #     except Exception as e:
    #         logging.error(f"Error measuring pressure: {e}")
    #         n_points = len(self.qg.grid.points)
    #         return MeasurementResult(
    #             value=np.zeros(n_points),
    #             uncertainty=np.zeros(n_points),
    #             metadata={'time': self.qg.state.time, 'error': str(e)}
    #         )

    # def _measure_temperature_profile(self) -> MeasurementResult:
    #     """Measure current temperature profile with proper error handling."""
    #     try:
    #         result = self.temp_obs.measure(self.qg.state)
            
    #         if isinstance(result, MeasurementResult):
    #             if np.isscalar(result.value):
    #                 value = np.full(len(self.qg.grid.points), result.value)
    #                 uncertainty = np.full(len(self.qg.grid.points), result.uncertainty)
    #                 return MeasurementResult(value=value, uncertainty=uncertainty, metadata=result.metadata)
    #             return result
    #         else:
    #             value = np.asarray(result)
    #             if np.isscalar(value):
    #                 value = np.full(len(self.qg.grid.points), value)
    #             return MeasurementResult(
    #                 value=value,
    #                 uncertainty=np.zeros_like(value),
    #                 metadata={'time': self.qg.state.time}
    #             )
                
    #     except Exception as e:
    #         logging.error(f"Error measuring temperature: {e}")
    #         n_points = len(self.qg.grid.points)
    #         return MeasurementResult(
    #             value=np.zeros(n_points),
    #             uncertainty=np.zeros(n_points),
    #             metadata={'time': self.qg.state.time, 'error': str(e)}
    #         )

    def _evolve_step(self, dt: float) -> None:
        """Evolve system one timestep."""
        self._setup_stellar_structure()
        self.qg.state.time += dt
        self._update_metric()

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

def main():
    """Run star simulation example."""
    configure_logging(simulation_type='star_simulation')
    sim = StarSimulation(mass=1.0, radius=1.0)
    sim.run_simulation(t_final=10.0)

if __name__ == "__main__":
    main()