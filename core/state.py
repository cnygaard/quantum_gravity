# core/state.py
from typing import Dict, Tuple, Optional
from .grid import AdaptiveGrid
import numpy as np
from scipy.sparse import csr_matrix
from constants import CONSTANTS

class QuantumState:
    """Efficient representation of quantum gravitational states."""
    def __init__(self, grid: AdaptiveGrid, initial_mass: float, eps_cut: float = 1e-10, simulation=None):
        # Core quantum state properties
        self.simulation = simulation  # Store simulation reference
        self.grid = grid
        self.eps_cut = eps_cut
        self.coefficients = {}
        self.basis_states = {}
        self._metric_array = np.zeros((4, 4, len(self.grid.points)))
        self.time = 0.0

        # Black hole properties
        if initial_mass <= 0:
            raise ValueError("Initial mass must be positive")
        self.mass = initial_mass
        self.initial_mass = initial_mass

        self.evaporation_rate = (CONSTANTS['hbar'] * CONSTANTS['c']**6) / (15360 * np.pi * CONSTANTS['G']**2)
        self.evaporation_timescale = self.mass**3 / self.evaporation_rate

        # Quantum information metrics
        self.entanglement = 0.0
        self.information = 0.0

        # Thermodynamic properties
        self._entropy = None
        self._temperature = None
        self._last_update = 0.0

        # Cosmological properties
        self.scale_factor = 1.0
        self.energy_density = 0.0
        self.equation_of_state = -1.0
        self.hubble_parameter = 0.
        
        self.velocity = np.zeros(len(self.grid.points))  # Initialize velocity field

    def compute_velocity(self) -> np.ndarray:
        """Compute velocity field with numerical stability"""
        points = self.grid.points
        r = np.linalg.norm(points, axis=1)
        
        # Scale to natural units
        r_scaled = r / CONSTANTS['l_p']
        mass_scaled = self.mass / CONSTANTS['m_p']
        
        # Compute scaled velocity
        v = np.sqrt(2 * CONSTANTS['G'] * mass_scaled / np.maximum(r_scaled, 1e-10))
        v = np.minimum(v, 0.99 * CONSTANTS['c'])  # Enforce subluminal speeds
        
        # Safe relativistic correction
        beta = v / CONSTANTS['c']
        gamma = 1 / np.sqrt(np.maximum(1e-10, 1 - beta**2))
        
        self.velocity = v * gamma
        return self.velocity


    @property
    def entropy(self) -> float:
        """Get black hole entropy using Bekenstein-Hawking formula."""
        if self._entropy is None or self.time > self._last_update:
            self._update_thermodynamics()
        return self._entropy

    @property
    def temperature(self) -> float:
        """Get black hole temperature using Hawking formula."""
        if self._temperature is None or self.time > self._last_update:
            self._update_thermodynamics()
        return self._temperature

    def _update_thermodynamics(self) -> None:
        """Update thermodynamic properties."""
        # Calculate horizon properties
        horizon_radius = 2 * CONSTANTS['G'] * self.mass
        area = 4 * np.pi * horizon_radius**2
        
        # Bekenstein-Hawking entropy
        self._entropy = area / (4 * CONSTANTS['l_p']**2)
        
        # Hawking temperature
        self._temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.mass)
        
        self._last_update = self.time

    def evolve(self, dt: float) -> None:
        """Evolve state by time step dt."""
        self.time += dt
        
        # Update mass with Hawking radiation
        self.mass = self.initial_mass * (1 - self.time/self.evaporation_timescale)**(1/3)
        self.mass = max(self.mass, CONSTANTS['m_p'])  # Enforce Planck mass cutoff
        
        # Update quantum information metrics
        self.compute_entanglement()
        self.compute_information()
        
        # Clear cached thermodynamics
        self._entropy = None
        self._temperature = None

    def compute_entanglement(self) -> float:
        """Compute entanglement measure."""
        entanglement = 0.0
        for i, coeff_i in self.coefficients.items():
            for j, coeff_j in self.coefficients.items():
                if i != j:
                    entanglement += abs(coeff_i * coeff_j.conjugate())
        self.entanglement = entanglement
        return entanglement

    def compute_information(self) -> float:
        """Compute von Neumann entropy as information measure."""
        information = 0.0
        for coeff in self.coefficients.values():
            p = abs(coeff)**2
            if p > self.eps_cut:
                information -= p * np.log(p)
        self.information = information
        return information

    def compute_temperature_profile(self):
        """Compute temperature profile with quantum corrections."""
        # Create temperature profile object
        class TempProfile:
            def __init__(self, core, surface):
                self.core = core
                self.surface = surface
        
        # Get radial coordinates
        r = np.linalg.norm(self.grid.points, axis=1)
        r_norm = r / np.max(r)
        
        # Core temperature from virial theorem
        T_core = 1.57e7  # Solar core temperature
        
        # Surface temperature from Stefan-Boltzmann
        T_surface = 5778  # Solar surface temperature
        
        # Apply quantum corrections
        quantum_factor = self.simulation._compute_quantum_factor()
        T_core *= quantum_factor
        T_surface *= quantum_factor
        
        return TempProfile(core=T_core, surface=T_surface)

    def compute_total_pressure(self) -> float:
        """Calculate total pressure including quantum effects"""
        G = CONSTANTS['G']
        M = self.mass
        R = self.grid.get_max_radius()
        
        # Base gravitational pressure
        P_classical = (3 * G * M**2) / (8 * np.pi * R**4)
        
        # Apply quantum corrections
        quantum_factor = self.simulation._compute_quantum_factor()
        P_total = P_classical * quantum_factor
        
        return P_total

    def compute_gravitational_pressure(self) -> float:
        """Calculate gravitational pressure without quantum corrections."""
        G = CONSTANTS['G']
        M = self.mass
        R = self.grid.get_max_radius()
        
        # Base gravitational pressure from hydrostatic equilibrium
        P_grav = (3 * G * M**2) / (8 * np.pi * R**4)
        
        return P_grav


    def set_metric_components_batch(self, indices_list, points, values):
        # Use numpy vectorized operations instead of extend
        indices_array = np.array(indices_list)
        points_array = np.array(points)
        values_array = np.array(values)
        self._metric_array[indices_array[:,0], indices_array[:,1], points_array] = values_array

    def add_basis_state(self, 
                       index: int, 
                       coeff: complex,
                       state_vector: np.ndarray) -> None:
        """Add a basis state if coefficient exceeds cutoff."""
        if abs(coeff) > self.eps_cut:
            self.coefficients[index] = coeff
            self.basis_states[index] = state_vector

    def expectation_value(self, operator: csr_matrix) -> float:
        """Compute expectation value of operator."""
        state_vector = np.zeros(len(self.grid.points), dtype=complex)
        for i, coeff in self.coefficients.items():
            state_vector[i] = coeff
        return np.real(state_vector.conjugate() @ operator @ state_vector)

    def get_metric_component(self, indices: Tuple[int, int], point_idx: int) -> float:
        """Get metric component value at specified indices and point."""
        return self._metric_array[indices[0], indices[1], point_idx]

    def set_metric_component(self, indices: Tuple[int, int], point_idx: int, value: float) -> None:
        """Set metric component value at specified indices and point."""
        self._metric_array[indices[0], indices[1], point_idx] = value

class CosmologicalState(QuantumState):
    """Quantum state for cosmological simulations."""
    
    def __init__(self, 
                 grid: 'AdaptiveGrid',
                 initial_scale: float,
                 hubble_parameter: float):
        """Initialize cosmological state.
        
        Args:
            grid: Spatial grid
            initial_scale: Initial scale factor a(t)
            hubble_parameter: Initial Hubble parameter H(t)
        """
        super().__init__(grid, initial_mass=1.0)  # Mass not relevant for cosmology
        self.initial_scale = initial_scale        # Initial scale factor 
        self.scale_factor = initial_scale
        self.hubble_parameter = hubble_parameter
        self.time = 0.0
        
        # Initialize cosmological observables
        self.energy_density = 3 * hubble_parameter**2 / (8 * np.pi * CONSTANTS['G'])
        self.pressure = -self.energy_density  # Vacuum dominated

    def _compute_slow_roll(self) -> float:
        """Compute slow-roll parameter epsilon."""
        # Inflation potential parameters
        m = 1e-6  # Mass parameter in Planck units
        
        # Compute slow-roll parameter ε = (V'/V)²/2
        V = 0.5 * m * m * self.phi * self.phi  # Potential
        V_prime = m * m * self.phi  # dV/dφ
        
        epsilon = 0.5 * (V_prime/V)**2 if V > 0 else 0.0
        
        return epsilon


    def _evolve_inflation_field(self, dt: float) -> float:
        """Evolve inflation field with proper potential."""
        # Inflation potential parameters
        m = 1e-6  # Mass parameter in Planck units
        
        # Current field value
        if not hasattr(self, 'phi'):
            self.phi = 3.0  # Initial field value
            self.phi_dot = 0.0  # Initial field velocity
        
        # Update field with chaotic inflation potential V = m²φ²/2
        V_prime = m * m * self.phi  # dV/dφ
        self.phi_dot -= (3 * self.hubble_parameter * self.phi_dot + V_prime) * dt
        self.phi += self.phi_dot * dt
        
        return self.phi

    def evolve(self, dt: float) -> None:
        # Inflation field dynamics
        self.phi = self._evolve_inflation_field(dt)
        
        # Track slow-roll parameters
        self.epsilon = self._compute_slow_roll()
        
        # Update scale factor with inflation
        self.scale_factor *= np.exp(self.hubble_parameter * dt)
