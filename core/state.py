# core/state.py
from typing import Dict, Tuple, Optional
from .grid import AdaptiveGrid
import numpy as np
from scipy.sparse import csr_matrix

class QuantumState:
    """Efficient representation of quantum gravitational states."""
    def __init__(self, grid: AdaptiveGrid, eps_cut: float = 1e-10):
        self.grid = grid
        self.eps_cut = eps_cut
        self.coefficients = {}
        self.basis_states = {}
        self._metric_array = np.zeros((4, 4, len(self.grid.points)))
        self.time = 0.0
        self.mass = 1000.0  # Initial mass in Planck units

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

    def evolve(self, dt: float) -> None:
        """Evolve state by time step dt."""
        # Update time
        self.time += dt
        
        # Calculate evaporation timescale
        evaporation_rate = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2)
        self.evaporation_timescale = self.initial_mass**3 / evaporation_rate
        
        # Update mass with proper Hawking radiation evolution
        # M(t) = M₀(1 - t/τ)^(1/3)
        self.mass = self.initial_mass * (1 - self.time/self.evaporation_timescale)**(1/3)
        self.mass = max(self.mass, CONSTANTS['m_p'])
        
        # Update temperature (T ∝ 1/M)
        self.temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * self.mass)
        
        # Update entropy (S = A/4)
        self.entropy = 4 * np.pi * (2 * CONSTANTS['G'] * self.mass)**2 / (4 * CONSTANTS['l_p']**2)
        
        # Update radiation flux (F ∝ 1/M^2)
        self.radiation_flux = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2 * self.mass**2)




    def expectation_value(self, operator: csr_matrix) -> float:
        """Compute expectation value of operator in current state.
        
        Args:
            operator: Sparse matrix representing the observable operator
            
        Returns:
            float: Expectation value <ψ|O|ψ>
        """
        # Convert state coefficients to vector form
        state_vector = np.zeros(len(self.grid.points), dtype=complex)
        for i, coeff in self.coefficients.items():
            state_vector[i] = coeff
            
        # Calculate <ψ|O|ψ>
        return np.real(state_vector.conjugate() @ operator @ state_vector)
