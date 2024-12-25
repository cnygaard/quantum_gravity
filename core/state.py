# core/state.py
from typing import Dict, Tuple, Optional
from .grid import AdaptiveGrid
import numpy as np
from scipy.sparse import csr_matrix

class QuantumState:
    """Efficient representation of quantum gravitational states."""
    
    def __init__(self, 
                 grid: AdaptiveGrid,
                 eps_cut: float = 1e-10):
        self.grid = grid
        self.eps_cut = eps_cut
        self.coefficients = {}  # Sparse representation
        self.basis_states = {}
        self.metric_components = {}  # Initialize metric components dictionary here

    def set_metric_component(self, 
                           indices: Tuple[int, int],
                           point_idx: int,
                           value: float) -> None:
        """Set metric component g_{μν} at specified point."""
        if indices not in self.metric_components:
            self.metric_components[indices] = {}
            
        self.metric_components[indices][point_idx] = value
        
        # Set symmetric component if off-diagonal
        if indices[0] != indices[1]:
            sym_indices = (indices[1], indices[0])
            if sym_indices not in self.metric_components:
                self.metric_components[sym_indices] = {}
            self.metric_components[sym_indices][point_idx] = value

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
        # Basic implementation for now
        new_coeffs = {}
        for idx, coeff in self.coefficients.items():
            if abs(coeff) > self.eps_cut:
                new_coeffs[idx] = coeff
        self.coefficients = new_coeffs

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
