import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from abc import ABC, abstractmethod
from constants import CONSTANTS

@dataclass
class OperatorResult:
    """Result of operator application."""
    coefficient: complex
    state: np.ndarray
    metadata: Optional[Dict] = None

class QuantumOperator:
    """Base class for quantum operators."""
    
    def __init__(self, grid: 'AdaptiveGrid', operator_type: str, **kwargs):
        self.grid = grid
        self.operator_type = operator_type
        self.kwargs = kwargs
        self._matrix = None
        
    @property
    def matrix(self) -> csr_matrix:
        """Get operator matrix, computing if necessary."""
        if self._matrix is None:
            self._matrix = self._construct_matrix()
        return self._matrix
    
    def _construct_matrix(self) -> csr_matrix:
        """Construct operator matrix based on type."""
        constructors = {
            'hamiltonian': self._construct_hamiltonian_matrix,
            'momentum': self._construct_momentum_matrix,
            'angular_momentum': self._construct_angular_momentum_matrix,
            'constraint': self._construct_constraint_matrix
        }
        return constructors[self.operator_type]()
    
    def _construct_hamiltonian_matrix(self) -> csr_matrix:
        """Construct Hamiltonian operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        G = self.kwargs.get('coupling_constant', CONSTANTS['G'])
        
        for i in range(n_points):
            # Diagonal terms (kinetic + potential)
            r = np.linalg.norm(self.grid.points[i])
            energy = -CONSTANTS['hbar']**2/(2*r**2) - G/r
            rows.append(i)
            cols.append(i)
            data.append(energy)
            
            # Off-diagonal kinetic terms
            for j in self.grid.neighbors[i]:
                if j > i:
                    coupling = -CONSTANTS['hbar']**2/(2*r**2)
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])
                    
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _construct_momentum_matrix(self) -> csr_matrix:
        """Construct momentum operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            for j in self.grid.neighbors[i]:
                dx = self.grid.points[j] - self.grid.points[i]
                p = -1j * CONSTANTS['hbar'] / np.linalg.norm(dx)
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([p, -p])
                
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _construct_angular_momentum_matrix(self) -> csr_matrix:
        """Construct angular momentum operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            r = self.grid.points[i]
            for j in self.grid.neighbors[i]:
                dr = self.grid.points[j] - r
                L = np.cross(r, dr) * (-1j * CONSTANTS['hbar'])
                rows.extend([i, j])
                cols.extend([j, i])
                data.extend([L[2], -L[2]])  # Using z-component
                
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _construct_constraint_matrix(self) -> csr_matrix:
        """Construct constraint operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        constraint_idx = self.kwargs.get('constraint_index', 0)
        
        for i in range(n_points):
            # Implement constraint based on index
            constraint = self._compute_constraint(i, constraint_idx)
            rows.append(i)
            cols.append(i)
            data.append(constraint)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _compute_constraint(self, point_idx: int, constraint_idx: int) -> float:
        """Compute constraint value for given point and constraint index."""
        x = self.grid.points[point_idx]
        if constraint_idx == 0:
            # Hamiltonian constraint
            return np.sum(x**2) / (2 * CONSTANTS['l_p']**2)
        else:
            # Momentum constraints
            return x[constraint_idx-1] / CONSTANTS['l_p']
    
    def apply(self, state: np.ndarray, *args, **kwargs) -> OperatorResult:
        """Apply operator to state."""
        result = self.matrix @ state
        return OperatorResult(
            coefficient=1.0,
            state=result
        )
    
    def expectation_value(self, state: np.ndarray) -> complex:
        """Compute expectation value."""
        return state.conjugate() @ (self.matrix @ state)