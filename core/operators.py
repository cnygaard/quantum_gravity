# core/operators.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from dataclasses import dataclass
from abc import ABC, abstractmethod

@dataclass
class OperatorResult:
    """Result of operator application."""
    coefficient: complex
    state: np.ndarray
    metadata: Optional[Dict] = None

class QuantumOperator(ABC):
    """Base class for quantum operators."""
    
    def __init__(self, grid: 'AdaptiveGrid'):
        self.grid = grid
        self._matrix: Optional[csr_matrix] = None
        
    @property
    def matrix(self) -> csr_matrix:
        """Get operator matrix, computing if necessary."""
        if self._matrix is None:
            self._matrix = self._construct_matrix()
        return self._matrix
    
    @abstractmethod
    def _construct_matrix(self) -> csr_matrix:
        """Construct operator matrix."""
        pass
    
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

class MetricOperator(QuantumOperator):
    """Metric operator g_μν."""
    
    def __init__(self, grid: 'AdaptiveGrid', indices: Tuple[int, int]):
        super().__init__(grid)
        self.mu, self.nu = indices
        
    def _construct_matrix(self) -> csr_matrix:
        """Construct metric operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Diagonal terms
            rows.append(i)
            cols.append(i)
            data.append(self._compute_metric_component(i))
            
            # Off-diagonal terms for nearest neighbors
            for j in self.grid.neighbors[i]:
                if j > i:  # Avoid double counting
                    coupling = self._compute_metric_coupling(i, j)
                    if abs(coupling) > 1e-10:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([coupling, coupling.conjugate()])
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _compute_metric_component(self, i: int) -> float:
        """Compute metric component at point i."""
        x = self.grid.points[i]
        # Basic flat metric plus quantum corrections
        if self.mu == self.nu:
            return 1.0 + self._quantum_correction(x)
        return self._quantum_correction(x)
    
    def _compute_metric_coupling(self, i: int, j: int) -> complex:
        """Compute metric coupling between points."""
        dx = self.grid.points[j] - self.grid.points[i]
        return 1j * self._quantum_correction(dx)
    
    def _quantum_correction(self, x: np.ndarray) -> float:
        """Compute quantum corrections to metric."""
        return (np.sum(x**2) / (self.grid.l_p**2)) * 1e-10

class MomentumOperator(QuantumOperator):
    """Momentum operator p_μ."""
    
    def __init__(self, grid: 'AdaptiveGrid', component: int):
        super().__init__(grid)
        self.component = component
        
    def _construct_matrix(self) -> csr_matrix:
        """Construct momentum operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            for j in self.grid.neighbors[i]:
                dx = self.grid.points[j] - self.grid.points[i]
                if dx[self.component] > 0:  # Forward difference
                    coeff = -1j * ℏ / dx[self.component]
                    rows.extend([i, i])
                    cols.extend([j, i])
                    data.extend([coeff, -coeff])
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

class HamiltonianOperator(QuantumOperator):
    """Full Hamiltonian operator."""
    
    def __init__(self, 
                 grid: 'AdaptiveGrid',
                 metric: Dict[Tuple[int, int], MetricOperator],
                 momentum: List[MomentumOperator]):
        super().__init__(grid)
        self.metric = metric
        self.momentum = momentum
        
    def _construct_matrix(self) -> csr_matrix:
        """Construct Hamiltonian operator matrix."""
        n_points = len(self.grid.points)
        
        # Kinetic term
        kinetic = self._construct_kinetic_term()
        
        # Potential term
        potential = self._construct_potential_term()
        
        # Quantum corrections
        quantum = self._construct_quantum_term()
        
        return kinetic + potential + quantum
    
    def _construct_kinetic_term(self) -> csr_matrix:
        """Construct kinetic energy term."""
        result = 0
        for mu in range(3):
            for nu in range(3):
                metric_inv = sparse_linalg.inv(self.metric[(mu, nu)].matrix)
                result += metric_inv @ self.momentum[mu].matrix @ \
                         self.momentum[nu].matrix
        return result / (2.0)
    
    def _construct_potential_term(self) -> csr_matrix:
        """Construct potential energy term."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(self._compute_potential(self.grid.points[i]))
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _construct_quantum_term(self) -> csr_matrix:
        """Construct quantum correction term."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(self._compute_quantum_correction(self.grid.points[i]))
            
            for j in self.grid.neighbors[i]:
                if j > i:
                    coupling = self._compute_quantum_coupling(i, j)
                    if abs(coupling) > 1e-10:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([coupling, coupling.conjugate()])
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _compute_potential(self, x: np.ndarray) -> float:
        """Compute classical potential."""
        r = np.linalg.norm(x)
        if r < self.grid.l_p:
            return 1e10  # Strong repulsion at small scales
        return -1.0 / r  # Gravitational potential
    
    def _compute_quantum_correction(self, x: np.ndarray) -> float:
        """Compute local quantum corrections."""
        r = np.linalg.norm(x)
        return ℏ**2 / (2.0 * r**3)  # Quantum potential
    
    def _compute_quantum_coupling(self, i: int, j: int) -> complex:
        """Compute quantum coupling between points."""
        dx = self.grid.points[j] - self.grid.points[i]
        r = np.linalg.norm(dx)
        return 1j * ℏ / (r * self.grid.l_p)

class ConstraintOperator(QuantumOperator):
    """Constraint operator for physical states."""
    
    def __init__(self, 
                 grid: 'AdaptiveGrid',
                 constraint_type: str):
        super().__init__(grid)
        self.constraint_type = constraint_type
        
    def _construct_matrix(self) -> csr_matrix:
        """Construct constraint operator matrix."""
        if self.constraint_type == 'hamiltonian':
            return self._construct_hamiltonian_constraint()
        elif self.constraint_type == 'momentum':
            return self._construct_momentum_constraint()
        elif self.constraint_type == 'gauss':
            return self._construct_gauss_constraint()
        else:
            raise ValueError(f"Unknown constraint type: {self.constraint_type}")
    
    def _construct_hamiltonian_constraint(self) -> csr_matrix:
        """Construct Hamiltonian constraint H|ψ⟩ = 0."""
        # Similar to HamiltonianOperator but without time evolution
        return self._construct_kinetic_term() + \
               self._construct_potential_term()
    
    def _construct_momentum_constraint(self) -> csr_matrix:
        """Construct momentum constraint P|ψ⟩ = 0."""
        result = 0
        for mu in range(3):
            result += self.momentum[mu].matrix @ self.momentum[mu].matrix
        return result
    
    def _construct_gauss_constraint(self) -> csr_matrix:
        """Construct Gauss constraint ∇·E|ψ⟩ = 0."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            divergence = self._compute_divergence(i)
            rows.append(i)
            cols.append(i)
            data.append(divergence)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _compute_divergence(self, i: int) -> float:
        """Compute divergence at point i."""
        div = 0.0
        for j in self.grid.neighbors[i]:
            dx = self.grid.points[j] - self.grid.points[i]
            r = np.linalg.norm(dx)
            div += 1.0 / r**2
        return div