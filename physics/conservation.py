# physics/conservation.py
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.sparse import csr_matrix
from dataclasses import dataclass

@dataclass
class ConservationQuantities:
    """Container for conserved quantities."""
    energy: float
    momentum: np.ndarray
    angular_momentum: np.ndarray
    constraints: np.ndarray

class ConservationLawTracker:
    """Track and verify conservation laws in quantum gravity simulation."""
    
    def __init__(self, 
                 grid: 'AdaptiveGrid',
                 tolerance: float = 1e-10):
        self.grid = grid
        self.tolerance = tolerance
        self.initial_quantities: Optional[ConservationQuantities] = None
        self.history: List[ConservationQuantities] = []
        
    def compute_quantities(self, 
                         state: 'QuantumState',
                         operators: Dict[str, 'QuantumOperator']) -> ConservationQuantities:
        """Compute all conserved quantities for current state."""
        # Energy conservation
        energy = self._compute_energy(state, operators['hamiltonian'])
        
        # Linear momentum conservation
        momentum = self._compute_momentum(state, operators['momentum'])
        
        # Angular momentum conservation
        angular_momentum = self._compute_angular_momentum(
            state, operators['angular_momentum']
        )
        
        # Constraint violations
        constraints = self._compute_constraints(
            state, operators['constraints']
        )
        
        return ConservationQuantities(
            energy=energy,
            momentum=momentum,
            angular_momentum=angular_momentum,
            constraints=constraints
        )
    
    def check_conservation(self, 
                         current: ConservationQuantities) -> Dict[str, float]:
        """Check conservation law violations."""
        if self.initial_quantities is None:
            self.initial_quantities = current
            return {'energy': 0.0, 'momentum': 0.0, 
                   'angular_momentum': 0.0, 'constraints': 0.0}
        
        return {
            'energy': abs(current.energy - self.initial_quantities.energy) / 
                     (abs(self.initial_quantities.energy) + self.tolerance),
            'momentum': np.max(np.abs(current.momentum - 
                                    self.initial_quantities.momentum)) /
                       (np.max(np.abs(self.initial_quantities.momentum)) + 
                        self.tolerance),
            'angular_momentum': np.max(np.abs(current.angular_momentum - 
                                            self.initial_quantities.angular_momentum)) /
                               (np.max(np.abs(self.initial_quantities.angular_momentum)) + 
                                self.tolerance),
            'constraints': np.max(np.abs(current.constraints))
        }
    
    def _compute_energy(self, 
                       state: 'QuantumState',
                       hamiltonian: 'QuantumOperator') -> float:
        """Compute total energy of the system."""
        energy = 0.0
        for idx, coeff in state.coefficients.items():
            energy += abs(coeff)**2 * hamiltonian.expectation_value(
                state.basis_states[idx]
            )
        return energy.real
    
    def _compute_momentum(self, 
                         state: 'QuantumState',
                         momentum_op: 'QuantumOperator') -> np.ndarray:
        """Compute total momentum vector."""
        momentum = np.zeros(3)
        for idx, coeff in state.coefficients.items():
            momentum += abs(coeff)**2 * momentum_op.expectation_value(
                state.basis_states[idx]
            )
        return momentum.real
    
    def _compute_angular_momentum(self, 
                                state: 'QuantumState',
                                angular_momentum_op: 'QuantumOperator') -> np.ndarray:
        """Compute total angular momentum vector."""
        angular_momentum = np.zeros(3)
        for idx, coeff in state.coefficients.items():
            angular_momentum += abs(coeff)**2 * angular_momentum_op.expectation_value(
                state.basis_states[idx]
            )
        return angular_momentum.real
    
    def _compute_constraints(self, 
                           state: 'QuantumState',
                           constraint_ops: List['QuantumOperator']) -> np.ndarray:
        """Compute constraint violations."""
        violations = []
        for constraint_op in constraint_ops:
            violation = 0.0
            for idx, coeff in state.coefficients.items():
                violation += abs(coeff)**2 * abs(constraint_op.expectation_value(
                    state.basis_states[idx]
                ))
            violations.append(violation.real)
        return np.array(violations)

