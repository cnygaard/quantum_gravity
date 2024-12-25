# numerics/errors.py

from typing import Dict, List, Optional
from physics.conservation import ConservationLawTracker
import numpy as np

class ErrorTracker:
    """Track various sources of numerical errors."""
    
    def __init__(self, 
                 grid: 'AdaptiveGrid',
                 base_tolerances: Dict[str, float]):
        self.grid = grid
        self.base_tolerances = base_tolerances
        self.error_history: Dict[str, List[float]] = {
            'truncation': [],
            'constraint': [],
            'conservation': [],
            'discretization': [],
            'total': []
        }
        
    def compute_errors(self, 
                      state: 'QuantumState',
                      prev_state: Optional['QuantumState'],
                      conservation_tracker: ConservationLawTracker) -> Dict[str, float]:
        """Compute all error measures."""
        errors = {}
        
        # Truncation error from basis state cutoff
        errors['truncation'] = self._compute_truncation_error(state)
        
        # Constraint violation errors
        if prev_state is not None:
            errors['constraint'] = self._compute_constraint_error(
                state, conservation_tracker
            )
            
            # Conservation law violations
            errors['conservation'] = self._compute_conservation_error(
                state, prev_state, conservation_tracker
            )
            
            # Discretization errors
            errors['discretization'] = self._compute_discretization_error(
                state, prev_state
            )
        
        # Total error estimate
        errors['total'] = np.sqrt(sum(error**2 for error in errors.values()))
        
        # Update history
        for key, value in errors.items():
            self.error_history[key].append(value)
            
        return errors
    
    def _compute_truncation_error(self, 
                                state: 'QuantumState') -> float:
        """Estimate error from basis truncation."""
        # Sum of squares of smallest retained coefficients
        small_coeffs = [abs(c)**2 for c in state.coefficients.values() 
                       if abs(c) < 10*state.eps_cut]
        return np.sqrt(sum(small_coeffs)) if small_coeffs else 0.0
    
    def _compute_constraint_error(self,
                                state: 'QuantumState',
                                conservation_tracker: ConservationLawTracker) -> float:
        """Compute normalized constraint violation."""
        quantities = conservation_tracker.compute_quantities(
            state, state.operators
        )
        return np.max(np.abs(quantities.constraints))
    
    def _compute_conservation_error(self,
                                  state: 'QuantumState',
                                  prev_state: 'QuantumState',
                                  conservation_tracker: ConservationLawTracker) -> float:
        """Compute conservation law violations."""
        violations = conservation_tracker.check_conservation(
            conservation_tracker.compute_quantities(state, state.operators)
        )
        return max(violations.values())
    
    def _compute_discretization_error(self,
                                    state: 'QuantumState',
                                    prev_state: 'QuantumState') -> float:
        """Estimate discretization error using Richardson extrapolation."""
        # Compare solutions at different grid resolutions
        coarse_solution = self._restrict_to_coarse_grid(state)
        prev_coarse = self._restrict_to_coarse_grid(prev_state)
        
        return np.max(np.abs(coarse_solution - prev_coarse)) / \
               (self.grid.l_p * len(self.grid.points))**2
    
    def _restrict_to_coarse_grid(self, 
                                state: 'QuantumState') -> np.ndarray:
        """Project state onto coarser grid for error estimation."""
        coarse_points = self.grid.points[::2]  # Take every other point
        restricted = np.zeros(len(coarse_points), dtype=complex)
        
        for idx, coeff in state.coefficients.items():
            if idx % 2 == 0:  # Only use points that exist in coarse grid
                restricted[idx//2] = coeff
                
        return restricted

    def get_error_summary(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of error history."""
        summary = {}
        for error_type, history in self.error_history.items():
            if history:
                summary[error_type] = {
                    'mean': np.mean(history),
                    'std': np.std(history),
                    'max': np.max(history),
                    'current': history[-1]
                }
        return summary