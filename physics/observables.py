# physics/observables.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from dataclasses import dataclass
from abc import ABC, abstractmethod
#from __init__ import CONSTANTS, QuantumGravity
from constants import CONSTANTS
#from quantum_gravity import CONSTANTS

@dataclass
class MeasurementResult:
    """Container for measurement results."""
    value: Union[float, np.ndarray]
    uncertainty: Union[float, np.ndarray]
    metadata: Optional[Dict] = None

class Observable(ABC):
    """Base class for quantum gravity observables."""
    
    def __init__(self, grid: 'AdaptiveGrid'):
        self.grid = grid
        self._operator: Optional[csr_matrix] = None
    
    @property
    def operator(self) -> csr_matrix:
        """Get observable operator, computing if necessary."""
        if self._operator is None:
            self._operator = self._construct_operator()
        return self._operator
    
    @abstractmethod
    def _construct_operator(self) -> csr_matrix:
        """Construct observable operator."""
        pass
    
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Perform measurement on state."""
        # Default implementation calls _compute_value
        value = self._compute_value(state)
        uncertainty = self._compute_uncertainty(state, value)
        return MeasurementResult(value, uncertainty)
    
    def _compute_value(self, state: 'QuantumState') -> float:
        """Compute observable expectation value."""
        operator = self.operator
        return state.expectation_value(operator)
    
    def _compute_uncertainty(self, state: 'QuantumState', value: float) -> float:
        """Compute measurement uncertainty."""
        operator_squared = self.operator @ self.operator
        expectation_squared = state.expectation_value(operator_squared)
        variance = expectation_squared - value**2
        return np.sqrt(abs(variance))


class GeometricObservable(Observable):
    """Base class for geometric observables."""
    
    def __init__(self, grid: 'AdaptiveGrid'):
        super().__init__(grid)
        
    def ensure_gauge_invariance(self, operator: csr_matrix) -> csr_matrix:
        """Ensure operator is gauge invariant."""
        # Project out gauge degrees of freedom
        gauge_proj = self._construct_gauge_projector()
        return gauge_proj @ operator @ gauge_proj

    def _construct_gauge_projector(self) -> csr_matrix:
        """Construct projector onto gauge-invariant subspace."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
            # Add gauge transformation generators
            for j in self.grid.neighbors[i]:
                if j > i:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([-0.5, -0.5])
                    
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

class VolumeObservable(GeometricObservable):
    """Volume measurement operator."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct volume operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Local volume element
            vol_element = self._compute_volume_element(i)
            rows.append(i)
            cols.append(i)
            data.append(vol_element)
            
        operator = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        return self.ensure_gauge_invariance(operator)
    
    def _compute_volume_element(self, i: int) -> float:
        """Compute volume element at point i."""
        x = self.grid.points[i]
        
        # Basic volume element
        dV = self.grid.get_volume_element(i)
        
        # Quantum corrections
        quantum_factor = 1.0 + (self.grid.l_p / np.linalg.norm(x))**2
        
        return dV * quantum_factor

class AreaObservable(GeometricObservable):
    """Area measurement operator."""
    
    def __init__(self, grid: 'AdaptiveGrid', normal: np.ndarray):
        super().__init__(grid)
        self.normal = normal / np.linalg.norm(normal)
    
    def _construct_operator(self) -> csr_matrix:
        """Construct area operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Local area element
            area_element = self._compute_area_element(i)
            rows.append(i)
            cols.append(i)
            data.append(area_element)
            
        operator = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        return self.ensure_gauge_invariance(operator)
    
    def _compute_area_element(self, i: int) -> float:
        """Compute area element at point i."""
        x = self.grid.points[i]
        
        # Project onto surface normal
        proj = np.dot(x, self.normal)
        
        # Basic area element
        dA = self.grid.get_area_element(i)
        
        # Quantum corrections
        quantum_factor = 1.0 + (self.grid.l_p / abs(proj))**2
        
        return dA * quantum_factor

class CurvatureObservable(GeometricObservable):
    """Curvature measurement operator."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct curvature operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Local curvature
            R = self._compute_curvature(i)
            rows.append(i)
            cols.append(i)
            data.append(R)
            
            # Curvature correlations
            for j in self.grid.neighbors[i]:
                if j > i:
                    R_corr = self._compute_curvature_correlation(i, j)
                    if abs(R_corr) > 1e-10:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([R_corr, R_corr.conjugate()])
        
        operator = csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        return self.ensure_gauge_invariance(operator)
    
    def _compute_curvature(self, i: int) -> float:
        """Compute local curvature at point i."""
        x = self.grid.points[i]
        r = np.linalg.norm(x)
        
        # Classical curvature
        R_class = -2.0 / r**3
        
        # Quantum corrections
        R_quantum = self.grid.l_p**2 / r**5
        
        return R_class + R_quantum
    
    def _compute_curvature_correlation(self, i: int, j: int) -> complex:
        """Compute curvature correlation between points."""
        dx = self.grid.points[j] - self.grid.points[i]
        r = np.linalg.norm(dx)
        
        return 1j * self.grid.l_p / r**3

class EntanglementObservable(Observable):
    """Entanglement entropy measurement operator."""
    
    def __init__(self, grid: 'AdaptiveGrid', region_A: List[int]):
        super().__init__(grid)
        self.region_A = set(region_A)
    
    def _construct_operator(self) -> csr_matrix:
        """Construct entanglement entropy operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        # Construct reduced density matrix elements
        for i in self.region_A:
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
            for j in self.grid.neighbors[i]:
                if j in self.region_A:
                    coupling = self._compute_entanglement_coupling(i, j)
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling.conjugate()])
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
    def _compute_entanglement_coupling(self, i: int, j: int) -> complex:
        """Compute entanglement coupling between points."""
        dx = self.grid.points[j] - self.grid.points[i]
        r = np.linalg.norm(dx)
        
        return 1j * np.exp(-r/self.grid.l_p) / r

    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure entanglement entropy."""
        # Construct reduced density matrix
        rho_A = self._construct_reduced_density_matrix(state)
        
        # Compute von Neumann entropy
        eigenvals = sparse_linalg.eigsh(rho_A, k=min(6, rho_A.shape[0]-1),
                                      which='LM', return_eigenvectors=False)
        
        # Remove numerical noise
        eigenvals = eigenvals[eigenvals > 1e-10]
        
        # Compute entropy
        S = -np.sum(eigenvals * np.log(eigenvals))
        
        # Estimate uncertainty
        dS = np.sqrt(np.sum(np.log(eigenvals)**2 * eigenvals))
        
        return MeasurementResult(S, dS)
    
    def _construct_reduced_density_matrix(self, 
                                        state: 'QuantumState') -> csr_matrix:
        """Construct reduced density matrix for region A."""
        n_A = len(self.region_A)
        rows, cols, data = [], [], []
        
        for i in self.region_A:
            for j in self.region_A:
                val = 0.0
                for k, coeff in state.coefficients.items():
                    if k not in self.region_A:
                        continue
                    val += abs(coeff)**2 * state.basis_states[k][i] * \
                           state.basis_states[k][j].conjugate()
                
                if abs(val) > 1e-10:
                    rows.append(i)
                    cols.append(j)
                    data.append(val)
        
        return csr_matrix((data, (rows, cols)), shape=(n_A, n_A))
    
# class ADMMassObservable(Observable):
#     """Measure ADM mass."""
    
#     def measure(self, state: 'QuantumState') -> MeasurementResult:
#         # Basic implementation
#         mass = self._compute_adm_mass(state)
#         uncertainty = abs(mass) * 1e-2  # Rough estimate
#         return MeasurementResult(mass, uncertainty)
        
#     def _compute_adm_mass(self, state: 'QuantumState') -> float:
#         # Placeholder implementation
#         return 1.0  # Temporary return
class ADMMassObservable(Observable):
    """Measure ADM mass of black hole."""
    
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Compute ADM mass using asymptotic metric behavior."""
        mass = self._compute_value(state)
        uncertainty = self._compute_uncertainty(state, mass)
        return MeasurementResult(mass, uncertainty)
    
    def _compute_value(self, state: 'QuantumState') -> float:
        """Compute ADM mass from asymptotic metric."""
        operator = self._construct_operator()
        expectation_value = state.expectation_value(operator)
        return expectation_value * CONSTANTS['c']**2 / (2 * CONSTANTS['G'])
    
    def _compute_uncertainty(self, state: 'QuantumState', value: float) -> float:
        """Compute mass uncertainty from quantum fluctuations."""
        l_p = np.sqrt(CONSTANTS['hbar'] * CONSTANTS['G'] / CONSTANTS['c']**3)
        return l_p * value / self.grid.l_p

    def _construct_operator(self) -> csr_matrix:
        """Construct ADM mass operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        # Get asymptotic points
        r = np.linalg.norm(self.grid.points, axis=1)
        asymptotic_mask = r > 100 * self.grid.l_p
        
        for i in range(n_points):
            if asymptotic_mask[i]:
                # Add diagonal elements for asymptotic points
                rows.append(i)
                cols.append(i)
                data.append(r[i]**2 / np.sum(asymptotic_mask))
                
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

class BlackHoleTemperatureObservable(Observable):
    """Measure black hole Hawking temperature."""
    
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Compute Hawking temperature T = ℏc³/8πGM."""
        # First measure mass
        mass_obs = ADMMassObservable(self.grid)
        mass_result = mass_obs.measure(state)
        
        # Compute temperature
        temp = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * mass_result.value)
        
        # Temperature uncertainty from mass uncertainty
        delta_T = temp * mass_result.uncertainty / mass_result.value
        
        return MeasurementResult(temp, delta_T)

    def _construct_operator(self) -> csr_matrix:
        """Construct temperature operator based on horizon properties."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        # Get horizon points
        r = np.linalg.norm(self.grid.points, axis=1)
        horizon_points = np.where(abs(r - 2.0) < 0.1)[0]
        
        for i in horizon_points:
            # Temperature operator elements near horizon
            rows.append(i)
            cols.append(i)
            # T = ℏc³/8πGM scaling
            data.append(1.0 / (8 * np.pi * r[i]))
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
    
class HawkingFluxObservable(Observable):
    """Measure Hawking radiation flux."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct Hawking flux operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        # Get points near horizon
        r = np.linalg.norm(self.grid.points, axis=1)
        near_horizon = np.where(abs(r - 2.0) < 0.1)[0]
        
        for i in near_horizon:
            # Flux operator elements
            rows.append(i)
            cols.append(i)
            # F ∝ 1/M² scaling for Hawking radiation
            data.append(1.0 / (r[i] * r[i]))
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Compute Hawking radiation flux."""
        # First measure temperature
        temp_obs = BlackHoleTemperatureObservable(self.grid)
        temp_result = temp_obs.measure(state)
        
        # Stefan-Boltzmann law with quantum corrections
        sigma = CONSTANTS['h'] * CONSTANTS['c']**2 / (15360 * np.pi**3 * CONSTANTS['G']**2)
        flux = sigma * temp_result.value**4
        
        # Uncertainty from temperature uncertainty
        delta_flux = 4 * flux * temp_result.uncertainty / temp_result.value
        
        #return MeasurementResult(flux, delta_flux)        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))        
        #return MeasurementResult(flux, delta_flux)