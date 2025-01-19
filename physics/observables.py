# physics/observables.py
import numpy as np
from typing import Dict, List, Optional, Union
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from dataclasses import dataclass
from abc import ABC, abstractmethod
from constants import CONSTANTS
from core.grid import AdaptiveGrid, LeechLattice
from core.state import QuantumState
import logging


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

    def _compute_uncertainty(self,
                             state: 'QuantumState',
                             value: float
                             ) -> float:
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

    def _construct_reduced_density_matrix(
            self,
            state: 'QuantumState'
            ) -> csr_matrix:
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


class AreaObservable:
    def __init__(self, grid, normal=None):
        self.grid = grid
        self.normal = (
            normal if normal is not None
            else np.array([1.0, 0.0, 0.0])
        )

    def measure(self, state):
        """Measure horizon area with quantum corrections."""
        # Get horizon radius from state mass
        horizon_radius = 2 * CONSTANTS['G'] * state.mass

        # Calculate area using horizon radius
        area = 4 * np.pi * horizon_radius**2

        # Include quantum corrections
        area_correction = (
            CONSTANTS['l_p']**2 *
            np.log(area/CONSTANTS['l_p']**2)
        )

        return MeasurementResult(
            value=area,
            uncertainty=area_correction,
            metadata={
                'mass': state.mass,
                'radius': horizon_radius,
                'normal': self.normal
            }
        )


class ADMMassObservable:
    def __init__(self, grid):
        self.grid = grid
        # Enhanced evaporation rate calculation
        self.evaporation_rate = (
            CONSTANTS['hbar'] *
            CONSTANTS['c']**6) / (15360 * np.pi * CONSTANTS['G']**2)

    def measure(self, state):
        """Measure ADM mass with Hawking radiation."""
        # Calculate mass including evaporation
        current_mass = (
            state.initial_mass *
            (1 - state.time/state.evaporation_timescale)**(1/3)
        )
        # Apply quantum corrections
        mass_correction = CONSTANTS['hbar'] / (current_mass * CONSTANTS['G'])

        return MeasurementResult(
            value=current_mass,
            uncertainty=mass_correction,
            metadata={'time': state.time}
        )


class BlackHoleTemperatureObservable:
    def __init__(self, grid):
        self.grid = grid
        self.mass_obs = ADMMassObservable(grid)  # Initialize mass observable

    def measure(self, state):
        """Measure black hole temperature with quantum corrections."""
        mass_result = self.mass_obs.measure(state)

        # Add Planck-scale corrections
        mass = max(mass_result.value, CONSTANTS['m_p'])  # Planck mass cutoff

        # Modified Hawking temperature with quantum corrections
        temp = (
            CONSTANTS['hbar'] * CONSTANTS['c']**3 /
            (8 * np.pi * CONSTANTS['G'] * mass) *
            (1 - CONSTANTS['l_p']/(2 * CONSTANTS['G'] * mass))
            )  # Leading quantum correction

        return MeasurementResult(
            value=temp,
            uncertainty=abs(temp * mass_result.uncertainty / mass),
            metadata={'mass': mass}
        )


class HawkingFluxObservable(Observable):
    """Measure Hawking radiation flux."""

    def __init__(self, grid):
        super().__init__(grid)
        # Initialize temperature observable
        self.temperature_obs = BlackHoleTemperatureObservable(grid)

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

    def measure(self, state):
        """Measure Hawking radiation flux."""
        # n_points = len(state.grid.points)
        # rows, cols, data = [], [], []

        # Calculate Stefan-Boltzmann constant in natural units
        sigma = (CONSTANTS['hbar'] * CONSTANTS['c']**6) / (15360 * np.pi**3 * CONSTANTS['G']**2)

        # Get temperature from black hole surface
        temp = self.temperature_obs.measure(state)

        # Calculate flux using Stefan-Boltzmann law
        flux_value = sigma * temp.value**4

        # Create measurement result with uncertainty propagation
        flux_uncertainty = 4 * sigma * temp.value**3 * temp.uncertainty

        return MeasurementResult(
            value=flux_value,
            uncertainty=flux_uncertainty,
            metadata={'temperature': temp.value}
        )
        # return csr_matrix((data, (rows, cols)),
        # shape=(n_points, n_points))
        # return MeasurementResult(flux, delta_flux)


class ScaleFactorObservable(Observable):
    """Observable for measuring cosmic scale factor."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct scale factor operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            # Scale factor from spatial metric components
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure cosmic scale factor."""
        # Extract scale factor from spatial metric components
        a = np.mean(np.sqrt(np.abs(state._metric_array[1:, 1:, :])))
        
        # Quantum corrections
        quantum_correction = CONSTANTS['l_p']**2 / a
        
        return MeasurementResult(
            value=a,
            uncertainty=quantum_correction,
            metadata={'time': state.time}
        )

class EnergyDensityObservable(Observable):
    """Observable for measuring energy density in stars and cosmological scenarios."""
    
    def __init__(self, grid):
        super().__init__(grid)
        
    def _construct_operator(self) -> csr_matrix:
        """Construct energy density operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure energy density with quantum corrections.
        
        Returns:
            MeasurementResult with:
            - value: np.ndarray for stellar case, float for cosmological
            - uncertainty: same shape as value
            - metadata: dict with additional information
        """
        is_cosmological = hasattr(state, 'scale_factor')
        
        if is_cosmological:
            return self._measure_cosmological(state)
        else:
            return self._measure_stellar(state)

    def _measure_stellar(self, state: 'QuantumState') -> MeasurementResult:
        points = self.grid.points
        r = np.linalg.norm(points, axis=1)
        r = np.maximum(r, CONSTANTS['l_p'])
        
        # Classical density with radial dependence
        mass = state.mass
        density = (3 * CONSTANTS['G'] * mass) / (4 * np.pi * r**3)
        
        # Get central density (minimum radius = maximum density)
        central_density = np.max(density)  # Use max since density peaks at center
        
        return MeasurementResult(
            value=central_density,
            uncertainty=CONSTANTS['hbar'] / (np.min(r)**3),
            metadata={'full_profile': density}
        )


    
    def _measure_cosmological(self, state: 'CosmologicalState') -> MeasurementResult:
        """Measure cosmological energy density."""
        # Get basic Friedmann density
        H = state.hubble_parameter
        rho = 3 * H**2 / (8 * np.pi * CONSTANTS['G'])
        
        # Add quantum corrections for cosmology
        quantum_factor = 1 + (CONSTANTS['l_p']/state.scale_factor)**2
        rho *= quantum_factor
        
        # Near bounce behavior
        rho_crit = 0.41 * CONSTANTS['rho_planck']
        if rho > rho_crit:
            rho = rho_crit * (2 - rho/rho_crit)  # Smooth bounce transition
        
        # Uncertainty from quantum corrections
        uncertainty = CONSTANTS['hbar'] * H**3
        
        # For cosmological case, we still return a scalar
        return MeasurementResult(
            value=float(rho),
            uncertainty=float(uncertainty),
            metadata={
                'time': state.time,
                'scale_factor': state.scale_factor,
                'hubble_parameter': H,
                'quantum_factor': quantum_factor
            }
        )
    

class QuantumCorrectionsObservable(Observable):
    """Observable for measuring quantum corrections to classical geometry."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct quantum corrections operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure quantum corrections to geometry."""
        # Calculate local quantum terms
        Q_local = CONSTANTS['hbar']**2 * np.sum(
            state._metric_array[1:, 1:, :]**2
        )
        
        # Add non-local corrections
        quantum_uncertainty = CONSTANTS['l_p'] / np.sqrt(len(self.grid.points))
        
        return MeasurementResult(
            value=Q_local,
            uncertainty=quantum_uncertainty,
            metadata={'time': state.time}
        )

class PerturbationSpectrumObservable(Observable):
    """Observable for measuring cosmological perturbation spectrum."""
    
    def _construct_operator(self) -> csr_matrix:
        """Construct perturbation spectrum operator."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))
        
    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure perturbation power spectrum."""
        # Get metric perturbations
        delta_g = state._metric_array[1:, 1:, :] - np.mean(state._metric_array[1:, 1:, :])
        
        # Compute power spectrum via FFT
        k_values = 2 * np.pi * np.fft.fftfreq(len(self.grid.points))
        power = np.abs(np.fft.fftn(delta_g))**2
        
        # Include quantum effects
        quantum_correction = CONSTANTS['hbar'] * k_values**2
        
        return MeasurementResult(
            value=(k_values, power),
            uncertainty=quantum_correction,
            metadata={'time': state.time}
        )

class CosmicEvolutionObservable:
    """Track detailed cosmic evolution parameters."""
    
    def measure(self, state: 'CosmologicalState') -> MeasurementResult:
        # Hubble parameter with quantum corrections
        H = state.hubble_parameter
        H_quantum = H * (1 + (CONSTANTS['l_p'] * H)**2)
        
        # Equation of state w(t)
        w = state.pressure / state.energy_density
        
        # Acceleration parameter
        q = -state.scale_factor * H_quantum**2 / H**2
        
        # Cosmic entropy
        S = 4 * np.pi * (state.scale_factor / CONSTANTS['l_p'])**2
        
        return MeasurementResult(
            value={
                'hubble': H_quantum,
                'eos': w,
                'acceleration': q,
                'entropy': S
            },
            uncertainty=CONSTANTS['l_p'] * H,
            metadata={'time': state.time}
        )

class CosmicMatterRadiationObservable:
    """Track matter-radiation coupling and phase transitions."""
    def measure(self, state: 'CosmologicalState') -> MeasurementResult:
        # Matter-radiation coupling
        coupling = self._compute_matter_radiation_coupling(state)
        
        # Phase transition detection
        phase = self._detect_phase_transitions(state)
        
        # Track perturbation growth
        perturbations = self._compute_perturbation_growth(state)
        
        return MeasurementResult(
            value={
                'coupling': coupling,
                'phase': phase,
                'perturbations': perturbations
            },
            uncertainty=CONSTANTS['l_p'] * state.hubble_parameter
        )

class StellarTemperatureObservable:
    def __init__(self, grid):
        self.grid = grid
        self.mass_obs = ADMMassObservable(grid)
        self.gamma = 0.55  # Coupling constant
        self.T_core = 1.57e7  # Core temperature in K
        self.T_surface = 5778  # Surface temperature in K

    def measure(self, state):
        try:
            r = np.linalg.norm(self.grid.points, axis=1)
            r_max = np.max(r)
            
            # Temperature profile with realistic scaling
            T_core = 1.57e7  # Core temperature in K
            T_surface = 5778  # Surface temperature in K
            
            # Combine core and surface contributions
            T = T_core * (r_max/r)**0.25 * np.exp(-r/r_max) + T_surface
            
            # Add quantum corrections with proper bounds
            beta = CONSTANTS['l_p'] / r_max
            gamma_eff = self.gamma * beta * np.sqrt(0.407)
            T *= (1 + min(gamma_eff, 0.1))  # Limit quantum effects
            
            return MeasurementResult(
                value=T,
                uncertainty=0.05 * T,  # 5% uncertainty
                metadata={'r_max': r_max, 'T_core': T_core}
            )
        except Exception as e:
            logging.error(f"Temperature measurement error: {str(e)}")
            return MeasurementResult(
                value=np.full(len(self.grid.points), 5778.0),
                uncertainty=0.0
            )

class PressureObservable(Observable):
    def __init__(self, grid):
        super().__init__(grid)
        self.energy_obs = EnergyDensityObservable(grid)

    def _construct_operator(self) -> csr_matrix:
        """Construct pressure operator matrix."""
        n_points = len(self.grid.points)
        rows, cols, data = [], [], []
        
        # Build sparse matrix elements
        for i in range(n_points):
            rows.append(i)
            cols.append(i)
            data.append(1.0)
            
            # Add neighbor coupling terms
            for j in self.grid.neighbors[i]:
                if j > i:  # Avoid double counting
                    coupling = 1.0 / len(self.grid.neighbors[i])
                    rows.extend([i, j])
                    cols.extend([j, i])
                    data.extend([coupling, coupling])
        
        return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    def measure(self, state: 'QuantumState') -> MeasurementResult:
        """Measure pressure with realistic solar scaling."""
        P_core = 2.65e16  # Solar core pressure in Pa
        
        points = self.grid.points
        r = np.linalg.norm(points, axis=1)
        r = np.maximum(r, CONSTANTS['l_p'])
        r_max = np.max(r)
        
        # Pressure profile with proper radial scaling
        pressure = P_core * (r_max/r)**4 * np.exp(-r/r_max)
        
        # Add quantum corrections
        beta = CONSTANTS['l_p'] / r_max
        gamma_eff = 0.55 * beta * np.sqrt(0.407)
        pressure *= (1 + gamma_eff)
        
        return MeasurementResult(
            value=pressure,
            uncertainty=0.1 * pressure,
            metadata={'r_max': r_max}
        )

class RingdownObservable(Observable):
    def __init__(self, grid):
        super().__init__(grid)
        self.leech = LeechLattice(points=CONSTANTS['LEECH_LATTICE_POINTS'])

    def _construct_operator(self) -> csr_matrix:
            """Construct ringdown operator matrix."""
            n_points = len(self.grid.points)
            rows, cols, data = [], [], []
            
            # Build sparse matrix elements for ringdown modes
            for i in range(n_points):
                rows.append(i)
                cols.append(i)
                # Diagonal elements for frequency operator
                data.append(1.0)
                
                # Add neighbor coupling terms
                for j in self.grid.neighbors[i]:
                    if j > i:  # Avoid double counting
                        coupling = 1.0 / len(self.grid.neighbors[i])
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([coupling, coupling])
            
            return csr_matrix((data, (rows, cols)), shape=(n_points, n_points))

    def measure(self, state):
        # Standard QNM frequencies
        M = state.mass
        l = 2  # Quadrupole mode
        n = 0  # Fundamental tone
        omega_standard = self._compute_standard_frequency(M, l, n)
        
        # Get Leech lattice coupling with default value
        beta_leech = self.leech.compute_effective_coupling()
        if beta_leech is None:
            beta_leech = 0.407  # Theoretical value from Leech lattice

        # Leech lattice correction
        #beta_leech = self.leech.compute_effective_coupling()
        omega_modified = omega_standard * (1 + beta_leech)
        
        return MeasurementResult(
            value=omega_modified,
            uncertainty=abs(omega_modified - omega_standard),
            metadata={
                'standard_freq': omega_standard,
                'leech_correction': beta_leech
            }
        )
        
    def _compute_standard_frequency(self, M, l, n):
        """Standard 4D Schwarzschild QNM frequency"""
        # For l=2, n=0 mode
        omega_R = (0.37367 - 0.08896j) / M
        return omega_R
