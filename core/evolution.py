# core/evolution.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from scipy.sparse import csr_matrix, linalg as sparse_linalg
from dataclasses import dataclass
from constants import CONSTANTS

@dataclass
class EvolutionConfig:
    """Configuration for time evolution."""
    dt_min: float  # Minimum allowed timestep
    dt_max: float  # Maximum allowed timestep
    rtol: float    # Relative tolerance
    atol: float    # Absolute tolerance
    method: str    # Evolution method ('rk4', 'adaptive', 'splitting')

class TimeEvolution:
    def __init__(self, grid, config, error_tracker, conservation_tracker):
        self.grid = grid
        self.dt = config['dt']
        self.error_tolerance = config['error_tolerance']
        self.error_tracker = error_tracker
        self.conservation_tracker = conservation_tracker

    def step(self, state):
        """Evolve quantum state forward by one timestep."""
        # Basic implementation of time evolution step
        # This will need to be expanded based on your specific evolution equations

        # Track errors
        errors = self.error_tracker.compute_errors(
            state, 
            None,  # No previous state for first step
            self.conservation_tracker
        )

        # For now, just return the state unchanged
        return state        
    def evolve_to(self,
                  state: 'QuantumState',
                  t_final: float,
                  callback: Optional[Callable] = None) -> 'QuantumState':
        """Evolve state to specified time."""
        dt = self.config.dt_max
        while self.t < t_final:
            # Adjust final step to hit t_final exactly
            if self.t + dt > t_final:
                dt = t_final - self.t
            
            # Take a step
            new_state, dt_next = self._take_step(state, dt)
            
            # Update state and time
            state = new_state
            self.t += dt
            self.step_count += 1
            
            # Adjust timestep
            dt = min(self.config.dt_max, max(self.config.dt_min, dt_next))
            
            # Call callback if provided
            if callback is not None:
                callback(state, self.t, self.step_count)
            
        return state
    def _take_step(self, state: 'QuantumState', dt: float) -> Tuple['QuantumState', float]:
        """Take a single time step using selected method."""
        # Calculate mass loss rate from Hawking radiation
        radiation_power = CONSTANTS['hbar'] * CONSTANTS['c']**6 / (15360 * np.pi * CONSTANTS['G']**2 * state.mass**2)
        dm_dt = -radiation_power / (CONSTANTS['c']**2)
    
        # Update mass
        new_mass = state.mass + dm_dt * dt
    
        # Recalculate temperature, entropy, and radiation flux
        new_temperature = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * new_mass)
        new_entropy = 4 * np.pi * (2 * CONSTANTS['G'] * new_mass)**2 / (4 * CONSTANTS['l_p']**2)
        new_radiation_flux = radiation_power
    
        # Create new state with updated parameters
        new_state = QuantumState(
            mass=new_mass,
            temperature=new_temperature,
            entropy=new_entropy,
            radiation_flux=new_radiation_flux,
            time=state.time + dt
        )
    
        return new_state, dt
    
    def _step_rk4(self,
                  state: 'QuantumState',
                  dt: float) -> Tuple['QuantumState', float]:
        """Fourth-order Runge-Kutta step."""
        # Compute RK4 stages
        k1 = self._compute_derivative(state)
        k2 = self._compute_derivative(state + 0.5 * dt * k1)
        k3 = self._compute_derivative(state + 0.5 * dt * k2)
        k4 = self._compute_derivative(state + dt * k3)
        
        # Update state
        new_state = state + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Compute error estimate and next timestep
        error = self._estimate_error(state, new_state)
        dt_next = self._adjust_timestep(dt, error)
        
        return new_state, dt_next
    
    def _step_adaptive(self,
                      state: 'QuantumState',
                      dt: float) -> Tuple['QuantumState', float]:
        """Adaptive step with error control."""
        # Try full step
        full_state, _ = self._step_rk4(state, dt)
        
        # Try two half steps
        half_state, _ = self._step_rk4(state, dt/2)
        final_state, _ = self._step_rk4(half_state, dt/2)
        
        # Estimate error
        error = self._compute_difference(full_state, final_state)
        
        # Accept step if error is small enough
        if error < self.config.rtol:
            return final_state, self._adjust_timestep(dt, error)
        else:
            # Reduce timestep and try again
            return self._step_adaptive(state, dt/2)
    
    def _step_splitting(self,
                       state: 'QuantumState',
                       dt: float) -> Tuple['QuantumState', float]:
        """Operator splitting method."""
        # Split evolution into geometric and quantum parts
        geometric_part = self._evolve_geometric(state, dt/2)
        quantum_part = self._evolve_quantum(geometric_part, dt)
        new_state = self._evolve_geometric(quantum_part, dt/2)
        
        # Estimate error and next timestep
        error = self._estimate_splitting_error(state, new_state)
        dt_next = self._adjust_timestep(dt, error)
        
        return new_state, dt_next
    def _compute_derivative(self, state: 'QuantumState') -> 'QuantumState':
        """Compute time derivative of state."""
        # Get current values
        H = state.hubble_parameter  # Use state's hubble parameter
        a = state.scale_factor
        rho = state.energy_density
    
        # Scale factor evolution with quantum corrections
        quantum_factor = 1 + (CONSTANTS['l_p']/a)**2
        da_dt = H * a * quantum_factor
    
        # Modified energy density evolution including quantum effects
        w = state.equation_of_state  # Get from state
        quantum_pressure = CONSTANTS['hbar'] * H**2 / (2 * a**3)
        drho_dt = -3 * H * (rho + w*rho + quantum_pressure)
    
        # Update metric components with proper time evolution
        for i in range(len(self.grid.points)):
            for mu in range(1, 4):
                current = state.get_metric_component((mu, mu), i)
                # Include both classical and quantum terms
                new_value = current * (1 + da_dt * self.dt) + \
                       quantum_factor * CONSTANTS['l_p']**2 / a**3
                state.set_metric_component((mu, mu), i, new_value)
    
        # Compute power spectrum evolution
        k_modes = 2 * np.pi * np.fft.fftfreq(len(self.grid.points))
        delta_k = np.fft.fftn(state._metric_array[1:, 1:, :] - 
                         np.mean(state._metric_array[1:, 1:, :]))
    
        return da_dt, drho_dt, (k_modes, np.abs(delta_k)**2)
    def _estimate_error(self,
                       old_state: 'QuantumState',
                       new_state: 'QuantumState') -> float:
        """Estimate local truncation error."""
        # Compute various error contributions
        errors = self.error_tracker.compute_errors(
            new_state, old_state, self.conservation_tracker
        )
        
        return errors['total']
    
    def _adjust_timestep(self,
                        dt: float,
                        error: float) -> float:
        """Adjust timestep based on error estimate."""
        if error > 0:
            # Standard timestep adjustment formula
            dt_new = 0.9 * dt * (self.config.rtol/error)**0.2
            
            # Enforce timestep bounds
            return min(self.config.dt_max,
                      max(self.config.dt_min, dt_new))
        return dt
    
    def _evolve_geometric(self,
                         state: 'QuantumState',
                         dt: float) -> 'QuantumState':
        """Evolve geometric part of the state."""
        new_state = QuantumState(self.grid, state.eps_cut)
        
        for idx, coeff in state.coefficients.items():
            # Apply geometric evolution operator
            evolved = state.operators['geometric'].apply(
                state.basis_states[idx], dt
            )
            
            if abs(evolved['coefficient']) > state.eps_cut:
                new_state.add_basis_state(
                    idx,
                    evolved['coefficient'],
                    evolved['state']
                )
        
        return new_state
    
    def _evolve_quantum(self,
                       state: 'QuantumState',
                       dt: float) -> 'QuantumState':
        """Evolve quantum part of the state."""
        new_state = QuantumState(self.grid, state.eps_cut)
        
        for idx, coeff in state.coefficients.items():
            # Apply quantum evolution operator
            evolved = state.operators['quantum'].apply(
                state.basis_states[idx], dt
            )
            
            if abs(evolved['coefficient']) > state.eps_cut:
                new_state.add_basis_state(
                    idx,
                    evolved['coefficient'],
                    evolved['state']
                )
        
        return new_state

    def _evolve_state(self, dt: float):
        """Single evolution step."""
        # Scale factor evolution
        self.qg.state.scale_factor *= (1 + self.hubble_parameter * dt)
        
        # Energy density should dilute with expansion
        # ρ ∝ a^(-3(1+w)) where w is equation of state
        w = self.qg.state.equation_of_state
        dilution_factor = (1 + self.hubble_parameter * dt)**(-3*(1+w))
        self.qg.state.energy_density *= dilution_factor


    def _estimate_splitting_error(self,
                                old_state: 'QuantumState',
                                new_state: 'QuantumState') -> float:
        """Estimate error from operator splitting."""
        # Compute commutator norm as error estimate
        commutator = self._compute_commutator(
            state.operators['geometric'],
            state.operators['quantum']
        )
        
        return np.linalg.norm(commutator) * (dt**3) / 24.0
    
    def _compute_commutator(self,
                          op1: 'QuantumOperator',
                          op2: 'QuantumOperator') -> csr_matrix:
        """Compute commutator of two operators."""
        # Sparse matrix multiplication for efficiency
        return op1.matrix @ op2.matrix - op2.matrix @ op1.matrix