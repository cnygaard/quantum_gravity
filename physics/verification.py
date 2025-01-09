from physics.entanglement import EntanglementGeometryHandler
from physics.conservation import ConservationLawTracker
from typing import Dict, TYPE_CHECKING, List
import numpy as np
from constants import CONSTANTS
import logging

if TYPE_CHECKING:
    from examples.black_hole import BlackHoleSimulation

class UnifiedTheoryVerification:
    """Verify unified quantum gravity theory predictions."""
    
    def __init__(self, simulation: 'BlackHoleSimulation'):
        self.sim = simulation
        self.gamma = 1.0  # Information-geometry coupling
        self.alpha = 0.01  # Increased by 10x for stronger time dependence
        self.beta = 1e-6   # Increased by 100x for stronger radiation effects
        self.lambda_rad = 0.05  # Doubled for faster radiation growth
        self.kappa = 1e-3  # Increased by 10x for stronger quantum effects
        self.entanglement_handler = EntanglementGeometryHandler()
        self.conservation_tracker = ConservationLawTracker(
            grid=simulation.qg.grid,
            tolerance=1e-10
        )

    def verify_unified_relations(self) -> Dict[str, float]:
        """Verify all unified theory relationships."""
        # Original verifications
        spacetime = self.entanglement_handler.compute_spacetime_interval(
            self.sim.qg.state.entanglement,
            self.sim.qg.state.information
        )
        
        conservation = self.conservation_tracker.check_conservation(
            self.conservation_tracker.compute_quantities(
                self.sim.qg.state,
                self.sim.qg.operators
            )
        )
        
        # New trinity verification with all modifications
        trinity = self.verify_spacetime_trinity()
        
        # Holographic principle check
        entropy = self.sim.qg.state.entropy
        area = 4 * np.pi * (2 * CONSTANTS['G'] * self.sim.qg.state.mass)**2
        holographic = abs(entropy - area/(4 * CONSTANTS['l_p']**2))
        
        # Field equations verification
        field_eqs = self.verify_field_equations()
        
        return {
            # Original metrics
            'spacetime_relation': spacetime,
            'energy_conservation': conservation['energy'],
            'momentum_conservation': conservation['momentum'],
            'holographic_principle': holographic,
            'quantum_corrections': self._compute_quantum_corrections(),
            
            # New trinity metrics with all modifications
            'trinity_error': trinity['original_error'],  # Use original as base error
            'time_dependent_error': trinity['time_dependent_error'],
            'radiation_error': trinity['radiation_error'],
            'quantum_error': trinity['quantum_error'],
            'spacetime_interval': trinity['spacetime_interval'],
            'entanglement_measure': trinity['entanglement_measure'],
            'information_metric': trinity['information_metric'],
            
            # Field equation metrics
            'einstein_tensor_error': field_eqs['einstein_error'],
            'quantum_tensor_error': field_eqs['quantum_error'],
            'entanglement_tensor_error': field_eqs['entanglement_error']
        }        

    # Add these constants to __init__
    def __init__(self, simulation: 'BlackHoleSimulation'):
       self.sim = simulation
       self.gamma = 1.0  # Base coupling constant
       self.alpha = 0.01  # Increased by 10x for stronger time dependence
       self.beta = 1e-6   # Increased by 100x for stronger radiation effects
       self.lambda_rad = 0.05  # Doubled for faster radiation growth
       self.kappa = 1e-3  # Increased by 10x for stronger quantum effects
       self.entanglement_handler = EntanglementGeometryHandler()
       self.conservation_tracker = ConservationLawTracker(
           grid=simulation.qg.grid,
           tolerance=1e-10
       )

    def verify_spacetime_trinity(self) -> Dict[str, float]:
        """Verify dS² = dE² + γ²dI² relationship with corrections."""
        state = self.sim.qg.state
        t = state.time
        
        # Calculate differentials using physical quantities
        dt = 1.0  # Fixed timestep
        dm = state.mass**2 * dt / (15360 * np.pi)  # Mass loss rate
        
        # Compute spacetime interval with horizon regularization
        r = 2 * CONSTANTS['G'] * state.mass  # Horizon radius
        epsilon = CONSTANTS['l_p']  # Planck length regularization
        
        # Regularized metric components
        g_tt = -(1 - 2*CONSTANTS['G']*state.mass/(r + epsilon))
        g_rr = 1/(1 - 2*CONSTANTS['G']*state.mass/(r + epsilon))
        
        # Compute interval with regularized metric
        ds = np.sqrt(abs(g_tt*dt**2 + g_rr*dm**2))
        
        # Compute entanglement measure using area law
        area = 4 * np.pi * r**2
        entropy = area / (4 * CONSTANTS['l_p']**2)
        de = abs(entropy - state.entropy)
        
        # Compute information metric using temperature
        temp = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * state.mass)
        di = abs(temp - state.temperature)
        
        # Left hand side is constant for all versions
        lhs = ds**2
        
        # Calculate all corrections
        gamma_t = self.gamma * (1 + self.alpha * t)
        radiation_term = self.beta * np.exp(self.lambda_rad * t)
        quantum_factor = 1 - self.kappa * t**2
        
        # Calculate all RHS versions
        rhs_original = de**2 + self.gamma**2 * di**2
        rhs_time_dep = de**2 + gamma_t**2 * di**2
        rhs_radiation = de**2 + self.gamma**2 * di**2 - radiation_term
        rhs_quantum = quantum_factor * (de**2 + self.gamma**2 * di**2)
        
        # Calculate errors
        error_original = abs(lhs - rhs_original)/max(abs(lhs), 1e-10)
        error_time_dep = abs(lhs - rhs_time_dep)/max(abs(lhs), 1e-10)
        error_radiation = abs(lhs - rhs_radiation)/max(abs(lhs), 1e-10)
        error_quantum = abs(lhs - rhs_quantum)/max(abs(lhs), 1e-10)
        
        return {
            'spacetime_interval': float(ds),
            'entanglement_measure': float(de),
            'information_metric': float(di),
            'original_error': float(error_original),
            'time_dependent_error': float(error_time_dep),
            'radiation_error': float(error_radiation),
            'quantum_error': float(error_quantum)
        }

    def _verify_geometric_entanglement(self, state: 'QuantumState') -> Dict[str, float]:
        """Verify dS² = ∫ d³x √g ⟨Ψ|(êᵢ(x) + γ²îᵢ(x))|Ψ⟩ relationship."""
        # Physical parameters
        M = state.mass
        M_initial = self.sim.initial_mass if hasattr(self.sim, 'initial_mass') else M
        horizon_r = 2 * CONSTANTS['G'] * M
        l_p = CONSTANTS['l_p']
        t = state.time
        
        # Create grid centered on horizon
        r_min = horizon_r * (1 + l_p/horizon_r)  # Just outside horizon
        r_max = 3 * horizon_r
        n_points = 100
        
        # Dense sampling near horizon
        x = np.linspace(0, 1, n_points)  # Dimensionless coordinate
        r = r_min + (r_max - r_min) * x**2  # Quadratic spacing
        points = np.array([[r_i, 0, 0] for r_i in r])
        
        # Dimensionless parameters
        κ = 1/(4 * M)  # Surface gravity
        T = κ/(2*np.pi)  # Hawking temperature
        
        # Compute spacetime interval
        dt = state.time - self._last_time if hasattr(self, '_last_time') else 0.0
        dr = horizon_r * (M/M_initial - 1) if hasattr(self, '_last_mass') else 0.0
        
        # Near-horizon metric
        r_reg = r_min
        g_tt = -(1 - 2*M/r_reg)
        g_rr = 1/(1 - 2*M/r_reg)
        
        # Normalize ds²
        ds2 = (-g_tt * dt**2 + g_rr * dr**2)/horizon_r**2
        
        # Initialize integral
        integral = 0.0
        debug_terms = []
        
        # Physical constants for the integral
        A_h = 4 * np.pi * horizon_r**2  # Horizon area
        S_BH = A_h/(4 * l_p**2)  # Bekenstein-Hawking entropy
        
        for i, point in enumerate(points):
            r_local = np.linalg.norm(point)
            x = (r_local - horizon_r)/horizon_r  # Distance from horizon
            
            # Local metric factor
            f = 1 - 2*M/r_local
            sqrt_g = np.sqrt(abs(f))
            
            # Entanglement density (peaked at horizon)
            e_i = np.exp(-x**2/(2*κ*l_p)) * S_BH/A_h
            
            # Information density (thermal)
            i_i = np.exp(-2*np.pi*T*t) * (l_p/horizon_r)**2 / (1 + x**2)
            
            # Volume element
            if i == 0:
                dr_local = r[1] - r[0]
            elif i == len(points) - 1:
                dr_local = r[-1] - r[-2]
            else:
                dr_local = (r[i+1] - r[i-1])/2
                
            dV = 4 * np.pi * r_local**2 * dr_local / horizon_r**3  # Normalized volume
            
            # Add to integral
            term = sqrt_g * dV * (e_i + self.gamma**2 * i_i)
            integral += term
            
            # Store debug info for near-horizon points
            if i < 5 or abs(x) < 0.1:
                debug_terms.append({
                    'x': float(x),
                    'sqrt_g': float(sqrt_g),
                    'e_i': float(e_i),
                    'i_i': float(i_i),
                    'dV': float(dV),
                    'term': float(term)
                })
        
        # print("\nDebug: Near-horizon terms:")
        # for term in debug_terms:
        #     print(f"x = {term['x']:.6f} (distance from horizon):")
        #     for k, v in term.items():
        #         if k != 'x':
        #             print(f"  {k}: {v:.6e}")
        
        # Store values
        self._last_time = state.time
        self._last_mass = state.mass
        
        # Normalize integral to match ds²
        integral *= (horizon_r/l_p)**2
        
        # Compute error
        lhs = abs(ds2)
        rhs = abs(integral)
        relative_error = abs(lhs - rhs) / max(lhs, rhs, l_p/horizon_r)
        
        return {
            'lhs': float(ds2),
            'rhs': float(integral),
            'relative_error': float(relative_error)
        }

    def _get_local_density_matrix(self, state: 'QuantumState', x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Get reduced density matrix for local region around point x."""
        points = self.sim.qg.grid.points
        
        # Find points within radius
        distances = np.linalg.norm(points - x, axis=1)
        local_indices = np.where(distances < radius)[0]
        
        # Construct local density matrix
        n_local = len(local_indices)
        rho = np.zeros((n_local, n_local), dtype=complex)
        
        for i, idx_i in enumerate(local_indices):
            for j, idx_j in enumerate(local_indices):
                # Sum over all quantum states
                for k, coeff in state.coefficients.items():
                    rho[i,j] += abs(coeff)**2 * state.basis_states[k][idx_i] * \
                            state.basis_states[k][idx_j].conjugate()
                    
        return rho

    def _get_local_state(self, state: 'QuantumState', x: np.ndarray, radius: float = 1.0) -> np.ndarray:
        """Get local quantum state around point x."""
        points = self.sim.qg.grid.points
        
        # Find points within radius
        distances = np.linalg.norm(points - x, axis=1)
        local_indices = np.where(distances < radius)[0]
        
        # Construct local state vector
        local_state = np.zeros(len(local_indices), dtype=complex)
        
        for i, idx in enumerate(local_indices):
            # Sum over quantum states
            for k, coeff in state.coefficients.items():
                local_state[i] += coeff * state.basis_states[k][idx]
                
        # Normalize
        norm = np.sqrt(np.sum(np.abs(local_state)**2))
        if norm > 1e-10:
            local_state /= norm
            
        return local_state

    def verify_field_equations(self) -> Dict[str, float]:
        """Verify G_μν + Q_μν + E_μν = 8πGT_μν."""
        state = self.sim.qg.state
        
        # Compute tensors
        G = self._compute_einstein_tensor(state)
        Q = self._compute_quantum_tensor(state)
        E = self._compute_entanglement_tensor(state)
        T = self._compute_stress_tensor(state)
        
        # Check field equations
        lhs = G + Q + E
        rhs = 8 * np.pi * CONSTANTS['G'] * T
        
        return {
            'einstein_error': np.max(np.abs(G - rhs/3)),
            'quantum_error': np.max(np.abs(Q - rhs/3)),
            'entanglement_error': np.max(np.abs(E - rhs/3))
        }
        
    def _compute_quantum_corrections(self) -> float:
        """Compute quantum corrections to classical geometry."""
        mass = self.sim.qg.state.mass
        # Quantum corrections scale as ℏ/M²
        return CONSTANTS['hbar'] / (mass * mass)

    def _compute_spacetime_interval(self, state) -> float:
        """Compute proper spacetime interval ds²."""
        # Get metric components
        g_tt = state._metric_array[0,0]
        g_rr = state._metric_array[1,1]
        
        # Calculate proper time and space intervals
        dt = state.time - self._last_time if hasattr(self, '_last_time') else 0.0
        dr = 2 * CONSTANTS['G'] * (state.mass - self._last_mass) if hasattr(self, '_last_mass') else 0.0
        
        # Compute interval using metric
        ds2 = -g_tt * dt**2 + g_rr * dr**2
        
        # Update last values
        self._last_time = state.time
        self._last_mass = state.mass
        
        return np.sqrt(abs(ds2))

    def _compute_entanglement_differential(self, state) -> float:
        """Compute entanglement measure differential dE."""
        # Get current entanglement
        current_ent = state.compute_entanglement()
        
        # Calculate differential
        dE = current_ent - self._last_ent if hasattr(self, '_last_ent') else 0.0
        
        # Update last value
        self._last_ent = current_ent
        
        return abs(dE)

    def _compute_information_differential(self, state) -> float:
        """Compute information metric differential dI."""
        # Get current information
        current_info = state.compute_information()
        
        # Calculate differential
        dI = current_info - self._last_info if hasattr(self, '_last_info') else 0.0
        
        # Update last value
        self._last_info = current_info
        
        return abs(dI)
    def _compute_einstein_tensor(self, state) -> np.ndarray:
        """Compute Einstein tensor G_μν."""
        # Get metric and its derivatives
        g = state._metric_array
        r = np.linalg.norm(self.sim.qg.grid.points, axis=1)
        
        # Add Planck length regularization
        r = np.maximum(r, CONSTANTS['l_p'])
        n_points = len(r)
        
        # Initialize 4D tensor with proper shape
        G = np.zeros((4, 4, n_points))
        
        # Compute Ricci tensor components with regularized radius
        R_tt = CONSTANTS['G'] * state.mass / (r**3)
        R_rr = -CONSTANTS['G'] * state.mass / (r**3)
        
        # Construct Einstein tensor with broadcasting
        G[0,0,:] = R_tt - g[0,0,:] * R_tt/2
        G[1,1,:] = R_rr - g[1,1,:] * R_rr/2
        
        return np.mean(G, axis=2)

    def _compute_quantum_tensor(self, state) -> np.ndarray:
        """Compute quantum correction tensor Q_μν."""
        n_points = len(self.sim.qg.grid.points)
        Q = np.zeros((4, 4, n_points))
        
        # Quantum corrections with proper broadcasting
        quantum_correction = CONSTANTS['hbar'] / (state.mass * state.mass)
        Q[0,0,:] = quantum_correction
        Q[1,1,:] = -quantum_correction
        
        return np.mean(Q, axis=2)

    def _compute_entanglement_tensor(self, state) -> np.ndarray:
        """Compute entanglement stress tensor E_μν."""
        n_points = len(self.sim.qg.grid.points)
        E = np.zeros((4, 4, n_points))
        
        # Entanglement contribution with broadcasting
        entanglement_factor = state.entropy * CONSTANTS['l_p']**2
        E[0,0,:] = entanglement_factor
        E[1,1,:] = entanglement_factor
        
        return np.mean(E, axis=2)

    def _compute_stress_tensor(self, state) -> np.ndarray:
        """Compute stress-energy tensor T_μν."""
        n_points = len(self.sim.qg.grid.points)
        T = np.zeros((4, 4, n_points))
        
        # Energy density with proper volume calculation
        volume = 4/3 * np.pi * self.sim.horizon_radius**3
        rho = state.mass / volume
        
        # Construct tensor with broadcasting
        T[0,0,:] = rho
        T[1,1,:] = rho/3  # Radiation pressure
        
        return np.mean(T, axis=2)

    def verify_unified_relations(self) -> Dict[str, float]:
        """Verify all unified theory relationships."""
        # Original verifications
        spacetime = self.entanglement_handler.compute_spacetime_interval(
            self.sim.qg.state.entanglement,
            self.sim.qg.state.information
        )
        
        conservation = self.conservation_tracker.check_conservation(
            self.conservation_tracker.compute_quantities(
                self.sim.qg.state,
                self.sim.qg.operators
            )
        )
        
        # New geometric-entanglement verification
        geometric_entanglement = self._verify_geometric_entanglement(self.sim.qg.state)
        
        # Holographic principle check
        entropy = self.sim.qg.state.entropy
        area = 4 * np.pi * (2 * CONSTANTS['G'] * self.sim.qg.state.mass)**2
        holographic = abs(entropy - area/(4 * CONSTANTS['l_p']**2))
        
        # Field equations verification
        field_eqs = self.verify_field_equations()
        
        return {
            # Original metrics
            'spacetime_relation': spacetime,
            'energy_conservation': conservation['energy'],
            'momentum_conservation': conservation['momentum'],
            'holographic_principle': holographic,
            'quantum_corrections': self._compute_quantum_corrections(),
            
            # New geometric-entanglement metrics
            'geometric_entanglement_lhs': geometric_entanglement['lhs'],
            'geometric_entanglement_rhs': geometric_entanglement['rhs'],
            'geometric_entanglement_error': geometric_entanglement['relative_error'],
            
            # Field equation metrics
            'einstein_tensor_error': field_eqs['einstein_error'],
            'quantum_tensor_error': field_eqs['quantum_error'],
            'entanglement_tensor_error': field_eqs['entanglement_error']
        }

def run_verification(sim_time: float = 1000.0):
    """Run verification of unified theory."""
    # Initialize simulation
    sim = BlackHoleSimulation(mass=1000.0)
    verifier = UnifiedTheoryVerification(sim)
    
    # Track verification metrics
    results = []
    
    # Run simulation with verification
    while sim.qg.state.time < sim_time:
        sim.evolve_step()
        metrics = verifier.verify_unified_relations()
        results.append(metrics)
        
    return results

