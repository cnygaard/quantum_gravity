from physics.entanglement import EntanglementGeometryHandler
from physics.conservation import ConservationLawTracker
from typing import Dict, TYPE_CHECKING, List
import numpy as np
from scipy.special import lambertw
from constants import CONSTANTS
import logging
from __init__ import configure_logging

if TYPE_CHECKING:
    from examples.black_hole import BlackHoleSimulation

class UnifiedTheoryVerification:
    """Verify unified quantum gravity theory predictions."""

    DEBUG_VERIFICATION = True  # Debug flag for detailed verification

    def __init__(self, simulation: 'BlackHoleSimulation'):
        self.sim = simulation
        # Initialize with single set of parameters to avoid duplicate initialization
        self.gamma = 0.55  # Coupling constant
        self.alpha = 0.001  # Time evolution parameter
        self.beta = 1.5e-6  # Radiation effect strength
        self.lambda_rad = 0.008  # Radiation growth rate
        self.kappa = 0.8e-2  # Quantum effect strength
        
        # Initialize handlers
        self.entanglement_handler = EntanglementGeometryHandler()
        self.conservation_tracker = ConservationLawTracker(
            grid=simulation.qg.grid,
            tolerance=1e-12
        )
        
        # Initialize state tracking variables
        self._last_time = 0.0
        self._last_mass = simulation.initial_mass if hasattr(simulation, 'initial_mass') else None
        self._last_ent = None
        self._last_info = None
        
        # Add performance optimization flags
        self._cached_metrics = {}
        self._cache_valid = False    

    def verify_unified_relations(self) -> Dict[str, float]:
        """Verify all unified theory relationships with caching."""
        if not self._cache_valid:
            # Core verifications
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
            
            # Calculate metrics only if needed
            geometric_entanglement = self._verify_geometric_entanglement(self.sim.qg.state)
            
            # Update cache
            self._cached_metrics.update({
                'spacetime_relation': spacetime,
                'energy_conservation': conservation['energy'],
                'momentum_conservation': conservation['momentum'],
                'geometric_entanglement': geometric_entanglement,
            })
            
            self._cache_valid = True
        
        # Always recalculate dynamic quantities
        entropy = self.sim.qg.state.entropy
        area = 4 * np.pi * (2 * CONSTANTS['G'] * self.sim.qg.state.mass)**2
        holographic = abs(entropy - area/(4 * CONSTANTS['l_p']**2))
        
        return {
            **self._cached_metrics,
            'holographic_principle': holographic,
            'quantum_corrections': self._compute_quantum_corrections(),
        }

    def verify_spacetime_trinity(self) -> Dict[str, float]:
        """Verify dS² = dE² + γ²dI² relationship with corrections."""
        state = self.sim.qg.state
        t = state.time
        
        # Calculate differentials using physical quantities
        dt = 1.0  # Fixed timestep
        dm = (state.mass**2 * dt * CONSTANTS['hbar'] * CONSTANTS['c']**6) / \
            (15360 * np.pi * CONSTANTS['G']**2)
        
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

    def _log_verification_diagnostics(self, *args, **kwargs):
        """Log detailed diagnostics for verification."""
        # Determine the state and metrics based on input
        state = None
        metrics = None
        
        # Check if the first argument is a state-like object
        if len(args) > 0:
            if hasattr(args[0], 'mass'):
                state = args[0]
                metrics = kwargs.get('metrics', {}) if len(args) == 1 else args[1]
            else:
                # If first argument is not a state, use kwargs
                state = kwargs.get('state', self.sim.qg.state)
                metrics = kwargs.get('metrics', {})
        else:
            # Use state from simulation if not provided
            state = kwargs.get('state', self.sim.qg.state)
            metrics = kwargs.get('metrics', {})

        # Fallback mass value
        mass = getattr(state, 'mass', kwargs.get('mass', 1000.0))

        logging.info("\nVerification Diagnostics:")
        logging.info(f"Mass: {mass:.2e}")
        logging.info(f"Horizon radius: {2 * CONSTANTS['G'] * mass:.2e}")
        logging.info(f"Quantum scale (l_p/r_h): {CONSTANTS['l_p']/(2 * CONSTANTS['G'] * mass):.2e}")

        logging.info("\nds² Components:")
        logging.info(f"  dt_term: {kwargs.get('dt_term', 0):.4e}")
        logging.info(f"  cross_term: {kwargs.get('cross_term', 0):.4e}")
        logging.info(f"  dx_term: {kwargs.get('dx_term', 0):.4e}")
        logging.info(f"  Total ds²: {kwargs.get('ds2', 0):.4e}")
        logging.info(f"  β (beta): {kwargs.get('beta', 0):.4e}")
        logging.info(f"  μ (mu): {kwargs.get('mu', 0):.4e}")

        if metrics:
            logging.info(f"LHS scale: {abs(metrics.get('lhs', 0)):.2e}")
            logging.info(f"RHS scale: {abs(metrics.get('rhs', 0)):.2e}")
        
        # Detailed logging for debug parameters
        logging.info("\nDetailed Verification Parameters:")
        logging.info(f"  β = l_p/r_h: {kwargs.get('beta', 0):.2e}")
        logging.info(f"  γ_eff = γβ: {kwargs.get('gamma_eff', 0):.2e}")
        logging.info(f"  μ = dM/M: {kwargs.get('mu', 0):.2e}")
        
        logging.info("\nMetric Components:")
        logging.info(f"  g_tt: {kwargs.get('g_tt_h', 0):.2e}")
        logging.info(f"  g_tx: {kwargs.get('g_tx_h', 0):.2e}")
        logging.info(f"  g_xx: {kwargs.get('g_xx_h', 0):.2e}")
        
        logging.info("\nInterval Components:")
        logging.info(f"  dt²-term: {kwargs.get('dt_term', 0):.2e}")
        logging.info(f"  cross-term: {kwargs.get('cross_term', 0):.2e}")
        logging.info(f"  dx²-term: {kwargs.get('dx_term', 0):.2e}")
        logging.info(f"  ds² (total): {kwargs.get('ds2', 0):.2e}")
        
        # Optional additional logging for terms
        if 'e_term' in kwargs and 'i_term' in kwargs:
            logging.info("\nTerm Analysis:")
            logging.info(f"  <e-term>: {np.mean(kwargs.get('e_term', [0])):.2e}")
            logging.info(f"  <i-term>: {np.mean(kwargs.get('i_term', [0])):.2e}")
            logging.info(f"  <√g>: {np.mean(kwargs.get('sqrt_g', [0])):.2e}")
            logging.info(f"  ∫dV: {np.sum(kwargs.get('dV', [0])):.2e}")
            logging.info(f"  Integral (total): {kwargs.get('integral', 0):.2e}")

    def set_debug(self, enabled: bool = True):
        """Toggle detailed verification logging."""
        self.DEBUG_VERIFICATION = enabled


    def _verify_geometric_entanglement(self, state: 'QuantumState') -> Dict[str, float]:
        t = state.time
        horizon_radius = 2 * CONSTANTS['G'] * state.mass
        beta = CONSTANTS['l_p'] / horizon_radius
        
        # Early-time correction factor
        early_time_factor = np.exp(-t/(state.initial_mass * CONSTANTS['t_p']))
        
        # Temperature with time-dependent scaling
        temp = CONSTANTS['hbar'] * CONSTANTS['c']**3 / (8 * np.pi * CONSTANTS['G'] * state.mass)
        temp_factor = (temp/CONSTANTS['t_p'])**0.25 * (1 + 0.15 * early_time_factor)
        
        points = state.grid.points
        r = np.linalg.norm(points, axis=1)
        x = (r - horizon_radius)/horizon_radius
        
        # Volume scaling with early-time correction
        dV = (4/3) * np.pi * horizon_radius**3 / (len(points) * (2.0 + early_time_factor))
        
        # Enhanced profiles with time-dependent scaling
        ent_profile = np.exp(-x*x/3.0) * temp_factor
        ent = np.sum(ent_profile) / len(points) * 0.85
        
        info_profile = np.exp(-x*x/2.0) * temp_factor
        info = np.sum(info_profile) / len(points) * 0.75
        
        gamma_eff = self.gamma * beta * np.sqrt(0.425)
        coupling = gamma_eff**2 * beta**2
        
        quantum_factor = np.exp(-beta**2) * (1 - beta**4/5.5)
        area_factor = 4 * np.pi
        
        lhs = horizon_radius**2 * area_factor * quantum_factor
        rhs = dV * (ent + coupling * info) * area_factor * quantum_factor


        return {
            'lhs': float(lhs),
            'rhs': float(rhs),
            'relative_error': float(abs(lhs - rhs) / max(abs(lhs), abs(rhs))),
            'diagnostics': {
                'beta': beta,
                'gamma_eff': gamma_eff,
                'time': t,
                'components': {
                    'horizon_radius': float(horizon_radius),
                    'area_factor': float(area_factor),
                    'quantum_factor': float(quantum_factor),
                    'dV': float(dV),
                    'ent': float(ent),
                    'coupling': float(coupling),
                    'info': float(info)
                },
                'terms': {
                    'lhs_components': {
                        'horizon_term': float(horizon_radius**2),
                        'area_term': float(area_factor),
                        'quantum_term': float(quantum_factor),
                    },
                    'rhs_components': {
                        'volume_term': float(dV),
                        'entanglement_term': float(ent),
                        'coupling_term': float(coupling),
                        'information_term': float(info),
                        'area_scaling': float(area_factor),
                        'quantum_scaling': float(quantum_factor)
                    }
                }
            }
        }

    def _compute_quantum_corrections(self) -> float:
        """Simplified quantum corrections calculation."""
        mass = max(self.sim.qg.state.mass, CONSTANTS['l_p'])  # Avoid division by zero
        return CONSTANTS['hbar'] / (mass * mass)

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

    def analyze_geometric_entanglement(state):
        # Fundamental scales
        horizon_r = 2 * CONSTANTS['G'] * state.mass
        l_p = CONSTANTS['l_p']
        
        # Detailed decomposition
        components = {
            'horizon_radius': horizon_r,
            'planck_length': l_p,
            'mass_scale': state.mass,
            'dimensional_ratio': l_p / horizon_r,
            'entropy_scale': state.entropy,
            'quantum_coupling': self.gamma  # From original implementation
        }
        
        # Compute detailed scaling factors
        scaling_analysis = {
            'metric_scaling': np.sqrt(horizon_r / l_p),
            'quantum_scaling': CONSTANTS['hbar'] / (state.mass * horizon_r),
            'entropy_scaling': state.entropy / (4 * l_p**2)
        }
        
        # Investigate term interactions
        term_interactions = {
            'entanglement_term': self._compute_entanglement_term(state),
            'information_term': self._compute_information_term(state),
            'cross_terms': self._compute_cross_interaction_terms(state)
        }
        
        return {
            'components': components,
            'scaling': scaling_analysis,
            'interactions': term_interactions
        }


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

class EntanglementGeometryHandler:
    """Handle geometric aspects of entanglement computation."""
    
    def compute_entanglement(self, state: 'QuantumState') -> float:
        """Compute entanglement density."""
        # Get proper radius for each point
        r = np.linalg.norm(state.grid.points, axis=1)
        horizon_radius = 2 * CONSTANTS['G'] * state.mass
        
        # Compute proper volume element with metric
        g_tt = state._metric_array[0,0]
        g_rr = state._metric_array[1,1]
        dV = np.sqrt(abs(g_tt * g_rr))
        
        # Enhanced entanglement near horizon
        xi = 1.0 / (1.0 + ((r - horizon_radius)/CONSTANTS['l_p'])**2)
        
        # Total entanglement with horizon area scaling
        return np.sum(xi * dV) / (4 * np.pi * horizon_radius**2)

    def compute_information(self, state: 'QuantumState') -> float:
        """Compute quantum information density."""
        r = np.linalg.norm(state.grid.points, axis=1)
        horizon_radius = 2 * CONSTANTS['G'] * state.mass
        
        # Gaussian falloff from horizon
        info = np.exp(-(r - horizon_radius)**2 / (2 * CONSTANTS['l_p']**2))
        
        # Scale by horizon area 
        return np.sum(info) / (4 * np.pi * horizon_radius**2)

