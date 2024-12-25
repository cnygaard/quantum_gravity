# numerics/integrator.py
class AdaptiveIntegrator:
    """Adaptive time integrator for quantum gravity evolution."""
    
    def __init__(self,
                 rtol: float = 1e-6,
                 atol: float = 1e-8):
        self.rtol = rtol
        self.atol = atol
        
    def step(self,
            state: QuantumState,
            dt: float,
            operators: List['QuantumOperator']) -> Tuple[QuantumState, float]:
        """Take one adaptive time step."""
        # Estimate local error
        state1 = self._step_rk4(state, dt, operators)
        state2 = self._step_rk4(state, dt/2, operators)
        state2 = self._step_rk4(state2, dt/2, operators)
        
        # Compute error estimate
        error = self._compute_error(state1, state2)
        
        # Adjust step size
        if error > self.rtol:
            dt_new = 0.9 * dt * (self.rtol/error)**0.2
            return self.step(state, dt_new, operators)
        
        return state2, dt * min(2.0, max(0.5, (self.rtol/error)**0.2))
    
    def _step_rk4(self,
                 state: QuantumState,
                 dt: float,
                 operators: List['QuantumOperator']) -> QuantumState:
        """Classical RK4 step with sparse operations."""
        # Standard RK4 implementation using sparse operations
        k1 = self._compute_derivative(state, operators)
        k2 = self._compute_derivative(state + 0.5*dt*k1, operators)
        k3 = self._compute_derivative(state + 0.5*dt*k2, operators)
        k4 = self._compute_derivative(state + dt*k3, operators)
        
        return state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)