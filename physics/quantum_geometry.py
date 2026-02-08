"""
Quantum Geometry for Quantum Gravity v8

Implements the geometric framework for the v8 Fisher Information formulation
where spacetime geometry emerges from quantum information:

    g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

v8 additions (Section 5):
- Quantum Geometric Tensor: Q_μν = ⟨∂_μΨ|(𝟙 - |Ψ⟩⟨Ψ|)|∂_νΨ⟩
- Berry curvature: Ω_μν (antisymmetric part)
- Kähler structure: (G^Fisher, Ω, J) with J² = -𝟙
- Lorentzian signature from unitarity

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v8
"""

import numpy as np
from typing import Tuple, Optional
from constants import CONSTANTS, coherence_length
from physics.fisher_metric import FisherMetric, compute_v7_spacetime_metric
from physics.entanglement_strain import EntanglementStrainTensor


class QuantumGeometry:
    """
    Quantum geometry framework for v8 formulation.

    Provides geometric tools for computing emergent spacetime metrics
    from Fisher information and entanglement strain, including the
    Quantum Geometric Tensor and Berry curvature from v8 Section 5.

    The Quantum Geometric Tensor Q_μν encodes both:
    - Symmetric part: ¼G_μν^Fisher (metric/distance)
    - Antisymmetric part: (i/2)Ω_μν (Berry curvature/phase)

    These form a Kähler triplet (G^Fisher, Ω, J) on projective Hilbert space.

    Attributes:
        phi: Golden ratio (kept for geometric constructions)
        l_p: Planck length
        gamma_0: Immirzi parameter (0.274)
        fisher: Fisher information metric calculator
        strain: Entanglement strain tensor calculator
    """

    def __init__(self):
        """Initialize quantum geometry with v7 parameters."""
        self.phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        self.l_p = CONSTANTS['l_p']
        self.gamma_0 = CONSTANTS['gamma_0']  # Immirzi parameter

        # Backward compatibility aliases (DEPRECATED)
        self.Lambda = CONSTANTS['LEECH_LATTICE_POINTS']  # Deprecated

        # v7 core components
        self.fisher = FisherMetric()
        self.strain = EntanglementStrainTensor()

        # Initialize universal scales (v7 formulation)
        self.l_universal = self.universal_quantum_length()
        self.cosmic_factor = self.v7_cosmic_scale_factor()
        self.phase = self.quantum_geometric_phase()

    def universal_quantum_length(self) -> float:
        """
        Universal quantum length scale.

        l_universal = l_P × φ

        Returns:
            Universal length in Planck units
        """
        return self.l_p * self.phi

    def v7_cosmic_scale_factor(self) -> float:
        """
        v7 cosmic scale factor based on Immirzi parameter.

        Replaces the old Leech lattice factor with:
        cosmic_factor = π / γ₀

        Returns:
            Cosmic scale factor (dimensionless)
        """
        return np.pi / self.gamma_0

    def cosmic_scale_factor(self) -> float:
        """Alias for v7_cosmic_scale_factor for backward compatibility."""
        return self.v7_cosmic_scale_factor()

    def quantum_geometric_phase(self) -> float:
        """
        Quantum geometric phase.

        phase = 2π/φ

        Returns:
            Geometric phase (radians)
        """
        return 2 * np.pi / self.phi

    def compute_emergent_metric(self, state, position: np.ndarray = None) -> np.ndarray:
        """
        Compute emergent spacetime metric from v7 master equation.

        g_μν = ℓ_P² (G_μν^Fisher + γ₀ E_μν)

        Args:
            state: Quantum state object
            position: Spacetime position (optional)

        Returns:
            4x4 emergent metric tensor
        """
        # Compute Fisher metric
        G_fisher = self.fisher.compute_full_metric(state)

        # Compute entanglement strain tensor
        E_strain = self.strain.compute_strain(state, position)

        # Combine using v7 master equation
        g = compute_v7_spacetime_metric(G_fisher, E_strain, self.gamma_0)

        return g

    def compute_schwarzschild_v7_metric(self, r: float, M: float) -> np.ndarray:
        """
        Compute v7 metric for Schwarzschild-like geometry.

        Args:
            r: Radial coordinate (in l_p units)
            M: Mass (in m_p units)

        Returns:
            4x4 v7 metric tensor
        """
        r_s = 2 * CONSTANTS['G'] * M

        # Get coherence length from Tolman-Ehrenfest
        sigma = coherence_length(r, r_s)

        # Compute Fisher and strain components
        G_fisher = self.fisher.compute_schwarzschild_fisher(r, M, sigma)
        E_strain = self.strain.compute_schwarzschild_strain(r, M)

        # Combine with v7 master equation
        g = compute_v7_spacetime_metric(G_fisher, E_strain, self.gamma_0)

        return g

    def compute_quantum_correction(self, r: float, r_s: float) -> float:
        """
        Compute quantum correction factor for v7 formulation.

        correction = 1 + γ₀ × (ℓ_P / σ(r))²

        Args:
            r: Radial coordinate
            r_s: Schwarzschild radius

        Returns:
            Quantum correction factor (dimensionless)
        """
        sigma = coherence_length(r, r_s)
        correction = 1 + self.gamma_0 * (self.l_p / sigma)**2
        return correction

    def compute_dark_matter_ratio(self) -> float:
        """
        Compute v7 dark matter to baryonic matter ratio.

        M_DM/M_b = π/(2γ₀) ≈ 5.73

        Returns:
            Dark matter ratio (dimensionless)
        """
        return np.pi / (2 * self.gamma_0)

    # =========================================================================
    # v8 Quantum Geometric Tensor and Berry Curvature (Section 5)
    # =========================================================================

    def compute_quantum_geometric_tensor(
        self,
        psi: np.ndarray,
        dpsi: np.ndarray,
        dim: int = 4
    ) -> np.ndarray:
        """
        Compute the Quantum Geometric Tensor Q_μν.

        Q_μν = ⟨∂_μΨ|(𝟙 - |Ψ⟩⟨Ψ|)|∂_νΨ⟩

        The QGT is complex-valued and decomposes as:
        Q_μν = ¼G_μν^Fisher + (i/2)Ω_μν

        Args:
            psi: Quantum state vector |Ψ⟩ (normalized)
            dpsi: Array of state derivatives [∂_0Ψ, ∂_1Ψ, ∂_2Ψ, ∂_3Ψ]
                  Shape: (4, len(psi)) for 4D spacetime
            dim: Spacetime dimension (default 4)

        Returns:
            dim×dim complex Quantum Geometric Tensor Q_μν
        """
        Q = np.zeros((dim, dim), dtype=complex)

        # Projector onto orthogonal complement: P = 𝟙 - |Ψ⟩⟨Ψ|
        # For normalized |Ψ⟩: P|∂_νΨ⟩ = |∂_νΨ⟩ - ⟨Ψ|∂_νΨ⟩|Ψ⟩
        psi_conj = np.conj(psi)

        for mu in range(dim):
            for nu in range(dim):
                # ⟨∂_μΨ|∂_νΨ⟩
                overlap_deriv = np.dot(np.conj(dpsi[mu]), dpsi[nu])

                # ⟨∂_μΨ|Ψ⟩⟨Ψ|∂_νΨ⟩
                overlap_mu = np.dot(np.conj(dpsi[mu]), psi)
                overlap_nu = np.dot(psi_conj, dpsi[nu])
                projection_term = overlap_mu * overlap_nu

                Q[mu, nu] = overlap_deriv - projection_term

        return Q

    def extract_fisher_metric_from_qgt(self, Q: np.ndarray) -> np.ndarray:
        """
        Extract the Fisher metric from the Quantum Geometric Tensor.

        G_μν^Fisher = 4 × Re(Q_μν)

        The factor of 4 comes from Q_μν = ¼G_μν^Fisher + (i/2)Ω_μν.

        Args:
            Q: Quantum Geometric Tensor (complex)

        Returns:
            Real symmetric Fisher metric tensor
        """
        # G_μν = 4 × symmetric part of Re(Q)
        G_real = np.real(Q)
        G_fisher = 2 * (G_real + G_real.T)  # 4 × ½(Q + Q†)
        return G_fisher

    def compute_berry_curvature(
        self,
        psi: np.ndarray,
        dpsi: np.ndarray,
        dim: int = 4
    ) -> np.ndarray:
        """
        Compute the Berry curvature tensor Ω_μν.

        Ω_μν = 2 × Im(Q_μν)

        The Berry curvature is the antisymmetric part of the QGT,
        representing geometric/topological phase effects.

        From v8 Section 5.1:
        Q_μν = ¼G_μν^Fisher + (i/2)Ω_μν
        => Im(Q_μν) = ½Ω_μν
        => Ω_μν = 2 × Im(Q_μν)

        Args:
            psi: Quantum state vector |Ψ⟩ (normalized)
            dpsi: Array of state derivatives [∂_0Ψ, ∂_1Ψ, ∂_2Ψ, ∂_3Ψ]
            dim: Spacetime dimension (default 4)

        Returns:
            dim×dim real antisymmetric Berry curvature tensor Ω_μν
        """
        Q = self.compute_quantum_geometric_tensor(psi, dpsi, dim)

        # Ω_μν = 2 × Im(Q_μν)
        Omega = 2 * np.imag(Q)

        # Ensure antisymmetry (numerical precision)
        Omega = 0.5 * (Omega - Omega.T)

        return Omega

    def compute_berry_curvature_2form(
        self,
        psi: np.ndarray,
        dpsi: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Compute Berry curvature as a 2-form and its magnitude.

        For 4D spacetime, returns both the tensor and the Pfaffian
        (which relates to topological invariants).

        Args:
            psi: Quantum state vector
            dpsi: State derivatives

        Returns:
            Tuple of (Ω_μν tensor, Pfaffian magnitude)
        """
        Omega = self.compute_berry_curvature(psi, dpsi)

        # Pfaffian of 4×4 antisymmetric matrix: Pf² = det(Ω)
        # For antisymmetric A: det(A) = Pf(A)²
        det_Omega = np.linalg.det(Omega)
        pfaffian_sq = det_Omega
        pfaffian = np.sqrt(max(0, pfaffian_sq))  # Real for valid Ω

        return Omega, pfaffian

    def compute_complex_structure(self, G_fisher: np.ndarray) -> np.ndarray:
        """
        Compute the complex structure tensor J from the Fisher metric.

        From v8 Section 5.1, the Kähler structure satisfies:
        J² = -𝟙

        For a Kähler manifold, J is related to G and Ω by:
        Ω_μν = G_μα J^α_ν

        This computes J assuming the standard Kähler structure on
        projective Hilbert space CP^∞.

        Args:
            G_fisher: Fisher metric tensor (4×4)

        Returns:
            4×4 complex structure tensor J with J² = -𝟙
        """
        dim = G_fisher.shape[0]
        J = np.zeros((dim, dim))

        # Standard complex structure for spacetime:
        # In 4D, we pair coordinates: (t,x), (y,z) or similar
        # J acts as 90° rotation in each 2-plane

        # For Lorentzian signature from unitarity (v8 Section 5.3),
        # the time direction gets special treatment
        # J_0^1 = 1, J_1^0 = -1 (t-x plane, modified for Lorentzian)
        # J_2^3 = 1, J_3^2 = -1 (y-z plane)

        if dim >= 2:
            # t-x plane (Lorentzian: imaginary rotation)
            J[0, 1] = -1  # Modified for (−,+,+,+) signature
            J[1, 0] = 1

        if dim >= 4:
            # y-z plane (Euclidean rotation)
            J[2, 3] = 1
            J[3, 2] = -1

        return J

    def verify_kahler_structure(
        self,
        G_fisher: np.ndarray,
        Omega: np.ndarray,
        J: np.ndarray,
        tol: float = 1e-10
    ) -> Tuple[bool, dict]:
        """
        Verify the Kähler triplet relations from v8 Section 5.1.

        Checks:
        1. J² = -𝟙
        2. Ω_μν = G_μα J^α_ν
        3. G is symmetric, Ω is antisymmetric

        Args:
            G_fisher: Fisher metric (symmetric)
            Omega: Berry curvature (antisymmetric)
            J: Complex structure (J² = -𝟙)
            tol: Numerical tolerance

        Returns:
            Tuple of (all_valid, details_dict)
        """
        dim = G_fisher.shape[0]
        results = {}

        # Check 1: J² = -𝟙
        J_squared = J @ J
        identity = -np.eye(dim)
        j_squared_valid = np.allclose(J_squared, identity, atol=tol)
        results['J_squared_is_minus_identity'] = j_squared_valid
        results['J_squared_error'] = np.max(np.abs(J_squared - identity))

        # Check 2: Ω = G @ J
        Omega_computed = G_fisher @ J
        omega_valid = np.allclose(Omega, Omega_computed, atol=tol)
        results['Omega_equals_GJ'] = omega_valid
        results['Omega_GJ_error'] = np.max(np.abs(Omega - Omega_computed))

        # Check 3: G symmetric
        G_symmetric = np.allclose(G_fisher, G_fisher.T, atol=tol)
        results['G_symmetric'] = G_symmetric

        # Check 4: Ω antisymmetric
        Omega_antisymmetric = np.allclose(Omega, -Omega.T, atol=tol)
        results['Omega_antisymmetric'] = Omega_antisymmetric

        all_valid = (j_squared_valid and omega_valid and
                     G_symmetric and Omega_antisymmetric)

        return all_valid, results

    def compute_berry_phase(
        self,
        Omega: np.ndarray,
        area_element: np.ndarray
    ) -> float:
        """
        Compute the Berry phase for a closed loop.

        γ = ∮ A·dl = ∫∫ Ω·dS  (by Stokes' theorem)

        The Berry phase is the integral of the Berry curvature
        over the surface bounded by the loop.

        Args:
            Omega: Berry curvature 2-form
            area_element: Area 2-form dS_μν for the surface

        Returns:
            Berry phase γ (in radians)
        """
        # Contract Ω with area element: γ = ½ Ω_μν dS^μν
        phase = 0.5 * np.sum(Omega * area_element)
        return phase

    def lorentzian_signature_from_unitarity(self) -> dict:
        """
        Demonstrate how Lorentzian signature emerges from unitarity.

        From v8 Section 5.3-5.4:
        - Time evolution: ∂_t|Ψ⟩ = (-i/ℏ)H|Ψ⟩
        - The factor of -i means time is "imaginary direction"
        - Wick rotation: t = -iτ => dτ² = (i dt)² = -dt²
        - This gives signature (−,+,+,+)

        Returns:
            Dictionary explaining the causal chain
        """
        return {
            'causal_chain': [
                'Information conservation',
                'Unitarity (probability preservation)',
                'Time evolution U = exp(-iHt/ℏ)',
                'Factor of i in Schrödinger equation',
                'Wick rotation: t = -iτ',
                'Signature (−,+,+,+)'
            ],
            'key_insight': (
                'Lorentzian signature is the geometric manifestation '
                'of information conservation in the quantum substrate.'
            ),
            'wick_rotation': {
                'euclidean_time': 'τ',
                'physical_time': 't = -iτ',
                'line_element': 'ds² = -dt² + dx² + dy² + dz²'
            }
        }

    def compute_qgt_for_coherent_state(
        self,
        alpha: complex,
        sigma: float,
        position: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute QGT, Fisher metric, and Berry curvature for a coherent state.

        For a Gaussian coherent state:
        |Ψ(x)⟩ ∝ exp(-|x - x₀|²/(4σ²)) exp(i k·x)

        This is a common test case for the QGT framework.

        Args:
            alpha: Complex amplitude (encodes position and momentum)
            sigma: Coherence length
            position: 4-vector position x^μ

        Returns:
            Tuple of (Q_μν, G_fisher, Omega)
        """
        dim = 4

        # For a coherent state, the Fisher metric is diagonal
        # G_μν = δ_μν / σ² (in flat space approximation)
        G_fisher = np.diag([1.0 / sigma**2] * dim)

        # Lorentzian signature correction for time component
        G_fisher[0, 0] *= -1

        # Berry curvature for coherent states in phase space
        # Ω_μν comes from the symplectic structure
        Omega = np.zeros((dim, dim))

        # Standard symplectic form (position-momentum pairing)
        # Ω_tx = -Ω_xt represents time-position phase space
        if dim >= 2:
            omega_0 = np.abs(alpha)**2 / sigma**2
            Omega[0, 1] = omega_0
            Omega[1, 0] = -omega_0

        if dim >= 4:
            Omega[2, 3] = omega_0
            Omega[3, 2] = -omega_0

        # Construct full QGT: Q = ¼G + (i/2)Ω
        Q = 0.25 * G_fisher + 0.5j * Omega

        return Q, G_fisher, Omega
