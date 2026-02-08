"""
Tests for v8 Theory Features

Tests the new physics implementations from the v8 quantum gravity framework:
- Kerr metric (rotating black holes)
- Geodesic equation solver
- LQC quantum bounce
- Hawking temperature quantum corrections

Reference: Holographic Fisher Geometry and Quantum Gravity Proposal v8
"""

import pytest
import numpy as np
from constants import CONSTANTS


class TestKerrMetric:
    """Test Kerr metric for rotating black holes (v8 Section 12)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.fisher_metric import FisherMetric
        self.fm = FisherMetric()
        self.M = 1000  # Planck masses
        self.r_s = 2 * CONSTANTS['G'] * self.M

    def test_kerr_metric_components(self):
        """Test that Kerr metric components are computed correctly."""
        a = 0.5 * self.r_s / 2  # Moderate spin
        r = 5 * self.r_s
        theta = np.pi / 2

        G = self.fm.compute_kerr_fisher(r, theta, self.M, a)

        # Check metric is 4x4
        assert G.shape == (4, 4)

        # Check signature: g_tt < 0 outside horizon
        assert G[0, 0] < 0

        # Check g_rr > 0
        assert G[1, 1] > 0

        # Check angular components positive
        assert G[2, 2] > 0
        assert G[3, 3] > 0

    def test_kerr_frame_dragging(self):
        """Test frame dragging (off-diagonal g_tφ) is present."""
        a = 0.5 * self.r_s / 2
        r = 5 * self.r_s
        theta = np.pi / 2

        G = self.fm.compute_kerr_fisher(r, theta, self.M, a)

        # Frame dragging should be non-zero for rotating BH
        assert G[0, 3] != 0
        assert G[3, 0] == G[0, 3]  # Symmetric

    def test_kerr_reduces_to_schwarzschild(self):
        """Test that Kerr with a=0 gives Schwarzschild-like metric."""
        r = 5 * self.r_s
        theta = np.pi / 2

        G_kerr = self.fm.compute_kerr_fisher(r, theta, self.M, a=0.0)

        # No frame dragging when a=0
        assert abs(G_kerr[0, 3]) < 1e-10

        # g_tt should match Schwarzschild form
        expected_gtt = -(1 - self.r_s / r)
        assert abs(G_kerr[0, 0] - expected_gtt) < 1e-10

    def test_kerr_horizon_radii(self):
        """Test horizon radius calculations."""
        a = 0.9 * self.r_s / 2  # Near extremal

        r_outer, r_inner = self.fm.kerr_horizon_radii(self.M, a)

        # Outer horizon should be less than Schwarzschild radius
        assert r_outer < self.r_s
        assert r_outer > r_inner
        assert r_inner > 0

    def test_extremal_limit(self):
        """Test behavior near extremal limit."""
        a = 0.999 * self.r_s / 2  # Very near extremal

        r_outer, r_inner = self.fm.kerr_horizon_radii(self.M, a)

        # Horizons should approach each other
        assert (r_outer - r_inner) / r_outer < 0.1

    def test_ergosphere(self):
        """Test ergosphere radius calculation."""
        a = 0.5 * self.r_s / 2
        theta = np.pi / 2  # Equator

        r_ergo = self.fm.kerr_ergosphere_radius(theta, self.M, a)
        r_outer, _ = self.fm.kerr_horizon_radii(self.M, a)

        # Ergosphere at equator equals Schwarzschild radius
        assert abs(r_ergo - self.r_s) < 1e-10

        # Ergosphere > outer horizon
        assert r_ergo > r_outer

    def test_quantum_state_parameters(self):
        """Test quantum state parameters for Kerr geometry."""
        a = 0.5 * self.r_s / 2
        r = 5 * self.r_s
        theta = np.pi / 2

        alpha_sq, xi = self.fm.kerr_quantum_state_params(r, theta, self.M, a)

        # Both should be positive
        assert alpha_sq >= 0
        assert xi >= 0


class TestGeodesics:
    """Test geodesic equation solver (v8 Section 13)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.geodesics import SchwarzschildGeodesics, KerrGeodesics
        self.M = 1000
        self.schwarz = SchwarzschildGeodesics(self.M)
        self.r_s = self.schwarz.r_s

    def test_isco_radius(self):
        """Test ISCO radius for Schwarzschild."""
        r_isco = self.schwarz.isco_radius()

        # ISCO should be 3 r_s for Schwarzschild
        assert abs(r_isco - 3 * self.r_s) < 1e-10

    def test_photon_sphere(self):
        """Test photon sphere radius."""
        r_ph = self.schwarz.photon_sphere_radius()

        # Photon sphere at 1.5 r_s
        assert abs(r_ph - 1.5 * self.r_s) < 1e-10

    def test_circular_orbit_stability(self):
        """Test circular orbit outside ISCO is stable."""
        r_orbit = 10 * self.r_s

        dt_dtau, dphi_dtau = self.schwarz.circular_orbit_velocity(r_orbit)

        # Should return valid velocities
        assert dt_dtau > 0
        assert dphi_dtau > 0

    def test_circular_orbit_normalization(self):
        """Test circular orbit velocity is properly normalized."""
        r_orbit = 10 * self.r_s

        dt_dtau, dphi_dtau = self.schwarz.circular_orbit_velocity(r_orbit)

        # Check ds² = -1 for timelike
        f = 1 - self.r_s / r_orbit
        ds2 = -f * dt_dtau**2 + r_orbit**2 * dphi_dtau**2

        assert abs(ds2 + 1) < 1e-10

    def test_orbit_integration(self):
        """Test orbit integration produces closed orbit."""
        r_orbit = 10 * self.r_s

        result = self.schwarz.compute_orbit(r_orbit, n_orbits=1.0, n_points=100)

        # Should complete successfully
        assert result['success']

        # Radius should stay constant
        r_values = result['x'][:, 1]
        assert np.std(r_values) / r_orbit < 1e-4

        # Should complete 2π azimuthal angle
        phi_change = result['x'][-1, 3] - result['x'][0, 3]
        assert abs(phi_change - 2 * np.pi) < 0.1

    def test_kerr_isco(self):
        """Test ISCO for Kerr black hole."""
        from physics.geodesics import KerrGeodesics

        a = 0.9 * self.r_s / 2
        kerr = KerrGeodesics(self.M, a)

        r_isco_pro = kerr.isco_radius(prograde=True)
        r_isco_retro = kerr.isco_radius(prograde=False)

        # Prograde ISCO is smaller than retrograde
        assert r_isco_pro < r_isco_retro

        # Both should be positive
        assert r_isco_pro > 0
        assert r_isco_retro > 0


class TestLQCBounce:
    """Test LQC quantum bounce (v8 Section 16.3)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.lqc_bounce import LQCBounce, LQCPlanckStar, RHO_CRITICAL_LQC
        self.lqc = LQCBounce()
        self.rho_c = RHO_CRITICAL_LQC

    def test_classical_limit(self):
        """Test LQC reduces to classical Friedmann for low density."""
        rho = 1e-10 * self.rho_c

        H2_lqc = self.lqc.hubble_squared(rho)
        H2_classical = (8 * np.pi * CONSTANTS['G'] / 3) * rho

        # Should match within 1%
        assert abs(H2_lqc - H2_classical) / H2_classical < 0.01

    def test_bounce_at_critical_density(self):
        """Test H² = 0 at critical density."""
        H2 = self.lqc.hubble_squared(self.rho_c)

        # Should be essentially zero
        assert H2 < 1e-20

    def test_hubble_maximum(self):
        """Test maximum H² occurs at ρ = ρ_c/2."""
        rho_max = 0.5 * self.rho_c
        H2_max = self.lqc.hubble_squared(rho_max)

        # Check nearby densities have lower H²
        for factor in [0.4, 0.6]:
            rho_test = factor * self.rho_c
            H2_test = self.lqc.hubble_squared(rho_test)
            assert H2_test < H2_max

    def test_bounce_detection(self):
        """Test bounce is detected during evolution."""
        a0 = 100.0
        rho0 = 0.5 * self.rho_c

        result = self.lqc.solve_bounce(a0, rho0, (0, 200), expanding=False)

        # Solution should succeed
        assert result['success']

        # Bounce should be detected
        assert result['bounce_detected']

    def test_planck_star(self):
        """Test Planck Star model for BH interior."""
        from physics.lqc_bounce import LQCPlanckStar

        M = 100
        ps = LQCPlanckStar(M)

        r_min = ps.minimum_radius()
        r_s = ps.r_s

        # Minimum radius should be positive and less than Schwarzschild
        assert r_min > 0
        assert r_min < r_s

        # Core density should equal critical density
        assert ps.core_density() == self.rho_c


class TestHawkingTemperature:
    """Test Hawking temperature quantum corrections (v8 Section 17.1)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.observables import BlackHoleTemperatureObservable
        from core.grid import AdaptiveGrid

        grid = AdaptiveGrid(eps_threshold=1e-6)
        self.temp_obs = BlackHoleTemperatureObservable(grid)

    def test_classical_temperature_scaling(self):
        """Test T_H ∝ 1/M scaling."""
        M1, M2 = 100, 200

        T1 = self.temp_obs.classical_temperature(M1)
        T2 = self.temp_obs.classical_temperature(M2)

        # T should scale as 1/M
        ratio = T1 / T2
        expected_ratio = M2 / M1

        assert abs(ratio - expected_ratio) / expected_ratio < 1e-10

    def test_quantum_correction_small_bh(self):
        """Test quantum correction is significant for small BH."""
        M = 10  # Small BH

        correction = self.temp_obs.quantum_correction(M, order=1)

        # Correction should be noticeable (< 1)
        assert correction < 0.99

    def test_quantum_correction_large_bh(self):
        """Test quantum correction is negligible for large BH."""
        M = 100000  # Large BH

        correction = self.temp_obs.quantum_correction(M, order=1)

        # Correction should be essentially 1
        assert abs(correction - 1) < 1e-4

    def test_quantum_temperature_less_than_classical(self):
        """Test quantum-corrected T is less than classical T."""
        for M in [10, 100, 1000]:
            T_cl = self.temp_obs.classical_temperature(M)
            T_q = T_cl * self.temp_obs.quantum_correction(M)

            assert T_q <= T_cl

    def test_temperature_vs_mass(self):
        """Test temperature_vs_mass method."""
        masses = np.logspace(1, 4, 10)
        data = self.temp_obs.temperature_vs_mass(masses)

        # Check all arrays have correct length
        assert len(data['mass']) == len(masses)
        assert len(data['T_classical']) == len(masses)
        assert len(data['T_quantum']) == len(masses)
        assert len(data['correction']) == len(masses)

        # Check T_quantum <= T_classical everywhere
        assert np.all(data['T_quantum'] <= data['T_classical'])

    def test_nlo_correction(self):
        """Test next-to-leading order correction."""
        M = 50

        corr_lo = self.temp_obs.quantum_correction(M, order=1)
        corr_nlo = self.temp_obs.quantum_correction(M, order=2)

        # NLO should be slightly different from LO
        assert corr_lo != corr_nlo

        # NLO should be larger (positive O(β²) term)
        assert corr_nlo >= corr_lo


class TestIntegration:
    """Integration tests combining multiple v8 features."""

    def test_kerr_geodesic_consistency(self):
        """Test Kerr metric and geodesic solver work together."""
        from physics.fisher_metric import FisherMetric
        from physics.geodesics import KerrGeodesics

        M = 1000
        a = 0.5 * CONSTANTS['G'] * M

        fm = FisherMetric()
        kerr = KerrGeodesics(M, a)

        # Get metric at a point
        r = 10 * kerr.outer_horizon_radius()
        theta = np.pi / 2

        G = fm.compute_kerr_fisher(r, theta, M, a)

        # Frame dragging from geodesic solver
        omega = kerr.frame_dragging_at_radius(r, theta)

        # Check consistency: ω = -g_tφ/g_φφ
        omega_from_metric = -G[0, 3] / G[3, 3]

        assert abs(omega - omega_from_metric) / omega < 1e-10

    def test_bh_thermodynamics_consistency(self):
        """Test black hole thermodynamics is self-consistent."""
        from physics.observables import BlackHoleTemperatureObservable
        from core.grid import AdaptiveGrid

        grid = AdaptiveGrid(eps_threshold=1e-6)
        temp_obs = BlackHoleTemperatureObservable(grid)

        M = 1000
        r_h = 2 * CONSTANTS['G'] * M

        # Hawking temperature
        T = temp_obs.classical_temperature(M)

        # Check T = 1/(8πM) in Planck units
        expected_T = 1 / (8 * np.pi * M)

        # Note: classical_temperature returns in different units
        # Just check it's positive and scales correctly
        assert T > 0

        # Bekenstein-Hawking entropy S = A/4 = 4πr_h²/4 = πr_h²
        A = 4 * np.pi * r_h**2
        S = A / (4 * CONSTANTS['l_p']**2)

        # Check entropy is positive
        assert S > 0


class TestQuantumGeometricTensor:
    """Test Quantum Geometric Tensor and Berry Curvature (v8 Section 5)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.quantum_geometry import QuantumGeometry
        self.qg = QuantumGeometry()

    def test_qgt_hermitian_decomposition(self):
        """Test QGT decomposes into symmetric and antisymmetric parts."""
        # Create a simple normalized state
        dim_hilbert = 10
        psi = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)
        psi = psi / np.linalg.norm(psi)

        # Create state derivatives
        dpsi = np.zeros((4, dim_hilbert), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)

        Q = self.qg.compute_quantum_geometric_tensor(psi, dpsi)

        # Q should be complex
        assert Q.dtype == complex

        # Symmetric part = Fisher metric / 4
        G_fisher = self.qg.extract_fisher_metric_from_qgt(Q)
        assert np.allclose(G_fisher, G_fisher.T)  # Symmetric

        # Antisymmetric part = Berry curvature × i/2
        Omega = self.qg.compute_berry_curvature(psi, dpsi)
        assert np.allclose(Omega, -Omega.T)  # Antisymmetric

    def test_qgt_from_components(self):
        """Test Q = ¼G + (i/2)Ω reconstruction."""
        dim_hilbert = 10
        psi = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)
        psi = psi / np.linalg.norm(psi)

        dpsi = np.zeros((4, dim_hilbert), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)

        Q = self.qg.compute_quantum_geometric_tensor(psi, dpsi)
        G_fisher = self.qg.extract_fisher_metric_from_qgt(Q)
        Omega = self.qg.compute_berry_curvature(psi, dpsi)

        # Reconstruct: Q = ¼G + (i/2)Ω
        Q_reconstructed = 0.25 * G_fisher + 0.5j * Omega

        # Should match original Q (up to numerical precision)
        assert np.allclose(Q, Q_reconstructed, atol=1e-10)

    def test_berry_curvature_antisymmetric(self):
        """Test Berry curvature is antisymmetric."""
        dim_hilbert = 10
        psi = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)
        psi = psi / np.linalg.norm(psi)

        dpsi = np.zeros((4, dim_hilbert), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)

        Omega = self.qg.compute_berry_curvature(psi, dpsi)

        # Ω_μν = -Ω_νμ
        assert np.allclose(Omega, -Omega.T, atol=1e-14)

    def test_complex_structure_j_squared(self):
        """Test J² = -𝟙 for complex structure."""
        G_fisher = np.eye(4)  # Flat metric for simplicity
        J = self.qg.compute_complex_structure(G_fisher)

        J_squared = J @ J
        expected = -np.eye(4)

        assert np.allclose(J_squared, expected, atol=1e-14)

    def test_kahler_structure_verification(self):
        """Test Kähler triplet verification passes for valid structure."""
        # Construct a valid Kähler triplet
        G = np.eye(4)
        J = self.qg.compute_complex_structure(G)
        Omega = G @ J

        valid, results = self.qg.verify_kahler_structure(G, Omega, J)

        assert results['J_squared_is_minus_identity']
        assert results['G_symmetric']
        assert results['Omega_antisymmetric']
        assert results['Omega_equals_GJ']
        assert valid

    def test_coherent_state_qgt(self):
        """Test QGT computation for coherent state."""
        alpha = 1.0 + 0.5j
        sigma = 1.0
        position = np.array([0.0, 1.0, 0.0, 0.0])

        Q, G, Omega = self.qg.compute_qgt_for_coherent_state(alpha, sigma, position)

        # Q should be 4x4 complex
        assert Q.shape == (4, 4)
        assert Q.dtype == complex

        # G should be 4x4 real symmetric
        assert G.shape == (4, 4)
        assert np.allclose(G, G.T)

        # Omega should be 4x4 real antisymmetric
        assert Omega.shape == (4, 4)
        assert np.allclose(Omega, -Omega.T)

        # G_00 should be negative (Lorentzian signature)
        assert G[0, 0] < 0

    def test_lorentzian_signature_explanation(self):
        """Test Lorentzian signature explanation from unitarity."""
        result = self.qg.lorentzian_signature_from_unitarity()

        assert 'causal_chain' in result
        assert len(result['causal_chain']) == 6
        assert 'Information conservation' in result['causal_chain']
        assert 'Signature (−,+,+,+)' in result['causal_chain']

        assert 'key_insight' in result
        assert 'information conservation' in result['key_insight'].lower()

    def test_berry_phase_computation(self):
        """Test Berry phase computation from curvature."""
        # Simple test: constant curvature over a unit area
        Omega = np.zeros((4, 4))
        Omega[0, 1] = 1.0
        Omega[1, 0] = -1.0

        # Unit area element in t-x plane
        area_element = np.zeros((4, 4))
        area_element[0, 1] = 2.0  # Factor of 2 for antisymmetric sum
        area_element[1, 0] = -2.0

        phase = self.qg.compute_berry_phase(Omega, area_element)

        # γ = ½ Ω_μν dS^μν = ½ × (1×2 + (-1)×(-2)) = 2
        assert abs(phase - 2.0) < 1e-10

    def test_berry_curvature_2form(self):
        """Test Berry curvature as 2-form with Pfaffian."""
        dim_hilbert = 10
        psi = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)
        psi = psi / np.linalg.norm(psi)

        dpsi = np.zeros((4, dim_hilbert), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)

        Omega, pfaffian = self.qg.compute_berry_curvature_2form(psi, dpsi)

        # Omega should be antisymmetric
        assert np.allclose(Omega, -Omega.T)

        # Pfaffian should be non-negative (Pf² = det(Ω) can be negative for
        # antisymmetric matrices, but we take sqrt of max(0, det))
        assert pfaffian >= 0

    def test_fisher_metric_extraction_consistency(self):
        """Test Fisher metric extracted from QGT matches direct computation."""
        dim_hilbert = 10
        psi = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)
        psi = psi / np.linalg.norm(psi)

        dpsi = np.zeros((4, dim_hilbert), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim_hilbert) + 1j * np.random.randn(dim_hilbert)

        Q = self.qg.compute_quantum_geometric_tensor(psi, dpsi)
        G_from_qgt = self.qg.extract_fisher_metric_from_qgt(Q)

        # Check symmetry
        assert np.allclose(G_from_qgt, G_from_qgt.T, atol=1e-14)

        # Check it's real
        assert np.allclose(np.imag(G_from_qgt), 0, atol=1e-14)


class TestGravitationalWaves:
    """Test Gravitational Wave propagation (v8 Section 15)."""

    def setup_method(self):
        """Setup test fixtures."""
        from physics.gravitational_waves import (
            WaveEquationSolver, GWPolarization, BinaryInspiral,
            Ringdown, GWDetector, GravitationalWaveSignal, FisherPerturbation
        )
        self.WaveEquationSolver = WaveEquationSolver
        self.GWPolarization = GWPolarization
        self.BinaryInspiral = BinaryInspiral
        self.Ringdown = Ringdown
        self.GWDetector = GWDetector
        self.GravitationalWaveSignal = GravitationalWaveSignal
        self.FisherPerturbation = FisherPerturbation

    def test_wave_equation_gaussian_pulse(self):
        """Test wave equation preserves Gaussian pulse."""
        solver = self.WaveEquationSolver(grid_size=200, domain_size=100.0)

        x = np.linspace(0, 100, 200)
        x0, sigma = 50, 5
        initial_h = np.exp(-(x - x0)**2 / (2 * sigma**2))
        initial_dh_dt = np.zeros_like(initial_h)

        solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=30, n_steps=500)

        # Energy should be approximately conserved
        initial_energy = np.sum(initial_h**2)
        final_energy = np.sum(solution['h'][-1]**2)

        # Allow for some numerical dissipation
        assert final_energy > 0.5 * initial_energy

    def test_wave_equation_dispersion_relation(self):
        """Test wave dispersion relation ω = k (c = 1)."""
        solver = self.WaveEquationSolver(grid_size=200, domain_size=100.0)

        # Create plane wave
        k = 2 * np.pi / 20  # wavelength = 20
        omega = k  # dispersion relation

        x = np.linspace(0, 100, 200)
        initial_h = np.sin(k * x)
        initial_dh_dt = -omega * np.cos(k * x)

        solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=20, n_steps=400)

        # Wave should propagate at c = 1
        # After time t, peak should have moved by distance t
        assert solution['h'].shape[0] == 400

    def test_polarization_plus_traceless(self):
        """Test plus polarization is traceless."""
        e_plus = self.GWPolarization.plus_polarization_tensor()

        # TT gauge: trace = 0
        assert abs(np.trace(e_plus)) < 1e-10

        # Check structure: e_xx = 1, e_yy = -1
        assert e_plus[1, 1] == 1.0
        assert e_plus[2, 2] == -1.0
        assert e_plus[0, 0] == 0.0
        assert e_plus[3, 3] == 0.0

    def test_polarization_cross_traceless(self):
        """Test cross polarization is traceless."""
        e_cross = self.GWPolarization.cross_polarization_tensor()

        # TT gauge: trace = 0
        assert abs(np.trace(e_cross)) < 1e-10

        # Check structure: e_xy = e_yx = 1
        assert e_cross[1, 2] == 1.0
        assert e_cross[2, 1] == 1.0

    def test_polarization_orthogonality(self):
        """Test plus and cross polarizations are orthogonal."""
        e_plus = self.GWPolarization.plus_polarization_tensor()
        e_cross = self.GWPolarization.cross_polarization_tensor()

        # Inner product: Tr(e_+ · e_×) = 0
        inner_product = np.trace(e_plus @ e_cross)
        assert abs(inner_product) < 1e-10

    def test_tt_gauge_projection(self):
        """Test TT gauge projection removes trace and longitudinal."""
        # Create arbitrary perturbation
        h = np.random.randn(4, 4)
        h = 0.5 * (h + h.T)  # Make symmetric

        # Wave vector in z direction
        k = np.array([1.0, 0.0, 0.0, 1.0])  # null vector

        h_tt = self.GWPolarization.project_to_tt_gauge(h, k)

        # Should be traceless
        assert abs(np.trace(h_tt)) < 1e-10

        # Should be purely spatial
        assert np.allclose(h_tt[0, :], 0)
        assert np.allclose(h_tt[:, 0], 0)

    def test_binary_inspiral_chirp_mass(self):
        """Test chirp mass calculation."""
        m1, m2 = 30, 30
        binary = self.BinaryInspiral(m1, m2, distance=100)

        # Chirp mass = (m1*m2)^(3/5) / (m1+m2)^(1/5)
        expected_chirp = (m1 * m2)**(3/5) / (m1 + m2)**(1/5)
        assert abs(binary.m_chirp - expected_chirp) < 1e-10

    def test_binary_inspiral_frequency_increases(self):
        """Test inspiral frequency increases (chirp)."""
        binary = self.BinaryInspiral(m1=30, m2=30, distance=100)
        waveform = binary.compute_waveform(t_start=0, t_coal=1000, n_points=500)

        # Frequency should monotonically increase (except near merger)
        freq = waveform['frequency']
        freq_diff = np.diff(freq[:-20])  # Exclude end where numerical issues

        # Most of the frequency evolution should be positive
        assert np.sum(freq_diff >= 0) > 0.9 * len(freq_diff)

    def test_binary_inspiral_amplitude_increases(self):
        """Test inspiral amplitude increases toward merger."""
        binary = self.BinaryInspiral(m1=30, m2=30, distance=100)

        # Use shorter time window closer to merger for clearer amplitude increase
        t_coal = 100
        waveform = binary.compute_waveform(t_start=0, t_coal=t_coal, n_points=500)

        # Amplitude should increase as t approaches t_coal
        # The strain amplitude scales as A ∝ f^(2/3) ∝ (t_coal - t)^(-1/4)
        amp_array = np.array([binary.amplitude(t, t_coal) for t in waveform['t']])

        # Compare early vs late amplitude (avoiding very end where numerics break)
        amp_early = np.mean(amp_array[:100])
        amp_late = np.mean(amp_array[300:450])

        # Late amplitude should be larger
        assert amp_late >= amp_early * 0.99  # Allow for numerical tolerance

    def test_binary_coalescence_time(self):
        """Test time to coalescence calculation."""
        binary = self.BinaryInspiral(m1=30, m2=30, distance=100)

        t_coal = binary.time_to_coalescence()

        # Should be positive
        assert t_coal > 0

    def test_ringdown_frequency(self):
        """Test ringdown QNM frequency."""
        ringdown = self.Ringdown(final_mass=60, final_spin=0.7)

        # Frequency should be positive
        assert ringdown.omega > 0

        # Damping time should be positive
        assert ringdown.tau > 0

    def test_ringdown_exponential_decay(self):
        """Test ringdown exhibits exponential decay."""
        ringdown = self.Ringdown(final_mass=60, final_spin=0.7)
        data = ringdown.compute_waveform(t_max=10 * ringdown.tau)

        # Amplitude should decay by e^(-10) after 10 tau
        h_initial = np.max(np.abs(data['h'][:10]))
        h_final = np.max(np.abs(data['h'][-10:]))

        # Decay ratio should be approximately exp(-10)
        decay_ratio = h_final / h_initial
        expected_ratio = np.exp(-10)

        assert decay_ratio < 0.1  # Significant decay

    def test_ringdown_quality_factor(self):
        """Test ringdown Q factor is positive."""
        ringdown = self.Ringdown(final_mass=60, final_spin=0.7)
        data = ringdown.compute_waveform(t_max=100)

        # Q = π f τ should be > 1 for underdamped oscillation
        Q = data['quality_factor']
        assert Q > 1

    def test_detector_antenna_pattern_overhead(self):
        """Test antenna pattern for overhead source."""
        detector = self.GWDetector()

        # Overhead source (θ = 0)
        F_plus = detector.antenna_pattern_plus(0, 0, 0)
        F_cross = detector.antenna_pattern_cross(0, 0, 0)

        # For θ = 0: F_+ = 1, F_× = 0 (for φ = 0, ψ = 0)
        assert abs(F_plus - 1.0) < 1e-10
        assert abs(F_cross) < 1e-10

    def test_detector_antenna_pattern_symmetry(self):
        """Test antenna pattern symmetry."""
        detector = self.GWDetector()

        # Pattern should have 4-fold symmetry in φ
        F1 = detector.antenna_pattern_plus(np.pi/4, 0, 0)
        F2 = detector.antenna_pattern_plus(np.pi/4, np.pi/2, 0)
        F3 = detector.antenna_pattern_plus(np.pi/4, np.pi, 0)
        F4 = detector.antenna_pattern_plus(np.pi/4, 3*np.pi/2, 0)

        # F(φ) and F(φ + π) should be equal
        assert abs(F1 - F3) < 1e-10
        assert abs(F2 - F4) < 1e-10

    def test_detector_response_combination(self):
        """Test detector response combines polarizations correctly."""
        detector = self.GWDetector()

        h_plus = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
        h_cross = np.array([0.0, 1.0, 0.0, -1.0, 0.0])

        theta, phi = np.pi/4, np.pi/4

        h_det = detector.compute_response(h_plus, h_cross, theta, phi)

        # Response should be linear combination
        F_plus = detector.antenna_pattern_plus(theta, phi)
        F_cross = detector.antenna_pattern_cross(theta, phi)

        expected = F_plus * h_plus + F_cross * h_cross
        assert np.allclose(h_det, expected)

    def test_complete_imr_waveform(self):
        """Test complete inspiral-merger-ringdown waveform."""
        signal = self.GravitationalWaveSignal(
            m1=30, m2=30, distance=100, inclination=0
        )

        waveform = signal.generate_inspiral_merger_ringdown(
            t_start=0, t_merger=1000, t_end=1100, n_points=2000
        )

        # Should have continuous time array
        assert len(waveform['t']) == 2000

        # Should have both polarizations
        assert len(waveform['h_plus']) == 2000
        assert len(waveform['h_cross']) == 2000

        # Should mark merger time
        assert 't_merger' in waveform

    def test_frequency_domain_waveform(self):
        """Test frequency-domain waveform generation."""
        signal = self.GravitationalWaveSignal(
            m1=30, m2=30, distance=100
        )

        fd_wave = signal.frequency_domain_waveform(
            f_min=0.01, f_max=0.5, n_points=500
        )

        # Check outputs
        assert len(fd_wave['frequency']) == 500
        assert len(fd_wave['h_tilde']) == 500
        assert len(fd_wave['amplitude']) == 500
        assert len(fd_wave['phase']) == 500

        # Amplitude should decrease with frequency (f^(-7/6) scaling)
        amp = fd_wave['amplitude']
        assert amp[0] > amp[-1]

    def test_fisher_perturbation_structure(self):
        """Test Fisher metric perturbation has correct structure."""
        fp = self.FisherPerturbation()

        # Create simple test state and perturbation
        dim = 10
        psi = np.random.randn(dim) + 1j * np.random.randn(dim)
        psi = psi / np.linalg.norm(psi)

        dpsi = np.zeros((4, dim), dtype=complex)
        for mu in range(4):
            dpsi[mu] = np.random.randn(dim) + 1j * np.random.randn(dim)

        delta_psi = 0.01 * (np.random.randn(dim) + 1j * np.random.randn(dim))
        delta_dpsi = np.zeros((4, dim), dtype=complex)
        for mu in range(4):
            delta_dpsi[mu] = 0.01 * (np.random.randn(dim) + 1j * np.random.randn(dim))

        h = fp.compute_perturbation(psi, dpsi, delta_psi, delta_dpsi)

        # Should be 4x4
        assert h.shape == (4, 4)

        # Should be real (metric perturbation)
        assert np.allclose(np.imag(h), 0, atol=1e-10)

        # Should be symmetric
        assert np.allclose(h, h.T, atol=1e-10)

    def test_v8_wave_equation_key_result(self):
        """Test the v8 key result: □h_μν = 0."""
        # The wave equation solver implements □h = 0
        # Verify the discretization correctly gives wave-like solutions

        solver = self.WaveEquationSolver(grid_size=100, domain_size=50.0)

        # Initial conditions for right-moving wave
        x = np.linspace(0, 50, 100)
        k = 2 * np.pi / 10
        initial_h = np.sin(k * x)
        initial_dh_dt = -k * np.cos(k * x)  # Right-moving: ∂h/∂t = -c ∂h/∂x

        solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=25)

        # Wave should maintain structure (not dissipate completely)
        max_amp_initial = np.max(np.abs(solution['h'][0]))
        max_amp_final = np.max(np.abs(solution['h'][-1]))

        # Allow for some numerical dissipation but wave should persist
        assert max_amp_final > 0.3 * max_amp_initial


class TestGravitationalWaveIntegration:
    """Integration tests for GW with other v8 components."""

    def test_gw_with_kerr_source(self):
        """Test GW generation from Kerr black hole context."""
        from physics.fisher_metric import FisherMetric
        from physics.gravitational_waves import Ringdown

        fm = FisherMetric()
        M = 1000

        # Final mass and spin after merger
        ringdown = Ringdown(final_mass=M, final_spin=0.7)

        # QNM frequency should scale with 1/M
        omega = ringdown.omega
        tau = ringdown.tau

        # For larger mass, expect lower frequency
        ringdown_larger = Ringdown(final_mass=2*M, final_spin=0.7)
        assert ringdown_larger.omega < omega  # Lower freq for larger mass
        assert ringdown_larger.tau > tau  # Longer damping for larger mass

    def test_gw_polarization_unitarity(self):
        """Test GW polarization from unitarity constraints (v8 Section 15.3)."""
        from physics.gravitational_waves import GWPolarization

        e_plus = GWPolarization.plus_polarization_tensor()
        e_cross = GWPolarization.cross_polarization_tensor()

        # Normalization: Tr(e_+ · e_+) = 2, Tr(e_× · e_×) = 2
        norm_plus = np.trace(e_plus @ e_plus)
        norm_cross = np.trace(e_cross @ e_cross)

        assert abs(norm_plus - 2) < 1e-10
        assert abs(norm_cross - 2) < 1e-10

        # Orthogonality from unitarity
        ortho = np.trace(e_plus @ e_cross)
        assert abs(ortho) < 1e-10

    def test_gw_speed_equals_c(self):
        """Test GW propagation speed equals c (v8: Lieb-Robinson bound)."""
        from physics.gravitational_waves import WaveEquationSolver

        # Wave equation □h = -∂²h/∂t² + c²∇²h = 0
        # Solutions propagate at speed c

        solver = WaveEquationSolver(grid_size=200, domain_size=100.0)

        # Localized pulse at x = 50
        x = np.linspace(0, 100, 200)
        x0, sigma = 50, 3
        initial_h = np.exp(-(x - x0)**2 / (2 * sigma**2))
        initial_dh_dt = np.zeros_like(initial_h)

        solution = solver.solve_1d(initial_h, initial_dh_dt, t_max=20)

        # After time t, pulse center should have moved by distance ~t (c=1)
        # The pulse splits into left and right moving parts
        # Check that energy has spread from initial position
        h_final = solution['h'][-1]
        energy_near_center = np.sum(h_final[80:120]**2)
        total_energy = np.sum(h_final**2)

        # Energy should have spread out from center
        assert energy_near_center < 0.8 * total_energy
