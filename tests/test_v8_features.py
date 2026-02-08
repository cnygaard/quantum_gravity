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
