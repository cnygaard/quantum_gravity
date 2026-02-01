"""
Tests for Page Curve implementation in black hole evaporation.

The Page curve describes the evolution of radiation entropy during unitary
black hole evaporation, resolving the information paradox.

Key physics:
- Black hole entropy: S_BH = pi * r_h^2 / (4 * l_p^2) where r_h = 2GM
- Mass evolution: M(t) = M_0 * (1 - t/t_evap)^(1/3)
- Radiation entropy: S_rad = min(entropy_released, current_entropy)
  where entropy_released = S_BH(0) - S_BH(t)

The Page time (when S_rad peaks) occurs at ~0.65 t_evap due to the
cubic-root mass dynamics, not 0.5 t_evap as would be the case for
linear evaporation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from constants import CONSTANTS


def compute_radiation_entropy(current_entropy: float, initial_entropy: float) -> float:
    """
    Compute radiation entropy using the Page curve model.

    For unitary black hole evaporation:
    S_rad = min(S_BH(0) - S_BH(t), S_BH(t))

    This gives the characteristic Page curve shape where radiation entropy
    rises until Page time, then decreases as the black hole approaches
    complete evaporation.

    Args:
        current_entropy: Current black hole entropy S_BH(t)
        initial_entropy: Initial black hole entropy S_BH(0)

    Returns:
        Radiation entropy S_rad
    """
    entropy_released = initial_entropy - current_entropy
    S_rad = min(entropy_released, current_entropy)
    return max(0.0, S_rad)


def compute_black_hole_entropy(mass: float) -> float:
    """
    Compute Bekenstein-Hawking entropy for a Schwarzschild black hole.

    S_BH = pi * r_h^2 / (4 * l_p^2)

    where r_h = 2GM is the horizon radius.

    In Planck units (G = l_p = 1): S_BH = pi * (2M)^2 / 4 = pi * M^2

    Args:
        mass: Black hole mass in Planck units

    Returns:
        Black hole entropy in Planck units
    """
    horizon_radius = 2 * CONSTANTS['G'] * mass
    return np.pi * horizon_radius**2 / (4 * CONSTANTS['l_p']**2)


def compute_evaporation_time(mass: float) -> float:
    """
    Compute Hawking evaporation time for a black hole.

    t_evap = 5120 * pi * G^2 * M^3 / (hbar * c^4)

    In Planck units: t_evap = 5120 * pi * M^3

    Args:
        mass: Initial black hole mass in Planck units

    Returns:
        Evaporation time in Planck units
    """
    return (5120 * np.pi * CONSTANTS['G']**2 * mass**3) / (
        CONSTANTS['hbar'] * CONSTANTS['c']**4
    )


def compute_mass_at_time(initial_mass: float, t: float, t_evap: float) -> float:
    """
    Compute black hole mass at time t during Hawking evaporation.

    M(t) = M_0 * (1 - t/t_evap)^(1/3)

    This follows from Stefan-Boltzmann emission with T ~ 1/M.

    Args:
        initial_mass: Initial black hole mass
        t: Current time
        t_evap: Total evaporation time

    Returns:
        Mass at time t, with Planck mass as minimum
    """
    if t >= t_evap:
        return CONSTANTS['m_p']
    mass = initial_mass * (1 - t / t_evap)**(1/3)
    return max(mass, CONSTANTS['m_p'])


class TestPageCurveFormula:
    """Tests for the Page curve formula: S_rad = min(S_BH(0) - S_BH(t), S_BH(t))"""

    def test_radiation_entropy_at_initial_time(self):
        """At t=0, no entropy has been radiated."""
        initial_entropy = 1000.0
        current_entropy = initial_entropy

        S_rad = compute_radiation_entropy(current_entropy, initial_entropy)

        assert S_rad == 0.0, "At t=0, radiation entropy should be zero"

    def test_radiation_entropy_before_page_time(self):
        """Before Page time, S_rad = S_BH(0) - S_BH(t) (entropy released)."""
        initial_entropy = 1000.0
        # Before Page time: current_entropy > initial_entropy / 2
        current_entropy = 800.0

        S_rad = compute_radiation_entropy(current_entropy, initial_entropy)
        entropy_released = initial_entropy - current_entropy

        # S_rad should equal entropy released (the smaller of the two)
        assert S_rad == entropy_released
        assert S_rad == 200.0

    def test_radiation_entropy_after_page_time(self):
        """After Page time, S_rad = S_BH(t) (current entropy)."""
        initial_entropy = 1000.0
        # After Page time: current_entropy < initial_entropy / 2
        current_entropy = 200.0

        S_rad = compute_radiation_entropy(current_entropy, initial_entropy)

        # S_rad should equal current entropy (the smaller of the two)
        assert S_rad == current_entropy
        assert S_rad == 200.0

    def test_radiation_entropy_at_page_time(self):
        """At Page time, entropy released equals current entropy."""
        initial_entropy = 1000.0
        # At Page time: current_entropy = initial_entropy / 2
        current_entropy = 500.0

        S_rad = compute_radiation_entropy(current_entropy, initial_entropy)
        entropy_released = initial_entropy - current_entropy

        assert S_rad == current_entropy
        assert S_rad == entropy_released
        assert S_rad == 500.0

    def test_radiation_entropy_non_negative(self):
        """Radiation entropy should never be negative."""
        # Edge case: current entropy exceeds initial (physically impossible
        # but testing defensive programming)
        initial_entropy = 1000.0
        current_entropy = 1100.0

        S_rad = compute_radiation_entropy(current_entropy, initial_entropy)

        assert S_rad >= 0.0, "Radiation entropy must be non-negative"


class TestPageCurveEvolution:
    """Tests for Page curve behavior during full black hole evaporation."""

    @pytest.fixture
    def evaporation_data(self):
        """
        Generate evaporation data for a test black hole.

        Returns tuple of (time_points, S_BH_history, S_rad_history, initial_S_BH, t_evap)
        """
        # Initial parameters
        M0 = 100.0  # Initial mass in Planck masses
        t_evap = compute_evaporation_time(M0)
        initial_S_BH = compute_black_hole_entropy(M0)

        # Simulate evolution with fine time resolution
        n_steps = 1000
        dt = t_evap * 0.99 / n_steps

        time_points = []
        S_BH_history = []
        S_rad_history = []

        t = 0.0
        while t < t_evap * 0.99:
            mass = compute_mass_at_time(M0, t, t_evap)
            if mass <= CONSTANTS['m_p']:
                break

            S_BH = compute_black_hole_entropy(mass)
            S_rad = compute_radiation_entropy(S_BH, initial_S_BH)

            time_points.append(t)
            S_BH_history.append(S_BH)
            S_rad_history.append(S_rad)

            t += dt

        return (
            np.array(time_points),
            np.array(S_BH_history),
            np.array(S_rad_history),
            initial_S_BH,
            t_evap
        )

    def test_max_radiation_entropy_ratio(self, evaporation_data):
        """
        Test that max S_rad / S_BH(0) is approximately 0.5.

        For unitary evolution, the maximum radiation entropy should be
        approximately half the initial black hole entropy. This is a
        key signature of information preservation.
        """
        _, _, S_rad_history, initial_S_BH, _ = evaporation_data

        max_S_rad = np.max(S_rad_history)
        ratio = max_S_rad / initial_S_BH

        # Allow tolerance for numerical discretization effects
        # and the fact that we use cubic-root mass dynamics
        assert 0.4 < ratio < 0.6, (
            f"Max S_rad/S_BH(0) = {ratio:.3f}, expected approximately 0.5 "
            "for unitary evolution"
        )

    def test_page_time_location(self, evaporation_data):
        """
        Test that Page time occurs at approximately 0.65 t_evap.

        Due to M(t) ~ (1 - t/t_evap)^(1/3) dynamics, the Page time
        (when S_rad peaks) occurs at ~0.65 t_evap, not 0.5 t_evap
        as would be expected for linear mass loss.

        This is because S_BH ~ M^2, and the cubic-root dynamics means
        entropy decreases more slowly initially.
        """
        time_points, _, S_rad_history, _, t_evap = evaporation_data

        # Find Page time as the time when S_rad reaches maximum
        page_idx = np.argmax(S_rad_history)
        page_time = time_points[page_idx]
        page_time_ratio = page_time / t_evap

        # The Page time should be around 0.65 t_evap for cubic-root dynamics
        # Allow wider tolerance since this depends on numerical resolution
        assert 0.55 < page_time_ratio < 0.75, (
            f"Page time at {page_time_ratio:.3f} t_evap, "
            "expected approximately 0.65 t_evap for M ~ (1-t/t_evap)^(1/3) dynamics"
        )

    def test_total_entropy_bounded(self, evaporation_data):
        """
        Test that total entropy (S_BH + S_rad) is bounded by initial S_BH.

        For unitary evolution, the total entropy should never exceed
        the initial black hole entropy. This is a consequence of
        information conservation.
        """
        _, S_BH_history, S_rad_history, initial_S_BH, _ = evaporation_data

        total_entropy = S_BH_history + S_rad_history
        max_total = np.max(total_entropy)
        ratio = max_total / initial_S_BH

        # Total entropy should be bounded by initial entropy
        # Allow small tolerance for numerical effects
        assert ratio <= 1.1, (
            f"Max total entropy / initial S_BH = {ratio:.3f}, "
            "should not exceed initial entropy (allowing 10% tolerance)"
        )

    def test_radiation_entropy_initial_zero(self, evaporation_data):
        """Radiation entropy should start at zero."""
        _, _, S_rad_history, _, _ = evaporation_data

        # First value should be approximately zero
        assert S_rad_history[0] < 0.01 * np.max(S_rad_history), (
            "Radiation entropy should start at approximately zero"
        )

    def test_radiation_entropy_decreases_after_page_time(self, evaporation_data):
        """After Page time, radiation entropy should decrease."""
        time_points, _, S_rad_history, _, _ = evaporation_data

        page_idx = np.argmax(S_rad_history)
        post_page_entropy = S_rad_history[page_idx:]

        # Check that entropy generally decreases after Page time
        # (allowing for some noise due to discretization)
        if len(post_page_entropy) > 10:
            # Compare early post-Page to late post-Page
            early_avg = np.mean(post_page_entropy[:5])
            late_avg = np.mean(post_page_entropy[-5:])
            assert late_avg < early_avg, (
                "Radiation entropy should decrease after Page time"
            )


class TestBlackHoleEntropyFormula:
    """Tests for the Bekenstein-Hawking entropy formula."""

    def test_entropy_scaling_with_mass(self):
        """Entropy should scale as M^2 (area law)."""
        masses = [10.0, 100.0, 1000.0]
        entropies = [compute_black_hole_entropy(m) for m in masses]

        # Check S ~ M^2 scaling
        for i in range(1, len(masses)):
            mass_ratio = masses[i] / masses[i-1]
            entropy_ratio = entropies[i] / entropies[i-1]
            expected_ratio = mass_ratio**2

            assert np.isclose(entropy_ratio, expected_ratio, rtol=0.01), (
                f"Entropy should scale as M^2: expected ratio {expected_ratio}, "
                f"got {entropy_ratio}"
            )

    def test_entropy_formula_explicit(self):
        """Test the explicit entropy formula S = pi * M^2."""
        mass = 100.0
        S_BH = compute_black_hole_entropy(mass)

        # In Planck units with G = l_p = 1:
        # S = pi * (2GM)^2 / (4 * l_p^2) = pi * M^2
        expected = np.pi * mass**2

        assert np.isclose(S_BH, expected, rtol=0.01), (
            f"S_BH = {S_BH}, expected pi * M^2 = {expected}"
        )


class TestMassEvolution:
    """Tests for black hole mass evolution during Hawking evaporation."""

    def test_initial_mass(self):
        """Mass at t=0 should equal initial mass."""
        M0 = 100.0
        t_evap = compute_evaporation_time(M0)

        mass = compute_mass_at_time(M0, 0, t_evap)

        assert np.isclose(mass, M0), "Mass at t=0 should equal initial mass"

    def test_mass_decreases_monotonically(self):
        """Mass should decrease monotonically during evaporation."""
        M0 = 100.0
        t_evap = compute_evaporation_time(M0)

        times = np.linspace(0, 0.9 * t_evap, 100)
        masses = [compute_mass_at_time(M0, t, t_evap) for t in times]

        for i in range(1, len(masses)):
            assert masses[i] <= masses[i-1], (
                f"Mass should decrease monotonically: M({times[i-1]}) = {masses[i-1]}, "
                f"M({times[i]}) = {masses[i]}"
            )

    def test_cubic_root_dynamics(self):
        """Test that mass follows M(t) = M_0 * (1 - t/t_evap)^(1/3)."""
        M0 = 100.0
        t_evap = compute_evaporation_time(M0)

        # Test at specific time fractions
        test_fractions = [0.1, 0.3, 0.5, 0.7, 0.9]

        for frac in test_fractions:
            t = frac * t_evap
            mass = compute_mass_at_time(M0, t, t_evap)
            expected = M0 * (1 - frac)**(1/3)

            assert np.isclose(mass, expected, rtol=0.01), (
                f"At t/t_evap = {frac}, mass = {mass}, "
                f"expected {expected}"
            )

    def test_planck_mass_floor(self):
        """Mass should not drop below Planck mass."""
        M0 = 10.0  # Small initial mass
        t_evap = compute_evaporation_time(M0)

        # Test beyond evaporation time
        mass = compute_mass_at_time(M0, t_evap * 1.1, t_evap)

        assert mass >= CONSTANTS['m_p'], (
            f"Mass should not drop below Planck mass: got {mass}"
        )


class TestEvaporationTime:
    """Tests for Hawking evaporation time calculation."""

    def test_evaporation_time_scaling(self):
        """Evaporation time should scale as M^3."""
        masses = [10.0, 100.0]
        t_evaps = [compute_evaporation_time(m) for m in masses]

        mass_ratio = masses[1] / masses[0]
        time_ratio = t_evaps[1] / t_evaps[0]
        expected_ratio = mass_ratio**3

        assert np.isclose(time_ratio, expected_ratio, rtol=0.01), (
            f"t_evap should scale as M^3: expected ratio {expected_ratio}, "
            f"got {time_ratio}"
        )

    def test_evaporation_time_formula(self):
        """Test the explicit evaporation time formula."""
        M0 = 100.0
        t_evap = compute_evaporation_time(M0)

        # In Planck units: t_evap = 5120 * pi * M^3
        expected = 5120 * np.pi * M0**3

        assert np.isclose(t_evap, expected, rtol=0.01), (
            f"t_evap = {t_evap}, expected 5120 * pi * M^3 = {expected}"
        )


class TestPageCurveShapeProperties:
    """Tests for qualitative properties of the Page curve shape."""

    @pytest.fixture
    def page_curve_data(self):
        """Generate Page curve data for shape analysis."""
        M0 = 100.0
        t_evap = compute_evaporation_time(M0)
        initial_S_BH = compute_black_hole_entropy(M0)

        n_steps = 500
        times = np.linspace(0, 0.99 * t_evap, n_steps)

        S_rad = []
        for t in times:
            mass = compute_mass_at_time(M0, t, t_evap)
            S_BH = compute_black_hole_entropy(mass)
            S_rad.append(compute_radiation_entropy(S_BH, initial_S_BH))

        return times / t_evap, np.array(S_rad), initial_S_BH

    def test_page_curve_is_unimodal(self, page_curve_data):
        """Page curve should have exactly one maximum (unimodal)."""
        time_fracs, S_rad, _ = page_curve_data

        # Find all local maxima
        local_maxima = []
        for i in range(1, len(S_rad) - 1):
            if S_rad[i] > S_rad[i-1] and S_rad[i] > S_rad[i+1]:
                local_maxima.append(i)

        # Should have exactly one clear maximum
        # (allow for some numerical noise at very small differences)
        assert len(local_maxima) >= 1, "Page curve should have at least one maximum"

        # The first significant maximum should be the only one
        max_val = np.max(S_rad)
        significant_maxima = [
            i for i in local_maxima
            if S_rad[i] > 0.9 * max_val
        ]
        assert len(significant_maxima) == 1, (
            f"Page curve should be unimodal, found {len(significant_maxima)} "
            "significant maxima"
        )

    def test_page_curve_starts_at_zero(self, page_curve_data):
        """Page curve should start at S_rad = 0."""
        _, S_rad, initial_S_BH = page_curve_data

        # First point should be essentially zero
        assert S_rad[0] / initial_S_BH < 0.01, (
            "Page curve should start at approximately zero"
        )

    def test_page_curve_ends_near_zero(self, page_curve_data):
        """Page curve should end near zero as black hole fully evaporates."""
        _, S_rad, initial_S_BH = page_curve_data

        # Last point should be small (black hole nearly gone)
        final_ratio = S_rad[-1] / initial_S_BH
        assert final_ratio < 0.1, (
            f"Page curve should end near zero, got S_rad/S_BH(0) = {final_ratio:.3f}"
        )

    def test_page_curve_monotonic_before_peak(self, page_curve_data):
        """Page curve should be monotonically increasing before the peak."""
        _, S_rad, _ = page_curve_data

        page_idx = np.argmax(S_rad)
        pre_peak = S_rad[:page_idx+1]

        # Check monotonic increase (allowing small numerical tolerance)
        for i in range(1, len(pre_peak)):
            assert pre_peak[i] >= pre_peak[i-1] - 1e-10, (
                f"Page curve should be monotonic before peak: "
                f"S_rad[{i-1}] = {pre_peak[i-1]}, S_rad[{i}] = {pre_peak[i]}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
