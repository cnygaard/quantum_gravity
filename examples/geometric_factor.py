#!/usr/bin/env python3
"""
Geometric Factor Visualization: The Origin of π/2 in the Dark Matter Ratio

This script demonstrates why the factor π/2 appears in the dark matter prediction:
    M_DM/M_b = π/(2γ₀) ≈ 5.73

The factor π/2 is the maximum ratio of boundary geodesic (arc) to bulk geodesic (chord)
for points on a sphere, achieved at antipodal points.

Reference: theory/darkmatter.md
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize_scalar
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CONSTANTS


def xi_function(theta: np.ndarray) -> np.ndarray:
    """
    Compute the boundary-to-bulk path length ratio.

    ξ(θ) = L_boundary / L_bulk = θ / (2·sin(θ/2))

    Where:
    - L_boundary = Rθ (arc length on sphere)
    - L_bulk = 2R·sin(θ/2) (chord length through interior)

    Args:
        theta: Angular separation between points (radians)

    Returns:
        The ratio ξ(θ)
    """
    # Handle θ = 0 case (limit is 1)
    theta = np.atleast_1d(theta)
    result = np.ones_like(theta)
    mask = theta > 1e-10
    result[mask] = theta[mask] / (2 * np.sin(theta[mask] / 2))
    return result if len(result) > 1 else result[0]


def numerical_verification():
    """
    Numerically verify that ξ(θ) is maximized at θ = π with value π/2.

    Returns:
        dict: Verification results
    """
    # Define scalar version of xi for optimization
    def xi_scalar(t):
        if t < 1e-10:
            return 1.0
        return t / (2 * np.sin(t / 2))

    # Find maximum numerically
    result = minimize_scalar(
        lambda t: -xi_scalar(t),
        bounds=(0.01, np.pi),
        method='bounded'
    )

    theta_max = result.x
    xi_max = xi_scalar(theta_max)

    # Theoretical values
    theta_theoretical = np.pi
    xi_theoretical = np.pi / 2

    return {
        'theta_max_numerical': theta_max,
        'theta_max_theoretical': theta_theoretical,
        'theta_error': abs(theta_max - theta_theoretical),
        'xi_max_numerical': xi_max,
        'xi_max_theoretical': xi_theoretical,
        'xi_error': abs(xi_max - xi_theoretical),
        'xi_relative_error': abs(xi_max - xi_theoretical) / xi_theoretical
    }


def plot_xi_function(ax):
    """Plot ξ(θ) = θ/(2sin(θ/2)) showing maximum at π/2."""
    theta = np.linspace(0.01, np.pi, 1000)
    xi = xi_function(theta)

    ax.plot(theta, xi, 'b-', linewidth=2, label=r'$\xi(\theta) = \frac{\theta}{2\sin(\theta/2)}$')
    ax.axhline(y=np.pi/2, color='r', linestyle='--', linewidth=1.5, label=r'$\xi_{max} = \pi/2$')
    ax.axvline(x=np.pi, color='g', linestyle=':', linewidth=1.5, alpha=0.7)

    # Mark the maximum
    ax.plot(np.pi, np.pi/2, 'ro', markersize=10, zorder=5)
    ax.annotate(
        f'Maximum: $\\xi(\\pi) = \\pi/2 \\approx {np.pi/2:.4f}$',
        xy=(np.pi, np.pi/2),
        xytext=(np.pi - 1.2, np.pi/2 + 0.15),
        fontsize=10,
        arrowprops=dict(arrowstyle='->', color='black', lw=1.5)
    )

    ax.set_xlabel(r'Angular separation $\theta$ (radians)', fontsize=11)
    ax.set_ylabel(r'Path ratio $\xi(\theta) = L_{boundary}/L_{bulk}$', fontsize=11)
    ax.set_title(r'Boundary-to-Bulk Path Ratio', fontsize=12, fontweight='bold')
    ax.set_xlim(0, np.pi + 0.1)
    ax.set_ylim(0.9, 1.7)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_sphere_paths(ax):
    """3D visualization of boundary arc vs bulk chord on a sphere."""
    # Create sphere
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    R = 1.0
    x = R * np.outer(np.cos(u), np.sin(v))
    y = R * np.outer(np.sin(u), np.sin(v))
    z = R * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot transparent sphere
    ax.plot_surface(x, y, z, alpha=0.15, color='lightblue', edgecolor='none')

    # Antipodal points (north and south poles)
    ax.scatter([0], [0], [R], color='red', s=100, zorder=5, label='North pole')
    ax.scatter([0], [0], [-R], color='red', s=100, zorder=5, label='South pole')

    # Bulk path (diameter through center)
    ax.plot([0, 0], [0, 0], [-R, R], 'g-', linewidth=3, label=f'Bulk (diameter): $2R$')

    # Boundary path (semicircle on surface)
    theta_arc = np.linspace(0, np.pi, 100)
    x_arc = R * np.sin(theta_arc)
    y_arc = np.zeros_like(theta_arc)
    z_arc = R * np.cos(theta_arc)
    ax.plot(x_arc, y_arc, z_arc, 'b-', linewidth=3, label=f'Boundary (arc): $\\pi R$')

    # Add text annotation
    ax.text(0.6, 0, 0.5, f'Ratio = $\\pi/2$', fontsize=11, color='purple', fontweight='bold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Boundary vs Bulk Paths\n(Antipodal Points)', fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8)

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])


def plot_derivative(ax):
    """Plot dξ/dθ showing it's always positive (ξ is monotonically increasing)."""
    theta = np.linspace(0.01, np.pi - 0.01, 1000)

    # Numerical derivative
    dtheta = theta[1] - theta[0]
    xi = xi_function(theta)
    dxi_dtheta = np.gradient(xi, dtheta)

    ax.plot(theta, dxi_dtheta, 'purple', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax.fill_between(theta, 0, dxi_dtheta, alpha=0.3, color='purple')

    ax.set_xlabel(r'Angular separation $\theta$ (radians)', fontsize=11)
    ax.set_ylabel(r'$d\xi/d\theta$', fontsize=11)
    ax.set_title(r'Derivative: $d\xi/d\theta > 0$ (Always Increasing)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, np.pi)
    ax.set_xticks([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    ax.set_xticklabels(['0', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$'])
    ax.grid(True, alpha=0.3)

    ax.annotate(
        r'$\xi$ increases monotonically' + '\n' + r'$\Rightarrow$ max at $\theta = \pi$',
        xy=(np.pi/2, 0.15),
        fontsize=10,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    )


def plot_dark_matter_connection(ax):
    """Show how π/2 combines with γ₀ to give the dark matter ratio."""
    gamma_0 = CONSTANTS['gamma_0']
    xi = np.pi / 2
    dm_ratio_predicted = xi / gamma_0
    dm_ratio_observed = 5.36
    dm_ratio_error = 0.3

    # Bar chart
    categories = ['Holographic\nFactor', 'LQG\nFactor', 'Combined\nPrediction', 'Planck 2018\nObserved']
    values = [xi, 1/gamma_0, dm_ratio_predicted, dm_ratio_observed]
    colors = ['steelblue', 'forestgreen', 'darkorange', 'crimson']

    bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=1.2)

    # Add error bar for observation
    ax.errorbar(3, dm_ratio_observed, yerr=dm_ratio_error, fmt='none', color='black', capsize=5, linewidth=2)

    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
                f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add formula
    ax.text(0.5, 0.95,
            r'$\frac{M_{DM}}{M_b} = \frac{\pi/2}{\gamma_0} = \frac{1.571}{0.274} \approx 5.73$',
            transform=ax.transAxes, fontsize=12, ha='center', va='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('Dark Matter Ratio: From Geometry to Observation', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 7)
    ax.grid(True, alpha=0.3, axis='y')

    # Discrepancy annotation
    discrepancy = (dm_ratio_predicted - dm_ratio_observed) / dm_ratio_observed * 100
    ax.annotate(
        f'Discrepancy: {discrepancy:.1f}%\n(~1.2$\\sigma$)',
        xy=(2.5, 6),
        fontsize=9,
        ha='center',
        bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7)
    )


def main():
    """Main visualization function."""
    print("=" * 70)
    print("GEOMETRIC FACTOR VISUALIZATION: The Origin of pi/2")
    print("=" * 70)
    print()

    # Numerical verification
    print("1. NUMERICAL VERIFICATION")
    print("-" * 40)
    verification = numerical_verification()
    print(f"   Maximum of xi(theta) occurs at:")
    print(f"   - Numerical:    theta = {verification['theta_max_numerical']:.10f}")
    print(f"   - Theoretical:  theta = {verification['theta_max_theoretical']:.10f} (pi)")
    print(f"   - Error:        {verification['theta_error']:.2e}")
    print()
    print(f"   Maximum value of xi:")
    print(f"   - Numerical:    xi_max = {verification['xi_max_numerical']:.10f}")
    print(f"   - Theoretical:  xi_max = {verification['xi_max_theoretical']:.10f} (pi/2)")
    print(f"   - Relative error: {verification['xi_relative_error']:.2e}")
    print()

    # Dark matter calculation
    print("2. DARK MATTER RATIO")
    print("-" * 40)
    gamma_0 = CONSTANTS['gamma_0']
    xi = np.pi / 2
    dm_predicted = xi / gamma_0
    dm_observed = 5.36
    print(f"   xi (holographic factor):  {xi:.6f}")
    print(f"   gamma_0 (Immirzi):        {gamma_0:.6f}")
    print(f"   M_DM/M_b = xi/gamma_0:    {dm_predicted:.4f}")
    print(f"   Planck 2018 observed:     {dm_observed:.2f} +/- 0.3")
    print(f"   Discrepancy:              {(dm_predicted - dm_observed)/dm_observed*100:.1f}%")
    print()

    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'geometric_factor')
    os.makedirs(output_dir, exist_ok=True)

    # Create figure
    print("3. GENERATING VISUALIZATION")
    print("-" * 40)

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        r'The Geometric Origin of $\pi/2$ in the Dark Matter Ratio',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Panel 1: xi function
    ax1 = fig.add_subplot(2, 2, 1)
    plot_xi_function(ax1)

    # Panel 2: 3D sphere visualization
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    plot_sphere_paths(ax2)

    # Panel 3: Derivative (proof of monotonicity)
    ax3 = fig.add_subplot(2, 2, 3)
    plot_derivative(ax3)

    # Panel 4: Dark matter connection
    ax4 = fig.add_subplot(2, 2, 4)
    plot_dark_matter_connection(ax4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_path = os.path.join(output_dir, 'geometric_factor_pi_over_2.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {output_path}")

    # Also save as PDF for publication quality
    pdf_path = os.path.join(output_dir, 'geometric_factor_pi_over_2.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"   Saved: {pdf_path}")

    plt.show()

    print()
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("""
The factor pi/2 is NOT arbitrary but emerges from fundamental geometry:

  1. For two points on a sphere at angular separation theta:
     - Boundary path (arc):  L_boundary = R * theta
     - Bulk path (chord):    L_bulk = 2R * sin(theta/2)

  2. The ratio xi(theta) = theta / (2*sin(theta/2)) is MAXIMIZED
     at antipodal points (theta = pi), giving xi_max = pi/2

  3. Combined with the Immirzi parameter gamma_0 = 0.274 from LQG:
     M_DM/M_b = (pi/2) / gamma_0 = 5.73

  4. This matches Planck 2018 observations (5.36 +/- 0.3) within 7%

The simulator VERIFIES this geometric relationship numerically.
The factor pi/2 is exact mathematics - the ratio of semicircle to diameter.
""")


if __name__ == '__main__':
    main()
