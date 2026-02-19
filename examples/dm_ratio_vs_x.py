#!/usr/bin/env python3
"""
Dark Matter Ratio vs Patch Parameter x

Plots M_DM/M_b = (pi/(2*gamma_0)) * sinc(x) as a function of the patch
parameter x, marking the three candidate identifications from Section 2.4
of the v9.8 paper.

Reference: theory/quantum-gravity-proposal-v9.8.tex, Section 2.4
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import CONSTANTS


def dm_ratio_vs_x(x, gamma_0=0.274):
    """
    Compute M_DM/M_b = (pi/(2*gamma_0)) * sinc(x).

    sinc(x) = sin(x)/x for x > 0, 1 for x = 0.
    """
    x = np.atleast_1d(np.float64(x))
    result = np.ones_like(x)
    mask = np.abs(x) > 1e-15
    result[mask] = np.sin(x[mask]) / x[mask]
    return (np.pi / (2 * gamma_0)) * result


def main():
    gamma_0 = CONSTANTS['gamma_0']
    omega_m = CONSTANTS['omega_m_v8']
    omega_lambda = CONSTANTS['omega_lambda_v8']

    # Candidate x values
    x_A = np.arcsin(np.sqrt(omega_lambda))
    x_B = np.sqrt(omega_m)
    x_C = np.sqrt(1.5 * omega_m)
    x_exact = 0.630  # From paper Table 1

    candidates = {
        'A': (x_A, r'A: $\arcsin\!\sqrt{\Omega_\Lambda}$'),
        'B': (x_B, r'B: $\sqrt{\Omega_m}$'),
        'C': (x_C, r'C: $\sqrt{3\Omega_m/2}$'),
        'exact': (x_exact, 'Exact match'),
    }

    # Observed value (Planck 2018)
    dm_obs = 5.36
    dm_obs_err = 0.07  # 1-sigma from Omega_c/Omega_b

    # Compute curve
    x = np.linspace(0, np.pi / 2, 500)
    y = dm_ratio_vs_x(x, gamma_0)

    # -- Figure --
    fig, ax = plt.subplots(figsize=(7, 5))

    # Main curve
    ax.plot(x, y, 'k-', linewidth=2,
            label=r'$M_{DM}/M_b = \frac{\pi}{2\gamma_0}\,\mathrm{sinc}(x)$')

    # Observation band
    ax.axhspan(dm_obs - dm_obs_err, dm_obs + dm_obs_err,
               color='gold', alpha=0.35, label=rf'Planck 2018: ${dm_obs} \pm {dm_obs_err}$')
    ax.axhline(dm_obs, color='goldenrod', linewidth=1, linestyle='-')

    # Candidate markers
    colors = {'A': 'C0', 'B': 'C3', 'C': 'C2', 'exact': 'C4'}
    markers = {'A': 's', 'B': 'o', 'C': 'D', 'exact': '*'}
    offsets_y = {'A': -0.25, 'B': 0.35, 'C': 0.25, 'exact': -0.40}
    offsets_x = {'A': 0.02, 'B': 0.02, 'C': 0.15, 'exact': -0.08}

    for key, (xv, label_text) in candidates.items():
        yv_scalar = float(dm_ratio_vs_x(xv, gamma_0).item())

        # Vertical dashed line from x-axis to curve
        ax.plot([xv, xv], [ax.get_ylim()[0] if ax.get_ylim()[0] < yv_scalar else 4.0, yv_scalar],
                color=colors[key], linestyle=':', linewidth=1, alpha=0.6)

        # Marker on curve
        ms = 12 if key == 'exact' else 8
        ax.plot(xv, yv_scalar, markers[key], color=colors[key],
                markersize=ms, zorder=5, markeredgecolor='black', markeredgewidth=0.5)

        # Label
        ax.annotate(
            f'{label_text}\n$x = {xv:.3f}$, $M_{{DM}}/M_b = {yv_scalar:.2f}$',
            xy=(xv, yv_scalar),
            xytext=(xv + offsets_x[key], yv_scalar + offsets_y[key]),
            fontsize=8,
            color=colors[key],
            fontweight='bold' if key == 'B' else 'normal',
            arrowprops=dict(arrowstyle='->', color=colors[key], lw=1.0),
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                      edgecolor=colors[key], linewidth=0.5),
        )

    # Flat-space limit marker at x=0
    y0 = float(dm_ratio_vs_x(0.0, gamma_0).item())
    ax.plot(0, y0, 'kv', markersize=8, zorder=5)
    ax.annotate(
        rf'Flat space: $\pi/(2\gamma_0) = {y0:.2f}$',
        xy=(0, y0), xytext=(0.12, y0 + 0.08),
        fontsize=8, arrowprops=dict(arrowstyle='->', color='black', lw=1.0),
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8,
                  edgecolor='black', linewidth=0.5),
    )

    # Formatting
    ax.set_xlabel(r'Patch parameter $x = R/L$', fontsize=12)
    ax.set_ylabel(r'$M_{DM}/M_b$', fontsize=12)
    ax.set_title(r'Dark matter ratio vs.\ patch parameter', fontsize=13, fontweight='bold')
    ax.set_xlim(-0.05, np.pi / 2 + 0.05)
    ax.set_ylim(3.5, 6.2)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, np.pi / 2])
    ax.set_xticklabels(['0', '0.25', '0.5', '0.75', '1.0', '1.25', r'$\pi/2$'])
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(True, alpha=0.25)

    plt.tight_layout()

    # Save
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'results', 'geometric_factor')
    os.makedirs(output_dir, exist_ok=True)

    pdf_path = os.path.join(output_dir, 'dm_ratio_vs_x.pdf')
    png_path = os.path.join(output_dir, 'dm_ratio_vs_x.png')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    # Print candidate summary
    print()
    print("Candidate summary:")
    print(f"  {'Candidate':<12} {'x':>8} {'M_DM/M_b':>10} {'vs Planck':>10}")
    print(f"  {'Flat space':<12} {'0':>8} {y0:>10.2f} {(y0 - dm_obs)/dm_obs*100:>+9.1f}%")
    for key, (xv, _) in candidates.items():
        yv = float(dm_ratio_vs_x(xv, gamma_0).item())
        print(f"  {key:<12} {xv:>8.3f} {yv:>10.2f} {(yv - dm_obs)/dm_obs*100:>+9.1f}%")


if __name__ == '__main__':
    main()
