#!/usr/bin/env python

"""
Evolutionary comparison of grid point distributions for galaxy simulations.

This test runs both standard and alternative grid distributions through
a longer evolution period to see if differences emerge over time.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import matplotlib.pyplot as plt
from examples.galaxy import GalaxySimulation
from __init__ import QuantumGravity, configure_logging
import logging
import json

def run_comparison():
    """Run evolutionary comparison of grid distributions."""
    # Initialize logging
    configure_logging(simulation_type='grid_evolution_test')
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Focus on spiral galaxy for more detailed testing
    galaxy_type = 'spiral'
    stellar_mass = 5e10  # solar masses
    radius = 15.0  # kiloparsecs
    dark_matter_ratio = 5.0
    
    # Reduce point count for faster simulation
    reduced_points = 100000
    
    # Evolution parameters
    t_final = 0.2  # 20% of rotation period
    n_checkpoints = 10  # Number of measurement points
    
    print(f"\nRunning evolutionary comparison for {galaxy_type.upper()} galaxy")
    print("-" * 80)
    
    # Create shared QuantumGravity instance
    qg = QuantumGravity()
    qg.config.config['grid']['points_max'] = reduced_points
    
    # Create galaxies with different grid distributions
    print(f"Creating galaxy with standard distribution (30% core)...")
    galaxy_standard = GalaxySimulation(
        stellar_mass=stellar_mass,
        radius=radius,
        galaxy_type=galaxy_type,
        bulge_fraction=0.2,
        dark_matter_ratio=dark_matter_ratio,
        core_concentration=0.3,  # Standard: ~30% core
        quantum_gravity=qg
    )
    
    print(f"Creating galaxy with alternative distribution (60% core)...")
    galaxy_alternative = GalaxySimulation(
        stellar_mass=stellar_mass,
        radius=radius,
        galaxy_type=galaxy_type,
        bulge_fraction=0.2,
        dark_matter_ratio=dark_matter_ratio,
        core_concentration=0.6,  # Alternative: 60% core
        quantum_gravity=qg
    )
    
    # Store evolution results
    time_points = np.linspace(0, t_final, n_checkpoints)
    standard_errors = []
    alternative_errors = []
    
    # Initial measurements
    print("Taking initial measurements...")
    metrics_standard = galaxy_standard.verifier._verify_geometric_entanglement(galaxy_standard.qg.state)
    metrics_alternative = galaxy_alternative.verifier._verify_geometric_entanglement(galaxy_alternative.qg.state)
    
    standard_errors.append(metrics_standard['relative_error'])
    alternative_errors.append(metrics_alternative['relative_error'])
    
    # Run simulations with measurement checkpoints
    dt = t_final / (n_checkpoints - 1)
    
    for i in range(1, n_checkpoints):
        current_time = i * dt
        
        # Evolve standard galaxy
        print(f"Evolving standard distribution to t = {current_time:.3f} rotation periods...")
        galaxy_standard.run_simulation(t_final=current_time, dt=dt/10)
        
        # Measure standard verification
        metrics_standard = galaxy_standard.verifier._verify_geometric_entanglement(galaxy_standard.qg.state)
        standard_errors.append(metrics_standard['relative_error'])
        
        # Evolve alternative galaxy
        print(f"Evolving alternative distribution to t = {current_time:.3f} rotation periods...")
        galaxy_alternative.run_simulation(t_final=current_time, dt=dt/10)
        
        # Measure alternative verification
        metrics_alternative = galaxy_alternative.verifier._verify_geometric_entanglement(galaxy_alternative.qg.state)
        alternative_errors.append(metrics_alternative['relative_error'])
        
        # Print intermediate results
        print(f"\nResults at t = {current_time:.3f} rotation periods:")
        print(f"Standard Distribution Error: {metrics_standard['relative_error']:.6e}")
        print(f"Alternative Distribution Error: {metrics_alternative['relative_error']:.6e}")
        
        improvement = ((metrics_standard['relative_error'] - metrics_alternative['relative_error']) / 
                      metrics_standard['relative_error'] * 100)
        print(f"Improvement: {improvement:.2f}%")
    
    # Determine overall better distribution
    avg_std_error = np.mean(standard_errors)
    avg_alt_error = np.mean(alternative_errors)
    avg_improvement = (avg_std_error - avg_alt_error) / avg_std_error * 100
    
    # Prepare results
    results = {
        "evolution": {
            "time_points": list(time_points),
            "standard_errors": standard_errors,
            "alternative_errors": alternative_errors
        },
        "summary": {
            "avg_standard_error": float(avg_std_error),
            "avg_alternative_error": float(avg_alt_error),
            "avg_improvement": float(avg_improvement),
            "recommendation": "alternative" if avg_improvement > 0 else "standard",
            "reason": f"{'Reduces' if avg_improvement > 0 else 'Increases'} error by {abs(avg_improvement):.2f}% on average"
        }
    }
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(time_points, standard_errors, 'b-o', label='Standard (30% core)')
    plt.plot(time_points, alternative_errors, 'r-o', label='Alternative (60% core)')
    plt.xlabel('Time (rotation periods)')
    plt.ylabel('Relative Error')
    plt.title('Evolution of Geometry-Entanglement Error by Grid Distribution')
    plt.grid(True)
    plt.legend()
    
    plt.text(0.5, 0.01, 
             f"Average Improvement: {avg_improvement:.2f}%\n"
             f"Recommendation: {results['summary']['recommendation'].capitalize()} distribution"
             f" ({results['summary']['reason']})",
             ha='center', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    plot_file = output_dir / "grid_distribution_evolution.png"
    plt.savefig(plot_file, dpi=150)
    print(f"\nPlot saved to {plot_file}")
    
    # Save results to JSON
    json_file = output_dir / "grid_distribution_evolution.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {json_file}")
    
    # Print final recommendation
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATION")
    print("=" * 80)
    print(f"Based on evolution over {t_final} rotation periods:")
    print(f"Standard Distribution Average Error: {avg_std_error:.6e}")
    print(f"Alternative Distribution Average Error: {avg_alt_error:.6e}")
    print(f"Average Improvement: {avg_improvement:.2f}%")
    print(f"\nRECOMMENDATION: Use the {results['summary']['recommendation']} distribution")
    print(f"Reason: {results['summary']['reason']}")
    
    return results

if __name__ == "__main__":
    run_comparison()