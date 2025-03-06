#!/usr/bin/env python

"""
Direct comparison of grid point distributions for galaxy simulations.

This test provides a direct measurement of the geometry-entanglement error
for different grid point distributions without running full simulations.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
from examples.galaxy import GalaxySimulation
from __init__ import QuantumGravity, configure_logging
import logging
import json

def main():
    """Run direct comparison of grid distributions."""
    # Initialize logging
    configure_logging(simulation_type='grid_test_direct')
    
    # Create output directory
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    # Galaxy parameters
    galaxy_types = ['spiral', 'elliptical', 'dwarf']
    stellar_mass = 5e10  # solar masses
    radius = 15.0  # kiloparsecs
    dark_matter_ratio = 5.0
    
    # Reduce point count for faster simulation
    reduced_points = 100000  # Reduced for quicker testing
    
    # Store results
    results = {
        'distributions': {
            'standard': {
                'description': '30% core, 60% disk, 10% halo',
                'galaxies': {},
            },
            'alternative': {
                'description': '60% core, 30% disk, 10% halo',
                'galaxies': {},
            }
        },
        'summary': {}
    }
    
    # Compare for each galaxy type
    for galaxy_type in galaxy_types:
        print(f"\nTesting {galaxy_type.upper()} galaxy with different grid distributions")
        print("-" * 80)
        
        # Create shared QuantumGravity instance
        qg = QuantumGravity()
        qg.config.config['grid']['points_max'] = reduced_points
        
        # Standard distribution (~30% core)
        print(f"Creating {galaxy_type} galaxy with standard distribution (30% core)...")
        galaxy_standard = GalaxySimulation(
            stellar_mass=stellar_mass,
            radius=radius,
            galaxy_type=galaxy_type,
            bulge_fraction=0.2,
            dark_matter_ratio=dark_matter_ratio,
            core_concentration=0.3,  # Standard: ~30% core
            quantum_gravity=qg
        )
        
        # Alternative distribution (60% core)
        print(f"Creating {galaxy_type} galaxy with alternative distribution (60% core)...")
        galaxy_alternative = GalaxySimulation(
            stellar_mass=stellar_mass,
            radius=radius,
            galaxy_type=galaxy_type,
            bulge_fraction=0.2,
            dark_matter_ratio=dark_matter_ratio,
            core_concentration=0.6,  # Alternative: 60% core
            quantum_gravity=qg
        )
        
        # Directly measure geometry-entanglement relation
        print(f"Measuring geometry-entanglement metrics for standard distribution...")
        metrics_standard = galaxy_standard.verifier._verify_geometric_entanglement(galaxy_standard.qg.state)
        
        print(f"Measuring geometry-entanglement metrics for alternative distribution...")
        metrics_alternative = galaxy_alternative.verifier._verify_geometric_entanglement(galaxy_alternative.qg.state)
        
        # Store results
        results['distributions']['standard']['galaxies'][galaxy_type] = {
            'lhs': float(metrics_standard['lhs']),
            'rhs': float(metrics_standard['rhs']),
            'relative_error': float(metrics_standard['relative_error'])
        }
        
        results['distributions']['alternative']['galaxies'][galaxy_type] = {
            'lhs': float(metrics_alternative['lhs']),
            'rhs': float(metrics_alternative['rhs']),
            'relative_error': float(metrics_alternative['relative_error'])
        }
        
        # Print comparison
        print(f"\nResults for {galaxy_type.capitalize()} Galaxy:")
        print(f"Standard Distribution (30% core, 60% disk, 10% halo):")
        print(f"  LHS: {metrics_standard['lhs']:.6e}, RHS: {metrics_standard['rhs']:.6e}")
        print(f"  Relative Error: {metrics_standard['relative_error']:.6e}")
        
        print(f"Alternative Distribution (60% core, 30% disk, 10% halo):")
        print(f"  LHS: {metrics_alternative['lhs']:.6e}, RHS: {metrics_alternative['rhs']:.6e}")
        print(f"  Relative Error: {metrics_alternative['relative_error']:.6e}")
        
        improvement = (metrics_standard['relative_error'] - metrics_alternative['relative_error']) / metrics_standard['relative_error'] * 100
        print(f"Improvement: {improvement:.2f}%")
        
        results['distributions']['standard']['galaxies'][galaxy_type]['improvement_percent'] = float(improvement)
    
    # Calculate average improvement
    improvement_values = [
        results['distributions']['standard']['galaxies'][galaxy_type]['improvement_percent']
        for galaxy_type in galaxy_types
    ]
    avg_improvement = sum(improvement_values) / len(improvement_values)
    
    # Find best distribution for each galaxy type
    best_distributions = {}
    for galaxy_type in galaxy_types:
        std_error = results['distributions']['standard']['galaxies'][galaxy_type]['relative_error']
        alt_error = results['distributions']['alternative']['galaxies'][galaxy_type]['relative_error']
        
        if alt_error <= std_error:
            best_distributions[galaxy_type] = 'alternative'
        else:
            best_distributions[galaxy_type] = 'standard'
    
    # Overall recommendation
    if avg_improvement > 0:
        recommendation = "alternative"
        reason = f"reduces error by {avg_improvement:.2f}% on average"
    else:
        recommendation = "standard"
        reason = f"alternative increases error by {-avg_improvement:.2f}% on average"
    
    # Add summary to results
    results['summary'] = {
        'best_distribution_by_galaxy_type': best_distributions,
        'average_improvement_percent': float(avg_improvement),
        'recommendation': recommendation,
        'reason': reason
    }
    
    # Print overall results
    print("\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)
    
    print(f"\nBest Distribution by Galaxy Type:")
    for galaxy_type, dist in best_distributions.items():
        print(f"  {galaxy_type.capitalize()}: {dist} distribution")
    
    print(f"\nAverage Improvement: {avg_improvement:.2f}%")
    print(f"\nRECOMMENDATION: Use the {recommendation} distribution ({results['distributions'][recommendation]['description']})")
    print(f"Reason: {reason}")
    
    # Save results to file
    output_file = output_dir / "grid_distribution_comparison.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_file}")

if __name__ == "__main__":
    main()