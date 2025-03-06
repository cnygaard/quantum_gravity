#!/usr/bin/env python

"""
Enhanced Grid Distribution Test Script

This script provides an efficient framework for testing different grid point
distribution strategies for galaxy simulations, with a focus on minimizing
the LHS/RHS error in geometric-entanglement verification.

It includes:
1. Standard distribution strategies (standard, alternative, even, custom)
2. Physics-informed strategies (entropy-weighted, dual-scale, etc.)
3. Options for different resolutions and test types
4. Comprehensive error analysis and visualization
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import json
import time
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime

from __init__ import QuantumGravity, configure_logging
from examples.galaxy import GalaxySimulation
from utils.test_grid_search import optimize_galaxy_grid_distribution

# Configure logging
configure_logging(simulation_type="grid_test", log_file="enhanced_grid_test")


class PhysicsInformedDistribution(GalaxySimulation):
    """Base class for physics-informed grid distributions."""
    
    def __init__(self, 
                 stellar_mass: float,
                 radius: float,
                 galaxy_type: str = 'spiral',
                 bulge_fraction: float = 0.2,
                 dark_matter_ratio: float = 5.0,
                 quantum_gravity: 'QuantumGravity' = None,
                 core_fraction: float = 0.5,
                 disk_fraction: float = 0.4,
                 halo_fraction: float = 0.1,
                 strategy_name: str = 'physics_informed'):
        """Initialize with specific core/disk/halo fractions."""
        # Store fractions
        self.core_fraction = core_fraction
        self.disk_fraction = disk_fraction
        self.halo_fraction = halo_fraction
        self.strategy_name = strategy_name
        
        # Initialize parent class
        super().__init__(
            stellar_mass=stellar_mass,
            radius=radius,
            galaxy_type=galaxy_type,
            bulge_fraction=bulge_fraction,
            dark_matter_ratio=dark_matter_ratio,
            quantum_gravity=quantum_gravity
        )
        
        # Log the strategy used
        logging.info(f"Created physics-informed galaxy with {strategy_name} distribution strategy")
    
    def _generate_galaxy_grid_points(self, n_points: int) -> np.ndarray:
        """Override to use specific distribution fractions."""
        points = []
        
        # Use fixed fraction distribution
        adaptive_bulge_fraction = self.core_fraction
        disk_fraction = self.disk_fraction
        halo_fraction = self.halo_fraction
        
        # Log the distribution
        logging.info(f"Using {self.strategy_name} distribution: {adaptive_bulge_fraction:.2f} core, {disk_fraction:.2f} disk, {halo_fraction:.2f} halo")
        
        # Convert fractions to actual point counts
        n_bulge = int(n_points * adaptive_bulge_fraction)
        n_disk = int(n_points * disk_fraction)
        n_halo = n_points - n_bulge - n_disk  # Ensure total is exactly n_points
        
        logging.info(f"{self.strategy_name} grid: {n_bulge} bulge points ({adaptive_bulge_fraction:.2f}), {n_disk} disk points, {n_halo} halo points")
        
        # Rest of point generation same as parent
        # Generate bulge points (spherical distribution)
        bulge_radius = self.radius * 0.1  # 10% of galaxy radius
        
        # Core concentration factor based on galaxy type
        core_concentration = 1.5
        if self.galaxy_type == 'elliptical':
            core_concentration = 2.0
        elif self.galaxy_type == 'dwarf':
            core_concentration = 2.5
            
        for i in range(n_bulge):
            # Use modified Hernquist profile with steeper central concentration
            r = bulge_radius * (np.cbrt(np.random.random()) / (1 - np.random.random()))**(1/core_concentration)
            if r > bulge_radius:
                r = bulge_radius * np.random.random()**(1/core_concentration)  # Fallback with concentration
                
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            points.append([x, y, z])
        
        # Generate disk points
        disk_scale_length = self.radius * 0.3 * 0.8  # Reduced scale length
        disk_scale_height = self.radius * 0.05  # Scale height
        
        # Split disk points between inner/outer regions
        inner_disk_fraction = 0.7
        n_inner_disk = int(n_disk * inner_disk_fraction)
        n_outer_disk = n_disk - n_inner_disk
        
        # Generate inner disk points
        for i in range(n_inner_disk):
            # Modified exponential disk profile with steeper inner profile
            r = disk_scale_length * np.random.exponential() * (1 - 0.3 * np.random.random())
            if r > self.radius * 0.5:  # Limit to inner half
                r = self.radius * 0.5 * np.random.random()  # Fallback
                
            phi = 2 * np.pi * np.random.random()
            z = disk_scale_height * np.random.exponential() * (1 if np.random.random() > 0.5 else -1)
            
            # Apply spiral arm perturbation for spiral galaxies
            if self.galaxy_type == 'spiral':
                n_arms = 2
                arm_phase = np.random.normal(0, 0.2)
                arm_strength = 0.4 * np.exp(-r / disk_scale_length)
                phi += arm_strength * np.sin(n_arms * np.log(r/disk_scale_length) + arm_phase)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = z
            
            points.append([x, y, z])
            
        # Generate outer disk points
        for i in range(n_outer_disk):
            # Standard exponential profile for outer disk
            r = self.radius * 0.5 + (self.radius * 0.5) * np.random.exponential() * 0.5
            if r > self.radius:
                r = self.radius * np.random.random()  # Fallback
                
            phi = 2 * np.pi * np.random.random()
            z = disk_scale_height * 1.5 * np.random.exponential() * (1 if np.random.random() > 0.5 else -1)
            
            # Apply spiral arm perturbation for spiral galaxies
            if self.galaxy_type == 'spiral':
                n_arms = 2
                arm_phase = np.random.normal(0, 0.2)
                arm_strength = 0.2 * np.exp(-r / disk_scale_length)
                phi += arm_strength * np.sin(n_arms * np.log(r/disk_scale_length) + arm_phase)
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            z = z
            
            points.append([x, y, z])
        
        # Generate halo points
        halo_scale = self.radius * 2  # Halo extends beyond visible galaxy
        
        # Use NFW-inspired profile with steeper inner concentration
        concentration = 1.2
        if self.galaxy_type == 'dwarf':
            concentration = 1.5  # Dwarf galaxies have more concentrated halos
        
        for i in range(n_halo):
            # Modified NFW-like profile for halo with adaptive concentration
            r = halo_scale * np.random.random()**(1/concentration)
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            points.append([x, y, z])
        
        return np.array(points)


class EntropyWeightedDistribution(PhysicsInformedDistribution):
    """Distribution that weights points according to entanglement profile."""
    
    def __init__(self, **kwargs):
        # Set distribution based on galaxy type
        galaxy_type = kwargs.get('galaxy_type', 'spiral')
        if galaxy_type == 'spiral':
            core_fraction = 0.55
            disk_fraction = 0.35
            halo_fraction = 0.10
        elif galaxy_type == 'elliptical':
            core_fraction = 0.65
            disk_fraction = 0.25
            halo_fraction = 0.10
        elif galaxy_type == 'dwarf':
            core_fraction = 0.75
            disk_fraction = 0.15
            halo_fraction = 0.10
        else:
            core_fraction = 0.60
            disk_fraction = 0.30
            halo_fraction = 0.10
            
        super().__init__(
            core_fraction=core_fraction,
            disk_fraction=disk_fraction,
            halo_fraction=halo_fraction,
            strategy_name='entropy_weighted',
            **kwargs
        )


class DualScaleDistribution(PhysicsInformedDistribution):
    """Distribution that balances both entanglement and information terms."""
    
    def __init__(self, **kwargs):
        # Use fixed distribution regardless of galaxy type
        super().__init__(
            core_fraction=0.50,
            disk_fraction=0.40,
            halo_fraction=0.10,
            strategy_name='dual-scale',
            **kwargs
        )


class AdaptiveVerificationDistribution(PhysicsInformedDistribution):
    """Distribution directly derived from verification functions."""
    
    def __init__(self, **kwargs):
        # Very high core concentration
        galaxy_type = kwargs.get('galaxy_type', 'spiral')
        if galaxy_type == 'spiral':
            core_fraction = 0.70
            disk_fraction = 0.25
            halo_fraction = 0.05
        elif galaxy_type == 'elliptical':
            core_fraction = 0.75
            disk_fraction = 0.20
            halo_fraction = 0.05
        else:
            core_fraction = 0.80
            disk_fraction = 0.15
            halo_fraction = 0.05
            
        super().__init__(
            core_fraction=core_fraction,
            disk_fraction=disk_fraction,
            halo_fraction=halo_fraction,
            strategy_name='adaptive_verification',
            **kwargs
        )


class EllipsoidalDistribution(PhysicsInformedDistribution):
    """Distribution using ellipsoidal shapes for better galaxy matching."""
    
    def __init__(self, **kwargs):
        # Depending on galaxy type
        galaxy_type = kwargs.get('galaxy_type', 'spiral')
        if galaxy_type == 'spiral':
            core_fraction = 0.55
            disk_fraction = 0.35
            halo_fraction = 0.10
        elif galaxy_type == 'elliptical':
            core_fraction = 0.60
            disk_fraction = 0.30
            halo_fraction = 0.10
        else:
            core_fraction = 0.70
            disk_fraction = 0.20
            halo_fraction = 0.10
            
        super().__init__(
            core_fraction=core_fraction,
            disk_fraction=disk_fraction,
            halo_fraction=halo_fraction,
            strategy_name='ellipsoidal',
            **kwargs
        )
        
    def _generate_galaxy_grid_points(self, n_points: int) -> np.ndarray:
        """Override to use ellipsoidal distributions."""
        points = []
        
        # Use fixed fraction distribution
        adaptive_bulge_fraction = self.core_fraction
        disk_fraction = self.disk_fraction
        halo_fraction = self.halo_fraction
        
        # Log the distribution
        logging.info(f"Using {self.strategy_name} distribution: {adaptive_bulge_fraction:.2f} core, {disk_fraction:.2f} disk, {halo_fraction:.2f} halo")
        
        # Convert fractions to actual point counts
        n_bulge = int(n_points * adaptive_bulge_fraction)
        n_disk = int(n_points * disk_fraction)
        n_halo = n_points - n_bulge - n_disk  # Ensure total is exactly n_points
        
        logging.info(f"Ellipsoidal grid: {n_bulge} bulge points ({adaptive_bulge_fraction:.2f}), {n_disk} disk points, {n_halo} halo points")
        
        # Modified point generation for bulge (ellipsoidal)
        bulge_radius = self.radius * 0.1
        
        # Set ellipsoid parameters based on galaxy type
        if self.galaxy_type == 'spiral':
            # Flattened bulge for spiral
            a, b, c = bulge_radius, bulge_radius, bulge_radius * 0.6
        elif self.galaxy_type == 'elliptical':
            # Triaxial ellipsoid for elliptical
            a, b, c = bulge_radius * 1.2, bulge_radius, bulge_radius * 0.8
        else:
            # Nearly spherical for dwarf
            a, b, c = bulge_radius, bulge_radius * 0.9, bulge_radius * 0.9
            
        # Generate bulge points using ellipsoidal distribution
        core_concentration = 1.5  # Concentration parameter
        for i in range(n_bulge):
            # Use rejection sampling for ellipsoidal distribution
            while True:
                x_norm = np.random.random()**(1/core_concentration) * 2 - 1
                y_norm = np.random.random()**(1/core_concentration) * 2 - 1
                z_norm = np.random.random()**(1/core_concentration) * 2 - 1
                
                if (x_norm/a)**2 + (y_norm/b)**2 + (z_norm/c)**2 <= 1:
                    break
                    
            x = x_norm * a
            y = y_norm * b
            z = z_norm * c
            
            points.append([x, y, z])
            
        # Generate disk points with modified scale parameters
        disk_scale_length = self.radius * 0.3  # Disk scale length
        disk_scale_height = self.radius * 0.05  # Scale height
        
        # Create adaptive exponential distribution
        inner_disk_fraction = 0.7
        n_inner_disk = int(n_disk * inner_disk_fraction)
        n_outer_disk = n_disk - n_inner_disk
        
        # Generate inner disk points with higher concentration
        for i in range(n_inner_disk):
            r = disk_scale_length * np.random.exponential()
            phi = 2 * np.pi * np.random.random()
            
            # Modified z distribution for better disk shape
            if self.galaxy_type == 'spiral':
                # Thinner disk for spiral
                z_scale = disk_scale_height * (1 - 0.5 * np.exp(-r/disk_scale_length))
            else:
                # Thicker disk for other types
                z_scale = disk_scale_height * (1 + 0.5 * np.exp(-r/disk_scale_length))
                
            z = z_scale * np.random.normal()
            
            # Apply spiral arm perturbation for spiral galaxies
            if self.galaxy_type == 'spiral':
                n_arms = 2
                arm_phase = np.random.normal(0, 0.2)
                arm_strength = 0.4 * np.exp(-r / disk_scale_length)
                phi_new = phi + arm_strength * np.sin(n_arms * np.log(r/disk_scale_length) + arm_phase)
            else:
                phi_new = phi
            
            x = r * np.cos(phi_new)
            y = r * np.sin(phi_new)
            
            points.append([x, y, z])
            
        # Generate outer disk points
        for i in range(n_outer_disk):
            r = self.radius * 0.5 + (self.radius * 0.5) * np.random.exponential() * 0.5
            phi = 2 * np.pi * np.random.random()
            z = disk_scale_height * 1.5 * np.random.normal()
            
            # Apply spiral arm perturbation
            if self.galaxy_type == 'spiral':
                n_arms = 2
                arm_phase = np.random.normal(0, 0.2)
                arm_strength = 0.2 * np.exp(-r / disk_scale_length)
                phi_new = phi + arm_strength * np.sin(n_arms * np.log(r/disk_scale_length) + arm_phase)
            else:
                phi_new = phi
            
            x = r * np.cos(phi_new)
            y = r * np.sin(phi_new)
            
            points.append([x, y, z])
            
        # Generate halo points with triaxial distribution
        halo_scale = self.radius * 2
        
        # Set halo shape based on galaxy type
        if self.galaxy_type == 'spiral':
            # Slightly flattened halo
            a, b, c = halo_scale, halo_scale, halo_scale * 0.8
        elif self.galaxy_type == 'elliptical':
            # Triaxial halo
            a, b, c = halo_scale * 1.1, halo_scale, halo_scale * 0.9
        else:
            # Nearly spherical
            a, b, c = halo_scale, halo_scale, halo_scale
            
        # Use concentration parameter
        concentration = 1.2
        
        for i in range(n_halo):
            # Rejection sampling for triaxial distribution
            while True:
                x_norm = np.random.random()**(1/concentration) * 2 - 1
                y_norm = np.random.random()**(1/concentration) * 2 - 1
                z_norm = np.random.random()**(1/concentration) * 2 - 1
                
                if (x_norm/a)**2 + (y_norm/b)**2 + (z_norm/c)**2 <= 1:
                    break
                    
            # Scale to halo size
            x = x_norm * a
            y = y_norm * b
            z = z_norm * c
            
            points.append([x, y, z])
            
        return np.array(points)


class EntropyInformationDistribution(PhysicsInformedDistribution):
    """Advanced distribution directly sampling the verification function profiles."""
    
    def __init__(self, **kwargs):
        super().__init__(
            core_fraction=0.60,  # Will be overridden in point generation
            disk_fraction=0.30,
            halo_fraction=0.10,
            strategy_name='entropy-information_weighted',
            **kwargs
        )
        
    def _generate_galaxy_grid_points(self, n_points: int) -> np.ndarray:
        """Directly sample points based on verification function profiles."""
        points = []
        
        # Maximum radius to consider
        max_radius = self.radius * 2.0
        
        # Define sampling density function based on verification profiles
        def density_function(r, theta, phi):
            # Normalize radius
            r_norm = r / self.radius
            
            # Entanglement profile (stronger in core, falls off with distance)
            entanglement = np.exp(-3 * r_norm)
            
            # Information profile (broader distribution)
            information = np.exp(-1 * r_norm)
            
            # Combined profile (weighted sum)
            if r_norm < 0.1:  # Core
                weight = 0.8 * entanglement + 0.2 * information
            elif r_norm < 1.0:  # Disk
                # Add spiral arm enhancement for spiral galaxies
                if self.galaxy_type == 'spiral':
                    n_arms = 2
                    arm_factor = 0.3 * np.cos(n_arms * phi + 3 * np.log(r_norm))**2
                    weight = 0.5 * entanglement + 0.3 * information + 0.2 * arm_factor
                else:
                    weight = 0.5 * entanglement + 0.5 * information
            else:  # Halo
                weight = 0.2 * entanglement + 0.8 * information
                
            # Apply galaxy-specific modifications
            if self.galaxy_type == 'spiral':
                # Enhance disk, suppress z-direction outside core
                if r_norm > 0.1:
                    disk_factor = np.exp(-5 * abs(np.cos(theta)))
                    weight *= disk_factor
            elif self.galaxy_type == 'elliptical':
                # More uniform distribution
                weight = 0.7 * weight + 0.3
                
            return weight
        
        # Use rejection sampling to generate points according to density
        accepted = 0
        attempts = 0
        max_attempts = n_points * 50  # Limit attempts to avoid infinite loop
        
        # Find approximate maximum density for normalization
        max_density = 0
        for _ in range(1000):
            r = np.random.random() * max_radius
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            density = density_function(r, theta, phi)
            max_density = max(max_density, density)
        
        # Add safety factor
        max_density *= 1.2
        
        # Generate points using rejection sampling
        while accepted < n_points and attempts < max_attempts:
            attempts += 1
            
            # Generate random point in spherical coordinates
            r = np.random.random() * max_radius
            theta = np.arccos(2 * np.random.random() - 1)
            phi = 2 * np.pi * np.random.random()
            
            # Calculate point density
            density = density_function(r, theta, phi)
            
            # Accept point with probability proportional to density
            if np.random.random() < density / max_density:
                # Convert to Cartesian
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                points.append([x, y, z])
                accepted += 1
                
        # If failed to generate enough points, fill with uniform distribution
        if accepted < n_points:
            remaining = n_points - accepted
            logging.warning(f"Rejection sampling generated only {accepted}/{n_points} points. Adding {remaining} uniform points.")
            
            for _ in range(remaining):
                r = np.random.random() * max_radius
                theta = np.arccos(2 * np.random.random() - 1)
                phi = 2 * np.pi * np.random.random()
                
                x = r * np.sin(theta) * np.cos(phi)
                y = r * np.sin(theta) * np.sin(phi)
                z = r * np.cos(theta)
                
                points.append([x, y, z])
        
        # Report effective distribution
        r_values = np.linalg.norm(np.array(points), axis=1)
        core_count = np.sum(r_values < 0.1 * self.radius)
        disk_count = np.sum((r_values >= 0.1 * self.radius) & (r_values < self.radius))
        halo_count = np.sum(r_values >= self.radius)
        
        core_fraction = core_count / n_points
        disk_fraction = disk_count / n_points
        halo_fraction = halo_count / n_points
        
        logging.info(f"Entropy-Information sampling resulted in: {core_fraction:.2f} core, {disk_fraction:.2f} disk, {halo_fraction:.2f} halo")
        
        return np.array(points)


def test_physics_distributions(
    galaxy_type: str = 'spiral',
    strategies: List[str] = None,
    comparison_only: bool = False,
    quick_test: bool = False):
    """Test physics-informed grid distributions against standard ones.
    
    Args:
        galaxy_type: Type of galaxy to test
        strategies: List of strategies to test (if None, test all)
        comparison_only: If True, only run standard/alternative comparison
        quick_test: If True, use minimal resolution and simulation time
    
    Returns:
        Dictionary of test results
    """
    start_time = time.time()
    
    # Configure galaxy parameters
    if galaxy_type == 'spiral':
        stellar_mass = 5e10
        radius = 15.0
        dm_ratio = 5.0
    elif galaxy_type == 'elliptical':
        stellar_mass = 1e11
        radius = 20.0
        dm_ratio = 7.0
    else:  # dwarf
        stellar_mass = 1e9
        radius = 5.0
        dm_ratio = 10.0
    
    # Set simulation parameters
    if quick_test:
        n_points = 50000
        t_final = 0.01
    else:
        n_points = 500000
        t_final = 0.05
    
    # Configure strategies to test
    if comparison_only:
        # Only compare standard vs alternative
        strategies = ['standard', 'alternative']
    elif strategies is None:
        # Test all strategies
        strategies = [
            'standard', 
            'alternative', 
            'entropy_weighted', 
            'dual_scale',
            'adaptive_verification',
            'ellipsoidal',
            'entropy_information'
        ]
    
    # Initialize quantum gravity instances
    qg_instances = {}
    for strategy in strategies:
        qg_instances[strategy] = QuantumGravity()
    
    # Print test configuration
    print(f"Testing distributions for {galaxy_type.upper()} galaxy")
    print(f"Strategies: {', '.join(strategies)}")
    print(f"Grid points: {n_points}, Simulation time: {t_final} rotation periods")
    print("-" * 80)
    
    # Initialize results dictionary
    results = {
        'galaxy_type': galaxy_type,
        'n_points': n_points,
        'simulation_time': t_final,
        'distributions': {}
    }
    
    # Create and test galaxies with different distributions
    galaxies = {}
    
    for strategy in strategies:
        print(f"Creating galaxy with {strategy} distribution...")
        
        # Create appropriate galaxy instance
        if strategy == 'standard':
            # Standard distribution - through original code
            galaxies[strategy] = GalaxySimulation(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy],
                distribution_strategy='standard'  # Force standard
            )
        elif strategy == 'alternative':
            # Alternative distribution 
            galaxies[strategy] = GalaxySimulation(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy],
                distribution_strategy='alternative'
            )
        elif strategy == 'entropy_weighted':
            # Entropy-weighted distribution
            galaxies[strategy] = EntropyWeightedDistribution(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy]
            )
        elif strategy == 'dual_scale':
            # Dual-scale distribution
            galaxies[strategy] = DualScaleDistribution(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy]
            )
        elif strategy == 'adaptive_verification':
            # Adaptive verification distribution
            galaxies[strategy] = AdaptiveVerificationDistribution(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy]
            )
        elif strategy == 'ellipsoidal':
            # Ellipsoidal distribution
            galaxies[strategy] = EllipsoidalDistribution(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy]
            )
        elif strategy == 'entropy_information':
            # Entropy-information weighted distribution
            galaxies[strategy] = EntropyInformationDistribution(
                stellar_mass=stellar_mass,
                radius=radius,
                galaxy_type=galaxy_type,
                dark_matter_ratio=dm_ratio,
                quantum_gravity=qg_instances[strategy]
            )
    
    # Run simulations and measure verification error
    for strategy, galaxy in galaxies.items():
        print(f"Running simulation with {strategy} distribution...")
        logging.info(f"Running galaxy simulation with {strategy} distribution")
        
        # Run simulation for short time
        galaxy.run_simulation(t_final)
        
        # Get verification metrics
        print(f"Triggering explicit verification for {strategy} distribution...")
        metrics = galaxy.verifier._verify_geometric_entanglement(galaxy.qg.state)
        
        # Store results
        results['distributions'][strategy] = {
            'lhs': metrics['lhs'],
            'rhs': metrics['rhs'],
            'relative_error': metrics['relative_error'],
            'description': strategy
        }
    
    # Find best distribution
    best_strategy = min(results['distributions'].items(), 
                        key=lambda x: x[1]['relative_error'])[0]
    results['best_distribution'] = best_strategy
    
    # Calculate improvements
    standard_error = results['distributions']['standard']['relative_error']
    results['improvements'] = {}
    
    for strategy, data in results['distributions'].items():
        if strategy != 'standard':
            improvement = (standard_error - data['relative_error']) / standard_error * 100
            results['improvements'][f'standard_to_{strategy}'] = improvement
    
    # Print results
    print("\nDistribution comparison results:")
    print(f"Best distribution: {best_strategy}")
    print("\nRelative errors:")
    for strategy, data in results['distributions'].items():
        print(f"  {strategy}: {data['relative_error']:.8f}")
    
    print("\nImprovements over standard:")
    for strategy, improvement in results['improvements'].items():
        target = strategy.split('_to_')[1]
        print(f"  vs {target}: {improvement:.6f}%")
    
    # Save results to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = f"results/grid_test_{galaxy_type}_{timestamp}.json"
    
    os.makedirs(os.path.dirname(result_file), exist_ok=True)
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to {result_file}")
    print(f"Total execution time: {time.time() - start_time:.1f} seconds")
    
    return results


def plot_distribution_comparison(results_file, output_file=None):
    """Plot comparison of distribution strategies."""
    # Load results
    with open(results_file) as f:
        results = json.load(f)
    
    # Extract data
    galaxy_type = results['galaxy_type']
    strategies = []
    errors = []
    lhs_values = []
    rhs_values = []
    
    for strategy, data in results['distributions'].items():
        strategies.append(strategy)
        errors.append(data['relative_error'])
        lhs_values.append(data['lhs'])
        rhs_values.append(data['rhs'])
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot relative errors
    bar_positions = np.arange(len(strategies))
    bar_colors = ['skyblue' if s != results['best_distribution'] else 'green' for s in strategies]
    
    ax1.bar(bar_positions, errors, color=bar_colors)
    ax1.set_xticks(bar_positions)
    ax1.set_xticklabels(strategies, rotation=45, ha='right')
    ax1.set_ylabel('Relative Error (LHS/RHS)')
    ax1.set_title(f'Distribution Strategy Comparison - {galaxy_type.capitalize()} Galaxy')
    
    # Add error values as text
    for i, v in enumerate(errors):
        ax1.text(i, v + 0.001, f"{v:.6f}", ha='center', va='bottom', fontsize=8)
    
    # Plot LHS/RHS values
    width = 0.35
    ax2.bar(bar_positions - width/2, lhs_values, width, label='LHS (Classical)')
    ax2.bar(bar_positions + width/2, rhs_values, width, label='RHS (Quantum)')
    ax2.set_xticks(bar_positions)
    ax2.set_xticklabels(strategies, rotation=45, ha='right')
    ax2.set_ylabel('Term Value')
    ax2.set_title('LHS vs RHS Values')
    ax2.legend()
    
    # Add text with improvements
    improvements_text = "Improvements over standard:\n"
    for strategy, improvement in results['improvements'].items():
        target = strategy.split('_to_')[1]
        improvements_text += f"{target}: {improvement:.6f}%\n"
    
    plt.figtext(0.5, 0.01, improvements_text, ha='center', fontsize=10, 
                bbox=dict(facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save or show
    if output_file:
        plt.savefig(output_file, dpi=150)
        print(f"Plot saved to {output_file}")
    else:
        plt.show()


def main():
    """Main function to run the enhanced grid test."""
    parser = argparse.ArgumentParser(description="Test physics-informed grid distributions")
    parser.add_argument('--galaxy_type', type=str, default='spiral',
                        choices=['spiral', 'elliptical', 'dwarf'],
                        help='Type of galaxy to test')
    parser.add_argument('--strategies', type=str, nargs='+',
                        help='Strategies to test (if not specified, test all)')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick test with reduced resolution')
    parser.add_argument('--comparison_only', action='store_true',
                        help='Only compare standard vs alternative')
    parser.add_argument('--plot', action='store_true',
                        help='Plot results after testing')
    
    args = parser.parse_args()
    
    # Run test
    results = test_physics_distributions(
        galaxy_type=args.galaxy_type,
        strategies=args.strategies,
        comparison_only=args.comparison_only,
        quick_test=args.quick
    )
    
    # Plot if requested
    if args.plot:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"results/grid_test_{args.galaxy_type}_{timestamp}.json"
        plot_file = f"results/grid_test_{args.galaxy_type}_{timestamp}.png"
        
        # Check if results file exists, otherwise use most recent
        if not os.path.exists(results_file):
            # Find most recent results file
            result_files = sorted(Path('results').glob(f'grid_test_{args.galaxy_type}_*.json'))
            if result_files:
                results_file = str(result_files[-1])
                plot_file = str(results_file).replace('.json', '.png')
        
        plot_distribution_comparison(results_file, plot_file)


if __name__ == "__main__":
    main()