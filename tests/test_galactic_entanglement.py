#!/usr/bin/env python

"""
Test for scale-adapted entanglement calculations at galactic scales.
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
from core.grid import AdaptiveGrid
from physics.observables import EntanglementObservable, RobustEntanglementObservable
from core.state import QuantumState
from constants import CONSTANTS

class MockGalacticState:
    """Mock state for testing galactic-scale entanglement calculations."""
    
    def __init__(self, galaxy_type, radius, mass, dark_matter_ratio=5.0, rotation_period=1.0):
        self.galaxy_type = galaxy_type
        self.radius = radius  # in meters
        self.mass = mass  # in kg
        self.dark_matter_ratio = dark_matter_ratio
        self.rotation_period = rotation_period
        self.time = 0.0
        self.grid = None  # Will be set by test
        
        # Add attributes needed by measurement methods
        self.basis_states = {0: np.ones(1000)}
        self.coefficients = {0: 1.0}
        
    def expectation_value(self, operator):
        """Mock expectation value calculation."""
        return 1.0

def setup_grid(size=1000):
    """Setup grid for entanglement tests."""
    grid = AdaptiveGrid(eps_threshold=1e-6)
    
    # Create points randomly distributed in a sphere
    points = []
    for _ in range(size):
        # Random direction
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        
        # Random radius with more points near center
        r = np.random.random()**(1/3)  # Cube root for uniform volume distribution
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points.append([x, y, z])
    
    # Set grid points
    grid.set_points(np.array(points))
    return grid

def test_galactic_entanglement_calculation():
    """Test entanglement calculations at galactic scales."""
    # Setup grid
    grid = setup_grid(1000)
    
    # Create region for entanglement calculation (half of the points)
    region_A = list(range(500))
    
    # Create entanglement observables
    ent_obs = EntanglementObservable(grid, region_A)
    robust_ent_obs = RobustEntanglementObservable(grid, region_A)
    
    # Setup galaxy states of different types and scales
    galaxy_types = ['dwarf', 'spiral', 'elliptical']
    
    # Use realistic galaxy parameters
    # Radius in meters, mass in kg
    galaxy_params = {
        'dwarf': {
            'radius': 3.086e19,  # ~1 kpc
            'mass': 2e37,         # ~10^7 solar masses
            'dark_matter_ratio': 10.0
        },
        'spiral': {
            'radius': 3.086e20,   # ~10 kpc
            'mass': 2e41,         # ~10^11 solar masses
            'dark_matter_ratio': 5.0
        },
        'elliptical': {
            'radius': 6.172e20,   # ~20 kpc
            'mass': 2e42,         # ~10^12 solar masses
            'dark_matter_ratio': 7.0
        }
    }
    
    # Test for each galaxy type
    for galaxy_type in galaxy_types:
        # Create galaxy state
        params = galaxy_params[galaxy_type]
        galaxy = MockGalacticState(
            galaxy_type=galaxy_type,
            radius=params['radius'],
            mass=params['mass'],
            dark_matter_ratio=params['dark_matter_ratio']
        )
        galaxy.grid = grid
        
        # Verify entanglement calculations work without errors
        try:
            # First try with regular EntanglementObservable
            coupling = ent_obs._compute_entanglement_coupling(0, 1, galaxy)
            assert coupling is not None, f"Entanglement coupling should not be None for {galaxy_type}"
            assert np.isfinite(coupling), f"Entanglement coupling should be finite for {galaxy_type}"
            
            # For realistic galaxy scales, this would previously fail due to numerical underflow
            print(f"{galaxy_type} galaxy with regular EntanglementObservable: coupling = {coupling}")
            
            # Now try with RobustEntanglementObservable
            robust_coupling = robust_ent_obs._compute_entanglement_coupling(0, 1, galaxy)
            assert robust_coupling is not None, f"Robust entanglement coupling should not be None for {galaxy_type}"
            assert np.isfinite(robust_coupling), f"Robust entanglement coupling should be finite for {galaxy_type}"
            
            print(f"{galaxy_type} galaxy with RobustEntanglementObservable: coupling = {robust_coupling}")
            
            # Try measuring entanglement
            result = robust_ent_obs.measure(galaxy)
            assert result is not None, f"Entanglement measurement should not be None for {galaxy_type}"
            assert np.isfinite(result.value), f"Entanglement value should be finite for {galaxy_type}"
            
            print(f"{galaxy_type} galaxy entanglement = {result.value} ± {result.uncertainty}")
            
        except Exception as e:
            pytest.fail(f"Entanglement calculation failed for {galaxy_type} galaxy: {str(e)}")

def test_entanglement_scaling():
    """Test that entanglement scales properly with galaxy size."""
    # Setup grid
    grid = setup_grid(1000)
    
    # Create region for entanglement calculation (half of the points)
    region_A = list(range(500))
    
    # Create entanglement observable
    robust_ent_obs = RobustEntanglementObservable(grid, region_A)
    
    # Create spiral galaxies of different sizes
    small_galaxy = MockGalacticState(
        galaxy_type='spiral',
        radius=3.086e19,   # ~1 kpc
        mass=2e37          # ~10^7 solar masses
    )
    small_galaxy.grid = grid
    
    medium_galaxy = MockGalacticState(
        galaxy_type='spiral',
        radius=3.086e20,   # ~10 kpc
        mass=2e41          # ~10^11 solar masses
    )
    medium_galaxy.grid = grid
    
    large_galaxy = MockGalacticState(
        galaxy_type='spiral',
        radius=9.258e20,   # ~30 kpc
        mass=2e42          # ~10^12 solar masses
    )
    large_galaxy.grid = grid
    
    # Calculate entanglement for each galaxy
    small_result = robust_ent_obs.measure(small_galaxy)
    medium_result = robust_ent_obs.measure(medium_galaxy)
    large_result = robust_ent_obs.measure(large_galaxy)
    
    # Verify entanglement values
    print(f"Small galaxy: {small_result.value}")
    print(f"Medium galaxy: {medium_result.value}")
    print(f"Large galaxy: {large_result.value}")
    
    # Entanglement should scale with galaxy size
    # But not decrease too dramatically due to our scale-adapted approach
    assert small_result.value > 0, "Small galaxy entanglement should be positive"
    assert medium_result.value > 0, "Medium galaxy entanglement should be positive"
    assert large_result.value > 0, "Large galaxy entanglement should be positive"
    
    # Verify that our scaling keeps reasonable values across orders of magnitude
    small_to_large_ratio = small_result.value / large_result.value
    print(f"Small-to-large entanglement ratio: {small_to_large_ratio:.4f}")
    assert 0.5 <= small_to_large_ratio <= 2.0, "Entanglement should scale reasonably with galaxy size"

if __name__ == "__main__":
    # Run tests
    test_galactic_entanglement_calculation()
    test_entanglement_scaling()
    print("All tests passed!")