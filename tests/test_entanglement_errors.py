#!/usr/bin/env python

"""
Test that verifies the fixes for entanglement calculation errors at galactic scales.
Specifically checks for resolution of:
1. "ARPACK error -9: Starting vector is zero"
2. "Matrix too sparse for reliable eigendecomposition"
3. "Initial density matrix construction failed"
"""

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pytest
import numpy as np
import logging
from scipy.sparse import linalg as sparse_linalg
from core.grid import AdaptiveGrid
from physics.observables import EntanglementObservable, RobustEntanglementObservable
from core.state import QuantumState
from constants import CONSTANTS
from utils.io import MeasurementResult

# Configure logging to track error messages
logging.basicConfig(level=logging.INFO)

class MockDwarfGalaxy:
    """Mock state for a dwarf galaxy with realistic parameters."""
    
    def __init__(self):
        self.galaxy_type = 'dwarf'
        self.radius = 5 * 3.086e19  # 5 kpc (in meters)
        self.mass = 2e37  # ~10^7 solar masses (in kg)
        self.dark_matter_ratio = 10.0
        self.rotation_period = 2e15  # ~100 million years (in seconds)
        self.time = 0.0
        self.grid = None  # Will be set by test
        
        # Add attributes needed by measurement methods
        self.basis_states = {0: np.ones(1000)}
        self.coefficients = {0: 1.0}
        
    def expectation_value(self, operator):
        """Mock expectation value calculation."""
        return 1.0

def test_entanglement_error_fixes():
    """
    Test that our scale-adapted implementation fixes the specific entanglement errors
    mentioned in the task.
    """
    print("\nTesting entanglement error fixes...")
    
    # Create a realistic grid size for a dwarf galaxy
    n_points = 10000
    
    # Setup grid with realistic distances for a dwarf galaxy (5 kpc)
    grid = AdaptiveGrid(eps_threshold=1e-6)
    
    # Set l_p to a very small value to force numerical issues
    # This would normally cause the errors we're trying to fix
    grid.l_p = CONSTANTS['l_p']
    
    # Create points randomly distributed in a sphere of radius 5 kpc
    dwarf_radius = 5 * 3.086e19  # 5 kpc in meters
    points = []
    for _ in range(n_points):
        # Random direction
        theta = np.arccos(2 * np.random.random() - 1)
        phi = 2 * np.pi * np.random.random()
        
        # Random radius with non-uniform distribution (more points in center)
        r = dwarf_radius * np.random.random()**(1/3)
        
        # Convert to Cartesian coordinates
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        points.append([x, y, z])
    
    # Set grid points
    grid.set_points(np.array(points))
    
    # Create galaxy state
    galaxy = MockDwarfGalaxy()
    galaxy.grid = grid
    
    # Create region for entanglement calculation
    region_A = list(range(n_points // 2))
    
    # Create original implementation of entanglement observable
    original_ent_obs = EntanglementObservable(grid, region_A)
    
    # Create robust implementation
    robust_ent_obs = RobustEntanglementObservable(grid, region_A)
    
    # Test 1: Check that our coupling function doesn't produce zero values
    # This would prevent "ARPACK error -9: Starting vector is zero"
    i, j = 0, n_points // 2  # Pick two distant points
    
    # Calculate coupling with original and robust implementations
    original_coupling = original_ent_obs._compute_entanglement_coupling(i, j, galaxy)
    robust_coupling = robust_ent_obs._compute_entanglement_coupling(i, j, galaxy)
    
    print(f"Original coupling: {original_coupling}")
    print(f"Robust coupling: {robust_coupling}")
    
    # Verify values are non-zero and finite
    assert original_coupling != 0, "Original coupling should not be zero"
    assert np.isfinite(original_coupling), "Original coupling should be finite"
    assert robust_coupling != 0, "Robust coupling should not be zero"
    assert np.isfinite(robust_coupling), "Robust coupling should be finite"
    
    # Test 2: Check that density matrix construction doesn't fail
    # This would prevent "Initial density matrix construction failed"
    try:
        # Try to measure entanglement with robust implementation
        result = robust_ent_obs.measure(galaxy)
        print(f"Entanglement measurement: {result.value} ± {result.uncertainty}")
        
        # Verify result is valid
        assert result is not None, "Entanglement result should not be None"
        assert np.isfinite(result.value), "Entanglement value should be finite"
        assert result.value > 0, "Entanglement should be positive"
    except Exception as e:
        pytest.fail(f"Density matrix construction failed: {str(e)}")
    
    # Test 3: Check that matrix isn't too sparse for eigendecomposition
    # This would prevent "Matrix too sparse for reliable eigendecomposition"
    
    # Choose a smaller subset for eigendecomposition test
    test_size = 50
    test_region = region_A[:test_size]
    test_galaxy = MockDwarfGalaxy()
    test_galaxy.grid = grid
    
    # Spy will try to use _construct_reduced_density_matrix from EntanglementObservable
    spy_ent_obs = EntanglementObservable(grid, test_region)
    
    try:
        # Try to construct a reduced density matrix (this would previously fail)
        rho_A = spy_ent_obs._construct_reduced_density_matrix(test_galaxy)
        
        # Check that matrix isn't too sparse
        if hasattr(rho_A, 'toarray'):
            # For sparse matrices, count non-zero elements
            matrix_dense = rho_A.toarray()
            nonzero_ratio = np.count_nonzero(matrix_dense) / matrix_dense.size
            print(f"Matrix non-zero ratio: {nonzero_ratio:.6f}")
            assert nonzero_ratio > 1e-4, "Matrix should not be too sparse"
        else:
            # For dense matrices, check for extreme small values
            nonzero_ratio = np.sum(np.abs(rho_A) > 1e-10) / rho_A.size
            print(f"Matrix non-zero ratio: {nonzero_ratio:.6f}")
            assert nonzero_ratio > 1e-4, "Matrix should not be too sparse"
        
        # Try eigendecomposition to verify it doesn't fail
        if hasattr(rho_A, 'toarray'):
            eigenvals = sparse_linalg.eigsh(rho_A, k=min(6, rho_A.shape[0]-1),
                                           which='LM', return_eigenvectors=False)
        else:
            eigenvals = np.linalg.eigvalsh(rho_A)
            
        # Verify eigenvalues are valid
        print(f"Eigenvalues: {eigenvals}")
        assert len(eigenvals) > 0, "Eigenvalues should not be empty"
        assert np.all(np.isfinite(eigenvals)), "Eigenvalues should be finite"
    except Exception as e:
        pytest.fail(f"Eigendecomposition failed: {str(e)}")
    
    print("All error-related tests passed!")

if __name__ == "__main__":
    # Run test
    test_entanglement_error_fixes()