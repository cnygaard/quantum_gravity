# core/grid.py
import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Tuple, Optional

class AdaptiveGrid:
    """Adaptive grid implementation for quantum gravity."""
    
    def __init__(self, 
                 rho_0: float,
                 eps_threshold: float,
                 l_p: float = 1.616255e-35):  # Planck length
        self.rho_0 = rho_0
        self.eps_threshold = eps_threshold
        self.l_p = l_p
        self.points: List[np.ndarray] = []
        self.neighbors: List[List[int]] = []

    def set_points(self, points: np.ndarray) -> None:
        """Set grid points and update neighbor relationships.
        
        Args:
            points: Array of shape (N, 3) containing point coordinates
        """
        self.points = points
        self._update_neighbors()

    def compute_density(self, x: np.ndarray, R: float) -> float:
        """Compute grid density based on local curvature."""
        return self.rho_0 * (1 + abs(R) / self.R_0)
        
    def refine_grid(self, curvature: np.ndarray) -> None:
        """Refine grid based on curvature threshold."""
        new_points = []
        for i, point in enumerate(self.points):
            if abs(curvature[i]) > self.eps_threshold:
                # Add refinement points
                new_points.extend(self._generate_refined_points(point))
        self.points.extend(new_points)
        self._update_neighbors()

    def _generate_refined_points(self, 
                               center: np.ndarray,
                               refinement_factor: float = 0.5) -> List[np.ndarray]:
        """Generate refined points around a given center."""
        offsets = self.l_p * refinement_factor * np.array([
            [1, 0, 0], [-1, 0, 0],
            [0, 1, 0], [0, -1, 0],
            [0, 0, 1], [0, 0, -1]
        ])
        return [center + offset for offset in offsets]

    def _update_neighbors(self) -> None:
        """Update nearest neighbor lists efficiently."""
        from scipy.spatial import cKDTree
        
        tree = cKDTree(self.points)
        self.neighbors = [tree.query_ball_point(p, self.l_p) for p in self.points]


