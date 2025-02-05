# core/grid.py
#import tracemalloc
import linecache
from typing import List, Tuple, Optional
import numpy as np
from scipy.spatial import cKDTree
from constants import CONSTANTS

def display_top(stats, limit=10):
    """Display top memory allocations."""
    print(f"\nTop {limit} memory allocations:")
    for index, stat in enumerate(stats[:limit], 1):
        frame = stat.traceback[0]
        # Size is in bytes, convert to MB
        print(f"#{index}: {frame.filename}:{frame.lineno}: {stat.size/1024/1024:.1f} MB")
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print(f"    {line}")

class AdaptiveGrid:
    """Adaptive grid implementation for quantum gravity."""
    
    def __init__(self, 
                 eps_threshold: float,
                 l_p: float = 1.0):  # Planck length
        """Initialize adaptive grid."""
        #tracemalloc.start()
        #self.snapshot1 = tracemalloc.take_snapshot()
        
        self.eps_threshold = eps_threshold
        self.l_p = l_p
        self.points = None  # Initialize as None
        self.neighbors = None
        self.max_points = 50000  # Add limit

    def set_points(self, points: np.ndarray) -> None:
        """Set grid points with memory-efficient neighbor computation."""
        if len(points) > self.max_points:
            raise ValueError(f"Too many points ({len(points)}). Maximum allowed: {self.max_points}")
            
        #snapshot2 = tracemalloc.take_snapshot()
        #stats = snapshot2.compare_to(self.snapshot1, 'lineno')
        #display_top(stats)

        # Store points efficiently
        self.points = points.astype(np.float64)  # Use float32 instead of float64
        
        # Build tree with memory-efficient parameters
        tree = cKDTree(self.points, leafsize=32, compact_nodes=True)
        
        #snapshot3 = tracemalloc.take_snapshot()
        #stats = snapshot3.compare_to(snapshot2, 'lineno')
        #display_top(stats)
        
        # Compute neighbors in chunks to reduce memory usage
        chunk_size = 10000
        n_points = len(points)
        self.neighbors = []
        
        for i in range(0, n_points, chunk_size):
            chunk = points[i:min(i+chunk_size, n_points)]
            r_search = self.l_p * 1.1
            chunk_neighbors = tree.query_ball_point(chunk, r_search, workers=-1)
            self.neighbors.extend(chunk_neighbors)
            
            # Force garbage collection after each chunk
            import gc
            gc.collect()
        
        #snapshot4 = tracemalloc.take_snapshot()
        #stats = snapshot4.compare_to(snapshot3, 'lineno')
        #display_top(stats)

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
        tree = cKDTree(self.points)
        # Use parallel processing for neighbor queries
        self.neighbors = tree.query_ball_point(self.points, self.l_p, workers=-1)


    def get_area_element(self, i: int) -> float:
        """Compute area element at grid point i."""
        # Get neighboring points
        neighbors = self.neighbors[i]
        if len(neighbors) < 2:
            return 0.0
            
        # Get vectors to two neighbors
        v1 = self.points[neighbors[0]] - self.points[i]
        v2 = self.points[neighbors[1]] - self.points[i]
        
        # Area element is magnitude of cross product
        area = np.linalg.norm(np.cross(v1, v2)) / 2.0
        
        # Scale by Planck area
        return area * self.l_p**2

    def _setup_grid(self) -> None:
        """Setup adaptive grid focused on horizon."""
        grid_config = self.qg.config.config['grid']
        
        # Reduce grid size significantly
        n_radial = 30  # Reduced from 50
        n_theta = 8    # Reduced from 10
        n_phi = 16     # Reduced from 20
        
        # Generate points more efficiently
        r_min = CONSTANTS['l_p'] 
        r_max = 20 * self.horizon_radius
        
        # Use float32 for memory efficiency
        r_points = np.geomspace(r_min, r_max, n_radial, dtype=np.float32)
        theta = np.linspace(0, np.pi, n_theta, dtype=np.float32)
        phi = np.linspace(0, 2*np.pi, n_phi, dtype=np.float32)
        
        # Calculate points in chunks
        chunk_size = 1000
        points_list = []
        
        for r in r_points:
            for t in theta:
                sin_t = np.sin(t)
                cos_t = np.cos(t)
                chunk = []
                for p in phi:
                    chunk.append([
                        r * sin_t * np.cos(p),
                        r * sin_t * np.sin(p),
                        r * cos_t
                    ])
                    if len(chunk) >= chunk_size:
                        points_list.extend(chunk)
                        chunk = []
                if chunk:
                    points_list.extend(chunk)
        
        points = np.array(points_list, dtype=np.float32)
        self.qg.grid.set_points(points)
        self.qg.state = QuantumState(self.qg.grid)

    def get_max_radius(self) -> float:
        """Calculate maximum radius from grid points.
        
        Returns:
            Maximum radial distance from origin
        """
        # Compute radial distances for all points
        radii = np.linalg.norm(self.points, axis=1)
        
        # Return maximum radius, ensuring non-zero value
        return max(np.max(radii), CONSTANTS['l_p'])


class LeechLattice:
    def __init__(self, points=CONSTANTS['LEECH_LATTICE_POINTS']):  # Reduced from 10000
        self.n_points = points
        self.dimension = CONSTANTS['LEECH_LATTICE_DIMENSION']
        self._cached_energy = None
        self._lattice_points = None
        self._setup_lattice()

    def _setup_lattice(self) -> None:
        """Initialize the Leech lattice structure."""
        # Generate initial lattice points
        self._lattice_points = self._generate_lattice_points()
        
        # Pre-compute common values for efficiency
        self._lattice_norms = np.linalg.norm(self._lattice_points, axis=1)
        self._symmetry_factor = self._compute_symmetry_factor()

    def compute_vacuum_energy(self) -> float:
        """Enhanced vacuum energy with geometric suppression"""
        base_energy = CONSTANTS['hbar']/(CONSTANTS['c'] * CONSTANTS['l_p']**4)
        lattice_sum = self._compute_lattice_sum()
        
        # Geometric suppression from Leech lattice symmetries
        symmetry_factor = self._compute_symmetry_factor()
        
        # Enhanced scale factor with refined corrections
        scale_factor = np.sqrt(self.n_points) * np.log(self.n_points) * (1 + np.log(self.n_points)/self.dimension)
        
        return base_energy * lattice_sum * symmetry_factor / scale_factor



    def _generate_lattice_points(self) -> np.ndarray:
        """Optimized Leech lattice point generation."""
        # Pre-allocate array
        points = np.zeros((self.n_points, self.dimension))
        # Vectorized operations
        points = np.random.randint(-2, 3, (self.n_points, self.dimension))
        # Adjust for divisibility conditions efficiently
        sums = np.sum(points, axis=1)
        squares = np.sum(points**2, axis=1)
        # Quick fixes for conditions
        points[:, 0] += (4 - sums % 4)
        points[:, -1] += (8 - squares % 8)
        return points


    def _compute_lattice_sum(self) -> float:
        """Sum over Leech lattice points"""
        # Efficient implementation using numpy vectorization
        lattice_points = self._generate_lattice_points()
        return np.sum(1.0/np.linalg.norm(lattice_points, axis=1)**4)

    def _compute_symmetry_factor(self) -> float:
        """Calculate symmetry factor from Leech lattice"""
        # M24 group order = 2^10 * 3^3 * 5 * 7 * 11 * 23
        return np.sqrt(244823040 / self.n_points)

    def compute_effective_coupling(self):
        """Calculate effective coupling from lattice symmetries"""
        # Coupling calculation code...
