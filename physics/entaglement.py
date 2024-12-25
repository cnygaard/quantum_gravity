# physics/entanglement.py
class EntanglementHandler:
    """Handle quantum entanglement with nearest-neighbor approximation."""
    
    def __init__(self, 
                 grid: AdaptiveGrid,
                 l_cutoff: float):
        self.grid = grid
        self.l_cutoff = l_cutoff
        self.entanglement_matrix = {}  # Sparse storage
        
    def compute_entanglement(self, 
                           state: QuantumState) -> csr_matrix:
        """Compute entanglement between nearby points."""
        rows, cols, data = [], [], []
        
        for i, neighbors in enumerate(self.grid.neighbors):
            for j in neighbors:
                if i < j:  # Avoid double counting
                    E_ij = self._compute_pair_entanglement(
                        state, i, j
                    )
                    if abs(E_ij) > state.eps_cut:
                        rows.extend([i, j])
                        cols.extend([j, i])
                        data.extend([E_ij, E_ij.conjugate()])
                        
        return csr_matrix((data, (rows, cols)), 
                         shape=(len(self.grid.points),) * 2)
    
    def _compute_pair_entanglement(self,
                                 state: QuantumState,
                                 i: int,
                                 j: int) -> complex:
        """Compute entanglement between two points."""
        if i in state.coefficients and j in state.coefficients:
            return (state.coefficients[i].conjugate() * 
                   state.coefficients[j])
        return 0.0
