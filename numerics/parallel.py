# numerics/parallel.py
from mpi4py import MPI
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.sparse import csr_matrix
from dataclasses import dataclass

@dataclass
class ParallelConfig:
    """Configuration for parallel computation."""
    load_balance: str = 'dynamic'  # 'static' or 'dynamic'
    comm_pattern: str = 'neighbor'  # 'neighbor' or 'all_to_all'
    chunk_size: int = 1000  # Points per work unit

class ParallelManager:
    """Manages parallel computation for quantum gravity."""
    
    def __init__(self, grid: 'AdaptiveGrid', config: ParallelConfig):
        self.grid = grid
        self.config = config
        
        # MPI setup
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Domain decomposition
        self.local_points = self._distribute_points()
        self.ghost_points = self._setup_ghost_points()
        
        # Communication setup
        self.neighbors = self._setup_neighbors()
        self.send_buffers = {}
        self.recv_buffers = {}
        
    def _distribute_points(self) -> List[int]:
        """Distribute grid points among processes."""
        n_points = len(self.grid.points)
        
        if self.config.load_balance == 'static':
            # Simple block distribution
            points_per_proc = n_points // self.size
            start = self.rank * points_per_proc
            end = start + points_per_proc if self.rank < self.size - 1 else n_points
            return list(range(start, end))
        else:
            # Dynamic load balancing based on computational cost
            costs = [self._estimate_cost(i) for i in range(n_points)]
            return self._dynamic_partition(costs)
    
    def _estimate_cost(self, point_idx: int) -> float:
        """Estimate computational cost for a point."""
        # Consider number of neighbors and local curvature
        n_neighbors = len(self.grid.neighbors[point_idx])
        x = self.grid.points[point_idx]
        curvature = np.linalg.norm(x) / self.grid.l_p
        return n_neighbors * (1.0 + abs(curvature))
    
    def _dynamic_partition(self, costs: List[float]) -> List[int]:
        """Partition points dynamically based on costs."""
        total_cost = sum(costs)
        target_cost = total_cost / self.size
        
        # Group points into chunks
        chunks = []
        current_chunk = []
        current_cost = 0.0
        
        for i, cost in enumerate(costs):
            current_chunk.append(i)
            current_cost += cost
            
            if current_cost >= self.config.chunk_size * target_cost:
                chunks.append((current_chunk, current_cost))
                current_chunk = []
                current_cost = 0.0
        
        if current_chunk:
            chunks.append((current_chunk, current_cost))
        
        # Distribute chunks to minimize cost imbalance
        my_chunks = []
        if self.rank < len(chunks):
            my_chunks.extend(chunks[self.rank][0])
        
        return my_chunks
    
    def _setup_ghost_points(self) -> Dict[int, List[int]]:
        """Set up ghost points for boundary communication."""
        ghost_points = {}
        
        for point_idx in self.local_points:
            # Find neighbors that belong to other processes
            for neighbor in self.grid.neighbors[point_idx]:
                owner = self._get_point_owner(neighbor)
                if owner != self.rank:
                    if owner not in ghost_points:
                        ghost_points[owner] = []
                    ghost_points[owner].append(neighbor)
        
        return ghost_points
    
    def _get_point_owner(self, point_idx: int) -> int:
        """Determine which process owns a point."""
        if self.config.load_balance == 'static':
            points_per_proc = len(self.grid.points) // self.size
            return min(point_idx // points_per_proc, self.size - 1)
        else:
            # Use lookup table or communication
            return self.comm.allreduce(
                self.rank if point_idx in self.local_points else -1,
                op=MPI.MAX
            )
    
    def _setup_neighbors(self) -> List[int]:
        """Set up neighboring processes for communication."""
        neighbors = set()
        
        for point_idx in self.local_points:
            for neighbor in self.grid.neighbors[point_idx]:
                owner = self._get_point_owner(neighbor)
                if owner != self.rank:
                    neighbors.add(owner)
        
        return sorted(list(neighbors))
    
    def synchronize_state(self, state: 'QuantumState') -> None:
        """Synchronize quantum state across processes."""
        # Prepare send buffers
        send_requests = []
        for neighbor in self.neighbors:
            ghost_indices = self.ghost_points.get(neighbor, [])
            if ghost_indices:
                send_data = self._pack_state_data(state, ghost_indices)
                req = self.comm.Isend(send_data, dest=neighbor, tag=0)
                send_requests.append(req)
        
        # Prepare receive buffers
        recv_requests = []
        for neighbor in self.neighbors:
            if neighbor in self.ghost_points:
                recv_size = len(self.ghost_points[neighbor])
                recv_buffer = np.empty(recv_size, dtype=np.complex128)
                req = self.comm.Irecv(recv_buffer, source=neighbor, tag=0)
                recv_requests.append((req, neighbor, recv_buffer))
        
        # Wait for completion
        MPI.Request.Waitall(send_requests)
        
        # Update state with received data
        for req, neighbor, buffer in recv_requests:
            req.Wait()
            self._unpack_state_data(state, self.ghost_points[neighbor], buffer)
    
    def _pack_state_data(self, 
                        state: 'QuantumState',
                        indices: List[int]) -> np.ndarray:
        """Pack state data for communication."""
        data = np.zeros(len(indices), dtype=np.complex128)
        for i, idx in enumerate(indices):
            if idx in state.coefficients:
                data[i] = state.coefficients[idx]
        return data
    
    def _unpack_state_data(self,
                          state: 'QuantumState',
                          indices: List[int],
                          data: np.ndarray) -> None:
        """Unpack received state data."""
        for i, idx in enumerate(indices):
            if abs(data[i]) > state.eps_cut:
                state.coefficients[idx] = data[i]
    
    def parallel_operator_apply(self,
                              operator: 'QuantumOperator',
                              state: 'QuantumState') -> 'QuantumState':
        """Apply operator in parallel."""
        # Synchronize state
        self.synchronize_state(state)
        
        # Local computation
        local_result = QuantumState(self.grid, state.eps_cut)
        for idx in self.local_points:
            if idx in state.coefficients:
                result = operator.apply_local(state.basis_states[idx], idx)
                if abs(result.coefficient) > state.eps_cut:
                    local_result.add_basis_state(
                        idx, result.coefficient, result.state
                    )
        
        # Gather results
        if self.config.comm_pattern == 'all_to_all':
            # All-to-all communication
            global_result = self.comm.allgather(local_result)
            
            # Merge results
            final_result = QuantumState(self.grid, state.eps_cut)
            for partial in global_result:
                final_result.merge_with(partial)
            
            return final_result
        else:
            # Neighbor-only communication
            self.synchronize_state(local_result)
            return local_result
    
    def parallel_measure(self,
                        observable: 'Observable',
                        state: 'QuantumState') -> 'MeasurementResult':
        """Perform measurement in parallel."""
        # Local measurement
        local_result = observable.measure_local(
            state, self.local_points
        )
        
        # Combine results
        if isinstance(local_result.value, (float, complex)):
            value = self.comm.allreduce(local_result.value, op=MPI.SUM)
            variance = self.comm.allreduce(
                local_result.uncertainty**2, op=MPI.SUM
            )
        else:
            value = self.comm.allreduce(local_result.value, op=MPI.SUM)
            variance = self.comm.allreduce(
                local_result.uncertainty**2, op=MPI.SUM
            )
        
        return MeasurementResult(
            value=value,
            uncertainty=np.sqrt(variance)
        )
    
    def barrier(self) -> None:
        """Synchronize all processes."""
        self.comm.Barrier()
    
    def gather_to_root(self, data: Any) -> Optional[List[Any]]:
        """Gather data to root process."""
        return self.comm.gather(data, root=0)
    
    def broadcast_from_root(self, data: Any) -> Any:
        """Broadcast data from root process."""
        return self.comm.bcast(data, root=0)