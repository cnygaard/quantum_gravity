# utils/io.py
import numpy as np
import h5py
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import pickle
from datetime import datetime
import logging

@dataclass
class SimulationMetadata:
    """Metadata for simulation runs."""
    timestamp: str
    grid_points: int
    cutoff: float
    parameters: Dict[str, Any]
    git_commit: Optional[str] = None
    
class QuantumGravityIO:
    """I/O utilities for quantum gravity simulations."""
    
    def __init__(self, base_path: str = "./output"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Setup logging
        self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Configure logging system."""
        log_path = self.base_path / "simulation.log"
        logging.basicConfig(
            filename=str(log_path),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
    
    def save_state(self,
                  state: 'QuantumState',
                  filename: str,
                  metadata: Optional[Dict] = None) -> None:
        """Save quantum state to HDF5 file."""
        try:
            filepath = self.base_path / f"{filename}.h5"
            
            with h5py.File(filepath, 'w') as f:
                # Save metadata
                meta = {
                    'timestamp': datetime.now().isoformat(),
                    'grid_points': len(state.grid.points),
                    'cutoff': state.eps_cut
                }
                if metadata:
                    meta.update(metadata)
                    
                f.attrs.update(meta)
                
                # Save grid points
                f.create_dataset('grid/points', data=state.grid.points)
                f.create_dataset('grid/neighbors', data=self._encode_neighbors(
                    state.grid.neighbors
                ))
                
                # Save state coefficients
                coeff_group = f.create_group('coefficients')
                indices = []
                values = []
                for idx, coeff in state.coefficients.items():
                    indices.append(idx)
                    values.append(coeff)
                    
                coeff_group.create_dataset('indices', data=indices)
                coeff_group.create_dataset('values', data=values)
                
                # Save basis states
                basis_group = f.create_group('basis_states')
                for idx, basis_state in state.basis_states.items():
                    basis_group.create_dataset(str(idx), data=basis_state)
                    
            logging.info(f"Successfully saved state to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving state: {str(e)}")
            raise
    
    def load_state(self, filename: str) -> Tuple['QuantumState', Dict]:
        """Load quantum state from HDF5 file."""
        try:
            filepath = self.base_path / f"{filename}.h5"
            
            with h5py.File(filepath, 'r') as f:
                # Load metadata
                metadata = dict(f.attrs)
                
                # Reconstruct grid
                points = f['grid/points'][:]
                neighbors = self._decode_neighbors(f['grid/neighbors'][:])
                grid = self._reconstruct_grid(points, neighbors)
                
                # Create new state
                state = QuantumState(grid, metadata['cutoff'])
                
                # Load coefficients
                indices = f['coefficients/indices'][:]
                values = f['coefficients/values'][:]
                for idx, val in zip(indices, values):
                    state.coefficients[int(idx)] = val
                
                # Load basis states
                basis_group = f['basis_states']
                for idx in basis_group.keys():
                    state.basis_states[int(idx)] = basis_group[idx][:]
                
            logging.info(f"Successfully loaded state from {filepath}")
            return state, metadata
            
        except Exception as e:
            logging.error(f"Error loading state: {str(e)}")
            raise
    
    def save_evolution_history(self,
                             history: List['QuantumState'],
                             filename: str,
                             metadata: Optional[Dict] = None) -> None:
        """Save evolution history to HDF5 file."""
        try:
            filepath = self.base_path / f"{filename}.h5"
            
            with h5py.File(filepath, 'w') as f:
                # Save metadata
                meta = {
                    'timestamp': datetime.now().isoformat(),
                    'n_steps': len(history)
                }
                if metadata:
                    meta.update(metadata)
                    
                f.attrs.update(meta)
                
                # Save each state
                states_group = f.create_group('states')
                for i, state in enumerate(history):
                    state_group = states_group.create_group(str(i))
                    self._save_state_to_group(state, state_group)
                    
            logging.info(f"Successfully saved evolution history to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving evolution history: {str(e)}")
            raise
    
    def load_evolution_history(self, 
                             filename: str) -> Tuple[List['QuantumState'], Dict]:
        """Load evolution history from HDF5 file."""
        try:
            filepath = self.base_path / f"{filename}.h5"
            
            with h5py.File(filepath, 'r') as f:
                # Load metadata
                metadata = dict(f.attrs)
                
                # Load states
                states_group = f['states']
                history = []
                
                for i in range(metadata['n_steps']):
                    state_group = states_group[str(i)]
                    state = self._load_state_from_group(state_group)
                    history.append(state)
                
            logging.info(f"Successfully loaded evolution history from {filepath}")
            return history, metadata
            
        except Exception as e:
            logging.error(f"Error loading evolution history: {str(e)}")
            raise
    
    def save_measurements(self,
                         measurements: List['MeasurementResult'],
                         filename: str,
                         metadata: Optional[Dict] = None) -> None:
        """Save measurement results to JSON file."""
        try:
            filepath = self.base_path / f"{filename}.json"
            
            data = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'n_measurements': len(measurements)
                },
                'measurements': [
                    {
                        'value': self._convert_to_serializable(m.value),
                        'uncertainty': self._convert_to_serializable(m.uncertainty),
                        'metadata': m.metadata
                    }
                    for m in measurements
                ]
            }
            
            if metadata:
                data['metadata'].update(metadata)
                
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
                
            logging.info(f"Successfully saved measurements to {filepath}")
            
        except Exception as e:
            logging.error(f"Error saving measurements: {str(e)}")
            raise
    
    def load_measurements(self, 
                         filename: str) -> Tuple[List['MeasurementResult'], Dict]:
        """Load measurement results from JSON file."""
        try:
            filepath = self.base_path / f"{filename}.json"
            
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            measurements = [
                MeasurementResult(
                    value=self._convert_from_serializable(m['value']),
                    uncertainty=self._convert_from_serializable(m['uncertainty']),
                    metadata=m['metadata']
                )
                for m in data['measurements']
            ]
            
            logging.info(f"Successfully loaded measurements from {filepath}")
            return measurements, data['metadata']
            
        except Exception as e:
            logging.error(f"Error loading measurements: {str(e)}")
            raise
    
    def _encode_neighbors(self, neighbors: List[List[int]]) -> np.ndarray:
        """Encode neighbor lists for HDF5 storage."""
        max_neighbors = max(len(n) for n in neighbors)
        encoded = np.full((len(neighbors), max_neighbors), -1, dtype=np.int32)
        
        for i, n in enumerate(neighbors):
            encoded[i, :len(n)] = n
            
        return encoded
    
    def _decode_neighbors(self, encoded: np.ndarray) -> List[List[int]]:
        """Decode neighbor lists from HDF5 storage."""
        neighbors = []
        for row in encoded:
            neighbors.append([int(x) for x in row[row >= 0]])
        return neighbors
    
    def _reconstruct_grid(self,
                         points: np.ndarray,
                         neighbors: List[List[int]]) -> 'AdaptiveGrid':
        """Reconstruct grid from saved data."""
        grid = AdaptiveGrid(rho_0=1.0, eps_threshold=1e-10)
        grid.points = points
        grid.neighbors = neighbors
        return grid
    
    def _save_state_to_group(self,
                            state: 'QuantumState',
                            group: h5py.Group) -> None:
        """Save quantum state to HDF5 group."""
        # Save coefficients
        coeff_group = group.create_group('coefficients')
        indices = []
        values = []
        for idx, coeff in state.coefficients.items():
            indices.append(idx)
            values.append(coeff)
            
        coeff_group.create_dataset('indices', data=indices)
        coeff_group.create_dataset('values', data=values)
        
        # Save basis states
        basis_group = group.create_group('basis_states')
        for idx, basis_state in state.basis_states.items():
            basis_group.create_dataset(str(idx), data=basis_state)
    
    def _load_state_from_group(self, group: h5py.Group) -> 'QuantumState':
        """Load quantum state from HDF5 group."""
        # Create new state
        state = QuantumState(self.grid, self.eps_cut)
        
        # Load coefficients
        indices = group['coefficients/indices'][:]
        values = group['coefficients/values'][:]
        for idx, val in zip(indices, values):
            state.coefficients[int(idx)] = val
        
        # Load basis states
        basis_group = group['basis_states']
        for idx in basis_group.keys():
            state.basis_states[int(idx)] = basis_group[idx][:]
            
        return state
    
    def _convert_to_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, complex):
            return {'real': obj.real, 'imag': obj.imag}
        return obj
    
    def _convert_from_serializable(self, obj: Any) -> Any:
        """Convert from JSON-serializable format."""
        if isinstance(obj, list):
            return np.array(obj)
        elif isinstance(obj, dict) and 'real' in obj and 'imag' in obj:
            return complex(obj['real'], obj['imag'])
        return obj