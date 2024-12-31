# utils/visualization.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import React from 'react'
from recharts import LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend

class QuantumVisualization:
    """Visualization tools for quantum gravity simulations."""
    
    def __init__(self, grid: 'AdaptiveGrid'):
        self.grid = grid
        
    def create_state_visualization(self, 
                                 state: 'QuantumState', 
                                 projection: str = '3d') -> React.Component:
        """Create visualization of quantum state."""
        # Extract data
        points = self.grid.points
        amplitudes = np.zeros(len(points), dtype=complex)
        
        for idx, coeff in state.coefficients.items():
            amplitudes[idx] = coeff
            
        if projection == '3d':
            return self._create_3d_state_plot(points, amplitudes)
        else:
            return self._create_2d_state_plot(points, amplitudes)
    
    def _create_3d_state_plot(self, 
                             points: np.ndarray, 
                             amplitudes: np.ndarray) -> React.Component:
        """Create 3D visualization of state."""
        # Convert to visualization data
        data = []
        for i, (point, amp) in enumerate(zip(points, amplitudes)):
            data.append({
                'x': float(point[0]),
                'y': float(point[1]),
                'z': float(point[2]),
                'amplitude': float(abs(amp)),
                'phase': float(np.angle(amp))
            })
            
        return self._create_3d_scatter_plot(data)
    
    def _create_2d_state_plot(self, 
                             points: np.ndarray, 
                             amplitudes: np.ndarray) -> React.Component:
        """Create 2D projection of state."""
        # Project onto xy-plane
        data = []
        for i, (point, amp) in enumerate(zip(points, amplitudes)):
            data.append({
                'x': float(point[0]),
                'y': float(point[1]),
                'amplitude': float(abs(amp)),
                'phase': float(np.angle(amp))
            })
            
        return self._create_2d_scatter_plot(data)
    
    def create_evolution_visualization(self, 
                                    history: List['QuantumState']) -> React.Component:
        """Create visualization of state evolution."""
        # Extract time series data
        times = np.arange(len(history))
        energy = []
        entropy = []
        constraints = []
        
        for state in history:
            # Compute observables
            e = self._compute_energy(state)
            s = self._compute_entropy(state)
            c = self._compute_constraints(state)
            
            energy.append(float(e))
            entropy.append(float(s))
            constraints.append(float(c))
            
        data = []
        for t, e, s, c in zip(times, energy, entropy, constraints):
            data.append({
                'time': float(t),
                'energy': e,
                'entropy': s,
                'constraints': c
            })
            
        return self._create_evolution_plot(data)
    
    def create_observable_visualization(self, 
                                     measurements: List['MeasurementResult'],
                                     observable_type: str) -> React.Component:
        """Create visualization of observable measurements."""
        # Extract measurement data
        times = np.arange(len(measurements))
        values = []
        uncertainties = []
        
        for result in measurements:
            values.append(float(result.value))
            uncertainties.append(float(result.uncertainty))
            
        data = []
        for t, v, u in zip(times, values, uncertainties):
            data.append({
                'time': float(t),
                'value': v,
                'uncertainty_up': v + u,
                'uncertainty_down': v - u
            })
            
        return self._create_measurement_plot(data, observable_type)
    
    def _create_3d_scatter_plot(self, data: List[Dict]) -> React.Component:
        """Create 3D scatter plot component."""
        return f'''
import React from 'react'

const ThreeDScatterPlot = () => {{
    return (
        <div className="w-full h-96 bg-white rounded-lg shadow-lg p-4">
            <ScatterChart width={800} height={400} data={{data}}>
                <CartesianGrid />
                <XAxis dataKey="x" />
                <YAxis dataKey="y" />
                <ZAxis dataKey="z" />
                <Tooltip />
                <Legend />
                <Scatter
                    name="State"
                    data={{data}}
                    fill="#8884d8"
                    opacity={0.5}
                    sizeRange={[100, 1000]}
                />
            </ScatterChart>
        </div>
    )
}}

export default ThreeDScatterPlot
'''
    
    def _create_2d_scatter_plot(self, data: List[Dict]) -> React.Component:
        """Create 2D scatter plot component."""
        return f'''
import React from 'react'

const TwoDScatterPlot = () => {{
    return (
        <div className="w-full h-96 bg-white rounded-lg shadow-lg p-4">
            <ScatterChart width={800} height={400} data={{data}}>
                <CartesianGrid />
                <XAxis dataKey="x" />
                <YAxis dataKey="y" />
                <Tooltip />
                <Legend />
                <Scatter
                    name="State"
                    data={{data}}
                    fill="#8884d8"
                    fillOpacity={0.5}
                />
            </ScatterChart>
        </div>
    )
}}

export default TwoDScatterPlot
'''
    
    def _create_evolution_plot(self, data: List[Dict]) -> React.Component:
        """Create evolution plot component."""
        return f'''
import React from 'react'

const EvolutionPlot = () => {{
    return (
        <div className="w-full h-96 bg-white rounded-lg shadow-lg p-4">
            <LineChart width={800} height={400} data={{data}}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="energy" stroke="#8884d8" />
                <Line type="monotone" dataKey="entropy" stroke="#82ca9d" />
                <Line type="monotone" dataKey="constraints" stroke="#ffc658" />
            </LineChart>
        </div>
    )
}}

export default EvolutionPlot
'''
    
    def _create_measurement_plot(self, 
                               data: List[Dict],
                               observable_type: str) -> React.Component:
        """Create measurement plot component."""
        return f'''
import React from 'react'

const MeasurementPlot = () => {{
    return (
        <div className="w-full h-96 bg-white rounded-lg shadow-lg p-4">
            <LineChart width={800} height={400} data={{data}}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="time" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Line
                    type="monotone"
                    dataKey="value"
                    stroke="#8884d8"
                    strokeWidth={2}
                />
                <Line
                    type="monotone"
                    dataKey="uncertainty_up"
                    stroke="#82ca9d"
                    strokeDasharray="3 3"
                />
                <Line
                    type="monotone"
                    dataKey="uncertainty_down"
                    stroke="#82ca9d"
                    strokeDasharray="3 3"
                />
            </LineChart>
            <div className="text-center mt-4">
                <h3 className="text-lg font-semibold">
                    {observable_type} Measurements
                </h3>
            </div>
        </div>
    )
}}

export default MeasurementPlot
'''
    
    def create_error_visualization(self, 
                                 error_tracker: 'ErrorTracker') -> React.Component:
        """Create visualization of error tracking."""
        # Extract error history
        summary = error_tracker.get_error_summary()
        data = []
        
        for error_type, stats in summary.items():
            data.append({
                'type': error_type,
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'max': float(stats['max']),
                'current': float(stats['current'])
            })
            
        return self._create_error_plot(data)
    
    def _create_error_plot(self, data: List[Dict]) -> React.Component:
        """Create error visualization component."""
        return f'''
import React from 'react'

const ErrorPlot = () => {{
    return (
        <div className="w-full h-96 bg-white rounded-lg shadow-lg p-4">
            <div className="grid grid-cols-2 gap-4">
                <div>
                    <BarChart width={400} height={300} data={{data}}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="type" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="mean" fill="#8884d8" />
                        <Bar dataKey="max" fill="#82ca9d" />
                    </BarChart>
                </div>
                <div>
                    <LineChart width={400} height={300} data={{data}}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="type" />
                        <YAxis />
                        <Tooltip />
                        <Legend />
                        <Line
                            type="monotone"
                            dataKey="current"
                            stroke="#8884d8"
                            strokeWidth={2}
                        />
                    </LineChart>
                </div>
            </div>
            <div className="text-center mt-4">
                <h3 className="text-lg font-semibold">Error Analysis</h3>
            </div>
        </div>
    )
}}

export default ErrorPlot
'''

    def plot_verification_metrics(results: List[Dict], save_path: str = None):
        """Plot verification metrics evolution."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        times = [r['time'] for r in results]
        
        # Plot energy conservation
        ax1.plot(times, [r['energy_conservation'] for r in results])
        ax1.set_ylabel('Energy Conservation Violation')
        
        # Plot holographic principle
        ax2.plot(times, [r['holographic_principle'] for r in results])
        ax2.set_ylabel('Holographic Principle Deviation')
        
        # Plot spacetime relation
        ax3.plot(times, [r['spacetime_relation'] for r in results])
        ax3.set_ylabel('Spacetime-Entanglement Relation')
        
        # Plot quantum corrections
        ax4.plot(times, [r['quantum_corrections'] for r in results])
        ax4.set_ylabel('Quantum Corrections Magnitude')
        
        if save_path:
            plt.savefig(save_path)
