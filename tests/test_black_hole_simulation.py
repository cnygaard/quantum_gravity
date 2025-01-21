import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import unittest
from __init__ import QuantumGravity, QuantumState, TimeEvolution
from examples.black_hole import BlackHoleSimulation
import numpy as np
from numerics.errors import ErrorTracker
from physics.conservation import ConservationLawTracker

class TestBlackHoleSimulation(unittest.TestCase):
    def setUp(self):
        self.default_mass = 1000.0
        self.default_sim = BlackHoleSimulation(mass=self.default_mass)

    def test_simulation_initialization(self):
        """Test proper initialization of simulation parameters"""
        sim = BlackHoleSimulation(mass=self.default_mass)
        self.assertEqual(sim.initial_mass, self.default_mass)
        self.assertIsNotNone(sim.qg.state)
        self.assertEqual(sim.qg.state.mass, self.default_mass)

    def test_invalid_mass_values(self):
        """Test handling of invalid mass parameters"""
        with self.assertRaises(ValueError):
            BlackHoleSimulation(mass=-1.0)
        with self.assertRaises(ValueError):
            BlackHoleSimulation(mass=0.0)

    def test_run_simulation(self):
        """Test simulation execution with default parameters"""
        t_final = 1000.0  # Increase from 100.0 to 1000.0
        self.default_sim.run_simulation(t_final=t_final)

        
        # Verify physical constraints using actual attributes
        self.assertLess(self.default_sim.qg.state.mass, self.default_mass)
        self.assertTrue(len(self.default_sim.temperature_history) > 0)
        self.assertTrue(len(self.default_sim.entropy_history) > 0)

    def test_full_evolution_verification(self):
        """Test complete evolution with verification metrics"""
        t_final = 2000.0
        
        # Set quantum state on grid
        self.default_sim.qg.grid.quantum_state = QuantumState(
            grid=self.default_sim.qg.grid,
            initial_mass=self.default_mass,
            eps_cut=self.default_sim.qg.config.config['numerics']['eps_cut']
        )
        
        # Initialize evolution with correct parameters
        self.default_sim.qg.evolution = TimeEvolution(
            grid=self.default_sim.qg.grid,
            config={
                'dt': 0.01,
                'error_tolerance': 1e-6
            },
            error_tracker=ErrorTracker(
                grid=self.default_sim.qg.grid,
                base_tolerances=self.default_sim.qg.config.config['numerics']
            ),
            conservation_tracker=ConservationLawTracker(self.default_sim.qg.grid)
        )
        
        self.default_sim.run_simulation(t_final=t_final)
        
        # Verify evolution length and parameters
        expected_steps = int(t_final / self.default_sim.qg.evolution.dt)
        self.assertGreaterEqual(len(self.default_sim.time_points), expected_steps)
        self.assertTrue(all(v['diagnostics']['beta'] > 0 for v in self.default_sim.verification_results))
        self.assertTrue(all(v['diagnostics']['gamma_eff'] > 0 for v in self.default_sim.verification_results))



    def test_measurement_recording(self):
        """Test measurement recording during simulation"""
        t_final = 10.0
        self.default_sim.run_simulation(t_final=t_final)
        
        self.assertTrue(len(self.default_sim.time_points) > 0)
        self.assertTrue(len(self.default_sim.mass_history) > 0)
        self.assertTrue(len(self.default_sim.radiation_flux_history) > 0)
        
        # Verify measurements are properly recorded
        self.assertEqual(len(self.default_sim.time_points), 
                        len(self.default_sim.mass_history))

    def test_verification_results(self):
        """Test trinity verification metrics recording"""
        self.default_sim.run_simulation(t_final=50.0)
        
        # Verify metrics are being recorded
        self.assertTrue(len(self.default_sim.verification_results) > 0)
        first_result = self.default_sim.verification_results[0]
        
        # Check required metrics are present
        self.assertIn('time', first_result)
        self.assertIn('mass', first_result)

    def test_geometric_entanglement_verification(self):
        """Test geometric-entanglement correspondence verification"""
        self.default_sim.run_simulation(t_final=10.0)
        metrics = self.default_sim.verifier._verify_geometric_entanglement(self.default_sim.qg.state)
        self.assertIn('lhs', metrics)
        self.assertIn('rhs', metrics)
        self.assertIn('relative_error', metrics)

    def test_quantum_parameters(self):
        """Test quantum parameter evolution"""
        self.default_sim.run_simulation(t_final=10.0)
        first_result = self.default_sim.verification_results[0]
        self.assertIn('diagnostics', first_result)
        self.assertIn('beta', first_result['diagnostics'])
        self.assertIn('gamma_eff', first_result['diagnostics'])

    def test_conservation_laws(self):
        """Test conservation law tracking"""
        self.default_sim.run_simulation(t_final=10.0)
        conservation = self.default_sim.verifier.conservation_tracker.check_conservation(
            self.default_sim.verifier.conservation_tracker.compute_quantities(
                self.default_sim.qg.state,
                self.default_sim.qg.operators
            )
        )
        self.assertIn('energy', conservation)
        self.assertIn('momentum', conservation)


if __name__ == '__main__':
    unittest.main()


