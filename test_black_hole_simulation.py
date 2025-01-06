import unittest
from __init__ import QuantumGravity
from examples.black_hole import BlackHoleSimulation
import numpy as np

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
        t_final = 100.0
        self.default_sim.run_simulation(t_final=t_final)
        
        # Verify physical constraints using actual attributes
        self.assertLess(self.default_sim.qg.state.mass, self.default_mass)
        self.assertTrue(len(self.default_sim.temperature_history) > 0)
        self.assertTrue(len(self.default_sim.entropy_history) > 0)

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

if __name__ == '__main__':
    unittest.main()
