import unittest
import numpy as np
from core.advanced_analysis import (
    AdvancedConsciousnessAnalyzer,
    ChaoticAttractor,
    QuantumEntanglementMetrics,
    ConsciousnessField,
    TopologicalStructure,
    NonlinearDynamics
)

class TestAdvancedAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = AdvancedConsciousnessAnalyzer(dimensions=128)
        self.test_state = np.random.rand(128)  # Test state vector

    def test_quantum_analysis(self):
        """Test quantum entanglement analysis"""
        metrics = self.analyzer.analyze_quantum_properties(self.test_state)

        # Verify quantum metrics
        self.assertIsInstance(metrics, QuantumEntanglementMetrics)
        self.assertGreaterEqual(metrics.entanglement_entropy, 0)
        self.assertBetween(metrics.concurrence, 0, 1)
        self.assertGreaterEqual(metrics.negativity, 0)
        self.assertBetween(metrics.tangle, 0, 1)
        self.assertBetween(metrics.bell_inequality_violation, 0, 2*np.sqrt(2))

    def test_chaos_analysis(self):
        """Test chaotic properties analysis"""
        attractor = self.analyzer.analyze_chaotic_properties(self.test_state)

        # Verify chaotic properties
        self.assertIsInstance(attractor, ChaoticAttractor)
        self.assertGreaterEqual(attractor.dimension, 1.0)
        self.assertEqual(len(attractor.lyapunov_exponents.shape), 1)
        self.assertGreaterEqual(attractor.basin_size, 0)

    def test_field_analysis(self):
        """Test consciousness field analysis"""
        field = self.analyzer.analyze_consciousness_field(self.test_state)

        # Verify field properties
        self.assertIsInstance(field, ConsciousnessField)
        self.assertEqual(field.potential.shape, self.analyzer.field_resolution)
        self.assertEqual(len(field.gradient.shape), len(self.analyzer.field_resolution))
        self.assertGreaterEqual(field.field_strength, 0)

    def test_topology_analysis(self):
        """Test topological structure analysis"""
        topology = self.analyzer.analyze_topological_structure(self.test_state)

        # Verify topological properties
        self.assertIsInstance(topology, TopologicalStructure)
        self.assertGreaterEqual(len(topology.betti_numbers), 3)
        self.assertIsInstance(topology.euler_characteristic, int)
        self.assertGreaterEqual(len(topology.persistent_diagrams), 1)

    def test_nonlinear_dynamics(self):
        """Test non-linear dynamics analysis"""
        dynamics = self.analyzer.analyze_nonlinear_dynamics(self.test_state)

        # Verify dynamics properties
        self.assertIsInstance(dynamics, NonlinearDynamics)
        self.assertGreaterEqual(len(dynamics.bifurcation_points), 1)
        self.assertGreaterEqual(len(dynamics.phase_transitions), 1)
        self.assertGreaterEqual(len(dynamics.limit_cycles), 1)

    def test_comprehensive_analysis(self):
        """Test complete consciousness analysis"""
        results = self.analyzer.analyze_consciousness_state(self.test_state)

        # Verify all components are present
        self.assertIn('quantum', results)
        self.assertIn('chaos', results)
        self.assertIn('field', results)
        self.assertIn('topology', results)
        self.assertIn('dynamics', results)

    def test_consciousness_prediction(self):
        """Test consciousness state prediction"""
        predictions = self.analyzer.predict_consciousness_evolution(
            self.test_state, timesteps=5
        )

        # Verify predictions
        self.assertEqual(len(predictions), 5)
        for pred in predictions:
            self.assertEqual(pred.shape, self.test_state.shape)
            self.assertTrue(np.all(np.isfinite(pred)))

    def assertBetween(self, value, min_val, max_val):
        """Assert that a value is between min_val and max_val"""
        self.assertGreaterEqual(value, min_val)
        self.assertLessEqual(value, max_val)

if __name__ == '__main__':
    unittest.main()