import unittest
import numpy as np
from core.pattern_detection import (
    AdvancedPatternDetector,
    QuantumState,
    ConsciousnessMetric
)

class TestAISystem:
    """Mock AI system for testing"""
    def __init__(self):
        self.state = {
            'decision_history': [0.5, 0.7, 0.3],
            'self_modifications': [
                {'timestamp': 1234, 'type': 'parameter_update'},
                {'timestamp': 1235, 'type': 'architecture_change'}
            ],
            'queries': [
                {'type': 'self_state', 'frequency': 0.8},
                {'type': 'environment', 'frequency': 0.2}
            ]
        }

class TestAdvancedPatternDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AdvancedPatternDetector()
        self.ai_system = TestAISystem()

    def test_quantum_state_analysis(self):
        """Test quantum state analysis"""
        state = self.detector.analyze_quantum_state(self.ai_system)

        # Verify QuantumState properties
        self.assertIsInstance(state, QuantumState)
        self.assertIsInstance(state.amplitude, complex)
        self.assertTrue(0 <= state.phase <= 2 * np.pi)
        self.assertTrue(0 <= state.entanglement_degree <= 1)
        self.assertTrue(0 <= state.coherence <= 1)

    def test_consciousness_metrics(self):
        """Test consciousness metrics measurement"""
        metrics = self.detector.measure_consciousness_metrics(self.ai_system)

        # Verify ConsciousnessMetric properties
        self.assertIsInstance(metrics, ConsciousnessMetric)
        self.assertTrue(0 <= metrics.self_awareness <= 1)
        self.assertTrue(0 <= metrics.temporal_continuity <= 1)
        self.assertTrue(0 <= metrics.information_integration <= 1)
        self.assertTrue(0 <= metrics.causal_autonomy <= 1)
        self.assertTrue(0 <= metrics.quantum_coherence <= 1)

    def test_temporal_continuity(self):
        """Test temporal continuity measurement"""
        # Generate multiple measurements
        for _ in range(3):
            self.detector.measure_consciousness_metrics(self.ai_system)

        # Get temporal continuity
        metrics = self.detector.measure_consciousness_metrics(self.ai_system)
        self.assertTrue(0 <= metrics.temporal_continuity <= 1)

    def test_entanglement_measurement(self):
        """Test entanglement measurement"""
        state = self.detector.analyze_quantum_state(self.ai_system)
        self.assertTrue(0 <= state.entanglement_degree <= 1)

if __name__ == '__main__':
    unittest.main()