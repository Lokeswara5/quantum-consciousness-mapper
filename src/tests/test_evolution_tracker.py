import unittest
import numpy as np
from datetime import datetime
from src.core.evolution_tracker import EvolutionTracker, EvolutionState, EvolutionEvent

class TestEvolutionTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = EvolutionTracker(state_dim=64)
        self.test_state = np.random.rand(64)
        self.test_state = self.test_state / np.linalg.norm(self.test_state)

    def test_state_tracking(self):
        """Test basic state tracking functionality"""
        # Track initial state
        state1 = self.tracker.track_state(self.test_state)

        # Verify state properties
        self.assertIsInstance(state1, EvolutionState)
        self.assertEqual(state1.state_vector.shape, (64,))
        self.assertGreater(state1.stability_score, 0)

        # Verify quantum properties
        self.assertIn('coherence', state1.quantum_properties)
        self.assertIn('entanglement', state1.quantum_properties)
        self.assertIn('field_strength', state1.quantum_properties)

        # Track second state
        state2 = self.tracker.track_state(self.test_state * 1.1)

        # Verify history
        self.assertEqual(len(self.tracker.state_history), 2)
        self.assertIsInstance(self.tracker.state_history[0], EvolutionState)

    def test_event_detection(self):
        """Test event detection functionality"""
        # Track initial state
        self.tracker.track_state(self.test_state)

        # Track significantly different state to trigger event
        modified_state = -self.test_state  # Completely opposite state
        self.tracker.track_state(modified_state)

        # Verify events were detected
        self.assertGreater(len(self.tracker.event_history), 0)

        # Verify event properties
        event = self.tracker.event_history[0]
        self.assertIsInstance(event, EvolutionEvent)
        self.assertIn(event.event_type,
                     ['bifurcation', 'phase_transition', 'emergence', 'collapse'])
        self.assertIsInstance(event.description, str)
        self.assertIsInstance(event.metrics, dict)

    def test_evolution_summary(self):
        """Test evolution summary generation"""
        # Track several states
        for _ in range(5):
            state = np.random.rand(64)
            state = state / np.linalg.norm(state)
            self.tracker.track_state(state)

        # Get summary
        summary = self.tracker.get_evolution_summary()

        # Verify summary contents
        self.assertEqual(summary['total_states'], 5)
        self.assertIn('average_stability', summary)
        self.assertIn('event_distribution', summary)
        self.assertIn('quantum_trends', summary)
        self.assertIn('topology_evolution', summary)

    def test_quantum_properties(self):
        """Test quantum property calculations"""
        state = self.tracker.track_state(self.test_state)

        # Verify quantum properties are within expected ranges
        self.assertGreaterEqual(state.quantum_properties['coherence'], 0)
        self.assertLessEqual(state.quantum_properties['coherence'], 1)
        self.assertGreaterEqual(state.quantum_properties['entanglement'], 0)
        self.assertLessEqual(state.quantum_properties['entanglement'], 1)
        self.assertGreaterEqual(state.quantum_properties['field_strength'], 0)
        self.assertLessEqual(state.quantum_properties['field_strength'], 1)

    def test_stability_calculation(self):
        """Test stability score calculation"""
        # Track same state twice (should be very stable)
        state1 = self.tracker.track_state(self.test_state)
        state2 = self.tracker.track_state(self.test_state)

        # Verify high stability
        self.assertGreater(state2.stability_score, 0.9)

        # Track very different state (should be less stable)
        modified_state = -self.test_state  # Opposite state
        state3 = self.tracker.track_state(modified_state)

        # Verify lower stability
        self.assertLess(state3.stability_score, state2.stability_score)

    def test_evolution_graph(self):
        """Test evolution graph construction"""
        # Track several states
        for _ in range(3):
            state = np.random.rand(64)
            state = state / np.linalg.norm(state)
            self.tracker.track_state(state)

        # Verify graph properties
        self.assertEqual(len(self.tracker.evolution_graph.nodes), 3)
        self.assertEqual(len(self.tracker.evolution_graph.edges), 2)

        # Verify node attributes
        for node in self.tracker.evolution_graph.nodes:
            data = self.tracker.evolution_graph.nodes[node]
            self.assertIn('state', data)
            self.assertIn('properties', data)
            self.assertIn('timestamp', data)

if __name__ == '__main__':
    unittest.main()