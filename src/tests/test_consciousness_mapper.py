import unittest
from datetime import datetime
from core.consciousness_mapper import ConsciousnessMapper, ConsciousnessPattern
from utils.pattern_analyzer import PatternAnalyzer

class MockAISystem:
    """Mock AI system for testing"""
    def __init__(self):
        self.state = {
            'decisions_made': 10,
            'self_modifications': 5,
            'state_queries': 15
        }

class TestConsciousnessMapper(unittest.TestCase):
    def setUp(self):
        self.mapper = ConsciousnessMapper()
        self.analyzer = PatternAnalyzer()
        self.ai_system = MockAISystem()

    def test_pattern_detection(self):
        pattern = self.mapper.analyze_system_state(self.ai_system)
        self.assertIsInstance(pattern, ConsciousnessPattern)
        self.assertTrue(0 <= pattern.confidence <= 1)

    def test_pattern_evolution(self):
        # Generate multiple patterns
        patterns = [
            self.mapper.analyze_system_state(self.ai_system)
            for _ in range(3)
        ]

        # Check evolution tracking
        evolution = self.mapper.get_evolution_trajectory()
        self.assertEqual(len(evolution), 3)

    def test_emergence_detection(self):
        pattern = self.mapper.analyze_system_state(self.ai_system)
        emergence = self.analyzer.analyze_pattern_emergence(pattern.features)

        # Check emergence scores
        self.assertTrue(all(0 <= score <= 1 for score in emergence.values()))

if __name__ == '__main__':
    unittest.main()