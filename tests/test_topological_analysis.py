import unittest
import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, TopologicalFeature

class TestTopologicalAnalysis(unittest.TestCase):
    def setUp(self):
        self.analyzer = HyperDimensionalAnalyzer(dimensions=3)  # Use 3D for easy visualization

    def test_distance_matrix(self):
        # Test points forming a simple triangle
        coordinates = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])

        distances = self.analyzer._calculate_distance_matrix(coordinates)

        # Expected distances in a 1-1-√2 triangle
        expected = np.array([
            [0, 1, 1],
            [1, 0, np.sqrt(2)],
            [1, np.sqrt(2), 0]
        ])

        np.testing.assert_array_almost_equal(distances, expected)

    def test_persistence_calculation(self):
        print("\nRunning persistence calculation test...")
        # Test points forming a triangle - simplest possible loop
        coordinates = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0]
        ])
        print("Test coordinates:")
        print(coordinates)

        distances = self.analyzer._calculate_distance_matrix(coordinates)
        persistence = self.analyzer._calculate_persistence(distances, dimension=1)

        # Should detect at least one loop
        self.assertTrue(any(feature[2] == 'loop' for feature in persistence))

    def test_feature_detection(self):
        # Create a simple dataset with known features:
        # - A circle (1D loop)
        # - Two clusters (0D components)
        points_circle = np.array([
            [np.cos(theta), np.sin(theta), 0]
            for theta in np.linspace(0, 2*np.pi, 8)[:-1]
        ])

        points_clusters = np.array([
            [3, 0, 0],
            [3.1, 0.1, 0],
            [2.9, -0.1, 0],
            [-3, 0, 0],
            [-3.1, 0.1, 0],
            [-2.9, -0.1, 0]
        ])

        coordinates = np.vstack([points_circle, points_clusters])

        features = self.analyzer._detect_topological_features(coordinates)

        # Check that we detect the expected features
        feature_types = [f.type for f in features]

        # Should find at least:
        # - One loop (the circle)
        # - Multiple components (the clusters)
        self.assertIn('loop', feature_types)
        self.assertIn('component', feature_types)

    def test_significance_thresholds(self):
        # Create persistence data with known noise levels
        persistence_by_dim = {
            0: [(0, 0.1, 'component'), (0, 0.2, 'component'), (0, 1.0, 'component')],
            1: [(0.1, 0.2, 'loop'), (0.1, 0.3, 'loop'), (0.1, 2.0, 'loop')]
        }

        thresholds = self.analyzer._calculate_significance_thresholds(persistence_by_dim)

        # Check that thresholds are reasonable
        self.assertTrue(0 < thresholds[0] < 1.0)  # Should filter noise but keep significant features
        self.assertTrue(0 < thresholds[1] < 2.0)

    def test_realistic_data(self):
        # Test with more realistic quantum consciousness data
        # Generate synthetic data that mimics quantum state evolution
        t = np.linspace(0, 2*np.pi, 100)
        coordinates = np.array([
            np.sin(t) + 0.1*np.random.randn(len(t)),  # Base oscillation + noise
            np.cos(2*t) + 0.1*np.random.randn(len(t)), # Different frequency
            0.5*np.sin(3*t) + 0.1*np.random.randn(len(t))  # Higher harmonic
        ]).T

        features = self.analyzer._detect_topological_features(coordinates)

        # Check that features are found and properly characterized
        self.assertTrue(len(features) > 0)
        for feature in features:
            self.assertIsInstance(feature, TopologicalFeature)
            self.assertTrue(feature.persistence > 0)
            self.assertTrue(feature.birth_time <= feature.death_time)

if __name__ == '__main__':
    unittest.main()