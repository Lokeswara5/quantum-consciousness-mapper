import unittest
import numpy as np
import networkx as nx
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

class TestEmergentPatterns(unittest.TestCase):
    def setUp(self):
        self.analyzer = HyperDimensionalAnalyzer(dimensions=3)

    def create_test_state(self, num_points=10):
        """Creates a test state with specified number of points"""
        coordinates = np.random.rand(num_points, 3)
        velocity = np.random.rand(num_points, 3) * 0.1
        acceleration = np.zeros_like(velocity)
        return HyperDimensionalState(
            coordinates=coordinates,
            velocity=velocity,
            acceleration=acceleration,
            phase_space_density=1.0,
            topological_features=[],
            emergent_patterns=[]
        )

    def test_state_graph_creation(self):
        """Test creation of state graph"""
        # Create test data with two distinct clusters
        coordinates = np.vstack([
            np.random.normal(0, 0.1, (5, 3)),  # Cluster 1
            np.random.normal(2, 0.1, (5, 3))   # Cluster 2
        ])

        graph = self.analyzer._create_state_graph(coordinates)

        # Check basic graph properties
        self.assertEqual(len(graph), 10)  # Should have all points
        self.assertTrue(nx.is_connected(graph))  # Should be connected

        # Check that intra-cluster edges have higher weights
        cluster1_edges = list(graph.subgraph(range(5)).edges(data=True))
        cluster2_edges = list(graph.subgraph(range(5, 10)).edges(data=True))
        cross_cluster_edges = [
            (u, v, d) for u, v, d in graph.edges(data=True)
            if (u < 5 and v >= 5) or (u >= 5 and v < 5)
        ]

        avg_intra_weight = np.mean([d['weight'] for _, _, d in cluster1_edges + cluster2_edges])
        avg_cross_weight = np.mean([d['weight'] for _, _, d in cross_cluster_edges])

        self.assertGreater(avg_intra_weight, avg_cross_weight)

    def test_community_detection(self):
        """Test detection of communities in state graph"""
        # Create test graph with two clear communities
        graph = nx.Graph()
        # Add two cliques
        for i in range(5):
            for j in range(i+1, 5):
                graph.add_edge(i, j, weight=1.0)
        for i in range(5, 10):
            for j in range(i+1, 10):
                graph.add_edge(i, j, weight=1.0)
        # Add weak connections between communities
        graph.add_edge(0, 5, weight=0.1)

        communities = self.analyzer._detect_communities(graph)

        self.assertGreaterEqual(len(communities), 1)  # Should find at least one community
        for comm in communities:
            self.assertGreater(len(comm), 0)  # Each community should have nodes
        self.assertEqual(len(communities[0]) + len(communities[1]), 10)

        # Communities should be cohesive
        for comm in communities:
            subgraph = graph.subgraph(comm)
            self.assertTrue(nx.is_connected(subgraph))

    def test_pattern_complexity(self):
        """Test pattern complexity calculation"""
        num_points = 5
        timesteps = 5

        # Simple pattern: linear motion at constant velocity
        simple_states = []
        t = np.linspace(0, 1, timesteps)
        x = np.linspace(0, 1, num_points)

        for step in range(timesteps):
            # Points move uniformly along x-axis
            coords = np.zeros((num_points, 3))
            coords[:, 0] = x + step * 0.2  # Linear motion

            velocity = np.zeros((num_points, 3))
            velocity[:, 0] = np.ones(num_points)  # Constant velocity

            simple_states.append(HyperDimensionalState(
                coordinates=coords.copy(),
                velocity=velocity.copy(),
                acceleration=np.zeros_like(velocity),
                phase_space_density=1.0,
                topological_features=[],
                emergent_patterns=[]
            ))

        # Complex pattern: rotational and oscillatory motion
        complex_states = []
        base_t = np.linspace(0, 2*np.pi, timesteps)

        for step in range(timesteps):
            t = base_t[step]
            phase = np.linspace(0, 2*np.pi, num_points)

            # Coordinates combine rotation and oscillation
            coords = np.zeros((num_points, 3))
            radius = 1.0 + 0.2 * np.sin(3*phase + t)  # Oscillating radius
            coords[:, 0] = radius * np.cos(phase + t)  # x rotation
            coords[:, 1] = radius * np.sin(phase + t)  # y rotation
            coords[:, 2] = 0.2 * np.sin(2*phase + 3*t)  # z oscillation

            # Velocities combine rotational and oscillatory components
            velocity = np.zeros((num_points, 3))
            r_dot = 0.2 * 3 * np.cos(3*phase + t)  # dr/dt
            theta_dot = 1.0  # dtheta/dt

            # Convert to Cartesian velocities
            velocity[:, 0] = (r_dot * np.cos(phase + t) -
                            radius * theta_dot * np.sin(phase + t))
            velocity[:, 1] = (r_dot * np.sin(phase + t) +
                            radius * theta_dot * np.cos(phase + t))
            velocity[:, 2] = 0.2 * 2 * np.cos(2*phase + 3*t) * 3  # dz/dt

            complex_states.append(HyperDimensionalState(
                coordinates=coords.copy(),
                velocity=velocity.copy(),
                acceleration=np.zeros_like(velocity),
                phase_space_density=1.0,
                topological_features=[],
                emergent_patterns=[]
            ))

        # Test simple pattern
        self.analyzer.state_history = simple_states
        simple_community = set(range(num_points))
        simple_complexity = self.analyzer._calculate_pattern_complexity(simple_community)

        # Test complex pattern with new analyzer instance
        complex_analyzer = HyperDimensionalAnalyzer(dimensions=3)
        complex_analyzer.state_history = complex_states

        # Test simple pattern (first 5 points - linear motion)
        simple_community = set(range(5))
        simple_complexity = self.analyzer._calculate_pattern_complexity(simple_community)

        # Test complex pattern with spiral motion
        complex_community = set(range(num_points))
        complex_complexity = complex_analyzer._calculate_pattern_complexity(complex_community)

        self.assertTrue(0 <= simple_complexity <= 1)
        self.assertTrue(0 <= complex_complexity <= 1)
        self.assertGreater(complex_complexity, simple_complexity)

    def test_pattern_stability(self):
        """Test pattern stability calculation"""
        num_points = 10
        velocity = np.ones((num_points, 3))  # Coherent motion

        # Test stable pattern (coherent velocity)
        stable_community = set(range(5))
        stable_score = self.analyzer._calculate_pattern_stability(stable_community, velocity)

        # Test unstable pattern (random velocities)
        unstable_velocity = np.random.rand(num_points, 3)
        unstable_score = self.analyzer._calculate_pattern_stability(stable_community, unstable_velocity)

        self.assertTrue(0 <= stable_score <= 1)
        self.assertTrue(0 <= unstable_score <= 1)
        self.assertGreater(stable_score, unstable_score)

    def test_influence_radius(self):
        """Test influence radius calculation"""
        # Create test state with known pattern
        self.analyzer.state_history = [self.create_test_state()]

        # Test compact pattern
        compact_community = set(range(3))
        compact_radius = self.analyzer._calculate_influence_radius(compact_community)

        # Test spread pattern
        spread_community = set(range(8))
        spread_radius = self.analyzer._calculate_influence_radius(spread_community)

        self.assertGreater(spread_radius, compact_radius)

    def test_interaction_strengths(self):
        """Test interaction strength calculation"""
        # Create test data with three patterns
        self.analyzer.state_history = [self.create_test_state(15)]

        communities = [
            set(range(5)),          # Pattern 1
            set(range(5, 10)),      # Pattern 2
            set(range(10, 15))      # Pattern 3
        ]

        # Test interactions for first pattern
        interactions = self.analyzer._calculate_interaction_strengths(communities[0], communities)

        self.assertEqual(len(interactions), 2)  # Should have interaction with other two patterns
        self.assertTrue(all(0 <= strength <= 1 for strength in interactions.values()))

    def test_emergent_pattern_detection(self):
        """Test complete emergent pattern detection"""
        # Create test data with clear patterns
        coordinates = np.vstack([
            np.random.normal(0, 0.1, (5, 3)),    # Cluster 1
            np.random.normal(2, 0.1, (5, 3)),    # Cluster 2
            np.random.normal(-2, 0.1, (5, 3))    # Cluster 3
        ])
        velocity = np.ones((15, 3))  # Coherent motion

        patterns = self.analyzer._analyze_emergent_patterns(coordinates, velocity)

        self.assertGreater(len(patterns), 0)
        for pattern in patterns:
            self.assertTrue(0 <= pattern.complexity <= 1)
            self.assertTrue(0 <= pattern.stability <= 1)
            self.assertTrue(pattern.influence_radius > 0)
            self.assertGreater(len(pattern.interaction_strength), 0)

if __name__ == '__main__':
    unittest.main()