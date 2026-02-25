import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer
import unittest
from tests.test_topological_analysis import TestTopologicalAnalysis

def visualize_test_case(test_name):
    """Visualize the test data and detected features"""
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    if test_name == "circle_clusters":
        # Generate test data
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

    elif test_name == "quantum_evolution":
        t = np.linspace(0, 2*np.pi, 100)
        coordinates = np.array([
            np.sin(t) + 0.1*np.random.randn(len(t)),
            np.cos(2*t) + 0.1*np.random.randn(len(t)),
            0.5*np.sin(3*t) + 0.1*np.random.randn(len(t))
        ]).T

    # Detect features
    features = analyzer._detect_topological_features(coordinates)

    # Visualize
    fig = plt.figure(figsize=(15, 5))

    # 3D scatter plot
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2])
    ax1.set_title('Data Points')

    # Persistence diagram
    ax2 = fig.add_subplot(132)
    for feature in features:
        color = 'blue' if feature.type == 'component' else 'red'
        ax2.scatter(feature.birth_time, feature.death_time, c=color)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_title('Persistence Diagram')
    ax2.set_xlabel('Birth Time')
    ax2.set_ylabel('Death Time')

    # Feature statistics
    ax3 = fig.add_subplot(133)
    persistence_values = [f.persistence for f in features]
    ax3.hist(persistence_values, bins=10)
    ax3.set_title('Feature Persistence Distribution')
    ax3.set_xlabel('Persistence')
    ax3.set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(f'output/topology_test_{test_name}.png')
    plt.close()

def main():
    # Run tests
    suite = unittest.TestLoader().loadTestsFromTestCase(TestTopologicalAnalysis)
    result = unittest.TextTestRunner(verbosity=2).run(suite)

    # Visualize test cases
    print("\nGenerating visualizations...")
    visualize_test_case("circle_clusters")
    visualize_test_case("quantum_evolution")

    print("\nTest visualizations saved in output directory")

    return result.wasSuccessful()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)