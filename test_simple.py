import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer

def generate_simple_pattern():
    """Generate a simple quantum pattern for testing"""
    t = np.linspace(0, 2*np.pi, 50)

    # Create two interacting patterns
    pattern1 = np.column_stack([
        np.sin(t),
        np.cos(t),
        0.5 * np.sin(2*t)
    ])

    pattern2 = np.column_stack([
        -np.sin(t + np.pi/4),
        -np.cos(t + np.pi/4),
        0.5 * np.sin(2*t + np.pi/4)
    ])

    coordinates = np.vstack([pattern1, pattern2])

    # Add small quantum noise
    noise = 0.05 * np.random.randn(*coordinates.shape)
    coordinates += noise

    # Calculate velocities
    dt = t[1] - t[0]
    velocities = np.gradient(coordinates, dt, axis=0)

    return coordinates, velocities

def main():
    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    print("Analyzing Simple Quantum Pattern...")
    print("-" * 40)

    # Generate and analyze pattern
    coordinates, velocities = generate_simple_pattern()
    patterns = analyzer._analyze_emergent_patterns(coordinates, velocities)

    print(f"\nDetected {len(patterns)} emergent patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"- Complexity: {pattern.complexity:.3f}")
        print(f"- Stability: {pattern.stability:.3f}")
        print(f"- Influence Radius: {pattern.influence_radius:.3f}")
        print(f"- Interactions: {len(pattern.interaction_strength)} other patterns")

    # Get topological features
    features = analyzer._detect_topological_features(coordinates)
    print("\nTopological Features:")
    dims = {f.dimension for f in features}
    for dim in sorted(dims):
        dim_features = [f for f in features if f.dimension == dim]
        print(f"- {len(dim_features)} {dim}-dimensional features")

if __name__ == "__main__":
    main()