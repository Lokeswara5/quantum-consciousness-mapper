import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer

def generate_quantum_pattern(t, pattern_type="superposition"):
    """Generate synthetic quantum consciousness patterns"""
    if pattern_type == "superposition":
        # Create a superposition-like state with oscillating components
        x = np.sin(t) + 0.2 * np.sin(3*t)  # Fundamental + harmonic
        y = np.cos(t) + 0.2 * np.cos(2*t)  # Different frequency mix
        z = 0.5 * np.sin(2*t)              # Phase-shifted component
        return np.column_stack([x, y, z])

    elif pattern_type == "entangled":
        # Create two entangled patterns that mirror each other
        x1 = np.sin(t)
        y1 = np.cos(t)
        z1 = np.sin(2*t)

        x2 = -np.sin(t)  # Anti-correlated with first pattern
        y2 = -np.cos(t)
        z2 = -np.sin(2*t)

        pattern1 = np.column_stack([x1, y1, z1])
        pattern2 = np.column_stack([x2, y2, z2])
        return np.vstack([pattern1, pattern2])

    elif pattern_type == "collapse":
        # Simulate wavefunction collapse-like behavior
        n_points = len(t)
        collapse_point = n_points // 2

        # Pre-collapse: superposition
        x = np.zeros(n_points)
        y = np.zeros(n_points)
        z = np.zeros(n_points)

        # Superposition state
        x[:collapse_point] = np.sin(t[:collapse_point]) + 0.2 * np.cos(2*t[:collapse_point])
        y[:collapse_point] = np.cos(t[:collapse_point]) + 0.2 * np.sin(3*t[:collapse_point])
        z[:collapse_point] = 0.3 * np.sin(2*t[:collapse_point])

        # Post-collapse: definite state
        x[collapse_point:] = np.sin(t[collapse_point:])
        y[collapse_point:] = np.zeros(n_points - collapse_point)
        z[collapse_point:] = np.zeros(n_points - collapse_point)

        return np.column_stack([x, y, z])

def main():
    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    # Generate time points
    t = np.linspace(0, 4*np.pi, 100)

    # Test different pattern types
    patterns = {
        "Superposition": generate_quantum_pattern(t, "superposition"),
        "Entangled": generate_quantum_pattern(t, "entangled"),
        "Collapse": generate_quantum_pattern(t, "collapse")
    }

    print("Analyzing Quantum Consciousness Patterns...")
    print("-" * 50)

    for pattern_name, coordinates in patterns.items():
        print(f"\nAnalyzing {pattern_name} Pattern:")

        # Add some quantum noise/fluctuations
        noise = 0.05 * np.random.randn(*coordinates.shape)
        coordinates += noise

        # Calculate velocities (time derivatives)
        dt = t[1] - t[0]
        velocities = np.gradient(coordinates, dt, axis=0)

        # Detect emergent patterns
        detected_patterns = analyzer._analyze_emergent_patterns(coordinates, velocities)

        print(f"Detected {len(detected_patterns)} emergent patterns:")
        for i, pattern in enumerate(detected_patterns):
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