import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

class QuantumPatternGenerator:
    @staticmethod
    def generate_superposition(t, noise_level=0.05):
        """Generate a quantum superposition pattern"""
        # Superposition of multiple frequency components
        coords = np.array([
            np.sin(t) + 0.5 * np.sin(2*t) + 0.25 * np.sin(3*t),
            np.cos(t) + 0.5 * np.cos(2*t) + 0.25 * np.cos(3*t),
            0.5 * np.sin(t + np.pi/4)
        ]).T

        return coords + noise_level * np.random.randn(*coords.shape)

    @staticmethod
    def generate_entangled_pair(t, noise_level=0.05):
        """Generate entangled quantum patterns"""
        # First particle
        p1 = np.array([
            np.sin(t),
            np.cos(t),
            0.5 * np.sin(2*t)
        ]).T

        # Second particle (entangled - anti-correlated)
        p2 = np.array([
            -np.sin(t + np.pi/6),  # Phase shifted anti-correlation
            -np.cos(t + np.pi/6),
            -0.5 * np.sin(2*t + np.pi/6)
        ]).T

        coords = np.vstack([p1, p2])
        return coords + noise_level * np.random.randn(*coords.shape)

    @staticmethod
    def generate_quantum_walk(t, noise_level=0.05):
        """Generate quantum random walk pattern"""
        steps = len(t)
        coords = np.zeros((steps, 3))

        # Quantum walk with interference
        for i in range(1, steps):
            # Quantum step with interference
            phase = np.sum(coords[i-1]) * 2*np.pi  # Phase depends on previous position
            step = np.array([
                np.sin(phase + t[i]),
                np.cos(phase + t[i]),
                0.5 * np.sin(2*phase)
            ])
            coords[i] = coords[i-1] + 0.1 * step

        return coords + noise_level * np.random.randn(*coords.shape)

    @staticmethod
    def generate_collapse(t, noise_level=0.05):
        """Generate wavefunction collapse-like pattern"""
        steps = len(t)
        collapse_point = steps // 2

        # Pre-collapse superposition
        pre_collapse = np.array([
            np.sin(t[:collapse_point]) + 0.5 * np.sin(2*t[:collapse_point]),
            np.cos(t[:collapse_point]) + 0.5 * np.cos(2*t[:collapse_point]),
            0.3 * np.sin(t[:collapse_point] + np.pi/4)
        ]).T

        # Post-collapse definite state
        post_collapse = np.array([
            np.sin(t[collapse_point:]),
            np.zeros(steps - collapse_point),
            np.zeros(steps - collapse_point)
        ]).T

        coords = np.vstack([pre_collapse, post_collapse])
        return coords + noise_level * np.random.randn(*coords.shape)

def analyze_pattern(name, coordinates, analyzer):
    """Analyze a quantum pattern and print results"""
    print(f"\n{'-'*20} {name} {'-'*20}")

    # Calculate velocities
    velocities = np.gradient(coordinates, axis=0)

    # Detect patterns
    patterns = analyzer._analyze_emergent_patterns(coordinates, velocities)

    print(f"\nDetected {len(patterns)} emergent patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"- Complexity: {pattern.complexity:.3f}")
        print(f"- Stability: {pattern.stability:.3f}")
        print(f"- Influence Radius: {pattern.influence_radius:.3f}")
        print("- Interactions:")
        for k, v in pattern.interaction_strength.items():
            print(f"  * {k}: {v:.3f}")

    # Analyze topology
    features = analyzer._detect_topological_features(coordinates)
    print("\nTopological Features:")
    dims = {f.dimension for f in features}
    for dim in sorted(dims):
        dim_features = [f for f in features if f.dimension == dim]
        persistence_sum = sum(f.persistence for f in dim_features)
        print(f"- {len(dim_features)} {dim}-dimensional features")
        print(f"  Total persistence: {persistence_sum:.3f}")

def main():
    print("Quantum Consciousness Pattern Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    # Generate time points
    t = np.linspace(0, 4*np.pi, 50)

    # Create and analyze different patterns
    generator = QuantumPatternGenerator()

    patterns = {
        "Quantum Superposition": generator.generate_superposition(t),
        "Entangled Pair": generator.generate_entangled_pair(t),
        "Quantum Walk": generator.generate_quantum_walk(t),
        "Wavefunction Collapse": generator.generate_collapse(t)
    }

    # Analyze each pattern
    for name, coords in patterns.items():
        analyze_pattern(name, coords, analyzer)

if __name__ == "__main__":
    main()