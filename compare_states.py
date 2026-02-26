import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer

def generate_ghz_state(t, noise=0.05):
    """Generate GHZ state with 3 particles"""
    n_steps = len(t)
    coords = np.zeros((n_steps * 3, 3))  # 3 particles, 3 dimensions

    # First particle
    coords[:n_steps, 0] = np.sin(t)
    coords[:n_steps, 1] = np.cos(t)
    coords[:n_steps, 2] = np.cos(2*t)

    # Second particle (synchronized with first)
    coords[n_steps:2*n_steps, 0] = np.sin(t)
    coords[n_steps:2*n_steps, 1] = np.cos(t)
    coords[n_steps:2*n_steps, 2] = -np.cos(2*t)  # Anti-correlated in z

    # Third particle (also synchronized)
    coords[2*n_steps:, 0] = np.sin(t)
    coords[2*n_steps:, 1] = np.cos(t)
    coords[2*n_steps:, 2] = np.cos(2*t)

    return coords + noise * np.random.randn(*coords.shape)

def generate_w_state(t, noise=0.05):
    """Generate W state with 3 particles"""
    n_steps = len(t)
    coords = np.zeros((n_steps * 3, 3))

    # Particles take turns being in excited state
    phases = [0, 2*np.pi/3, 4*np.pi/3]
    for i in range(3):
        start_idx = i * n_steps
        end_idx = (i + 1) * n_steps
        phase = phases[i]

        coords[start_idx:end_idx, 0] = np.sin(t + phase)
        coords[start_idx:end_idx, 1] = np.cos(t + phase)
        coords[start_idx:end_idx, 2] = np.sin(2*t + phase)

    return coords + noise * np.random.randn(*coords.shape)

def analyze_state(coordinates, state_type, analyzer):
    """Analyze quantum state properties"""
    print(f"\n{'-'*20} {state_type} State {'-'*20}")

    # Calculate velocities
    t = np.linspace(0, 2*np.pi, len(coordinates)//3)
    velocities = np.gradient(coordinates, t[1]-t[0], axis=0)

    # Analyze patterns
    patterns = analyzer._analyze_emergent_patterns(coordinates, velocities)
    print(f"\nDetected {len(patterns)} emergent patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"- Complexity: {pattern.complexity:.3f}")
        print(f"- Stability: {pattern.stability:.3f}")
        print(f"- Influence Radius: {pattern.influence_radius:.3f}")

    # Analyze topology
    features = analyzer._detect_topological_features(coordinates)
    print("\nTopological Features:")
    dims = {f.dimension for f in features}
    for dim in sorted(dims):
        dim_features = [f for f in features if f.dimension == dim]
        persistence_sum = sum(f.persistence for f in dim_features)
        print(f"- {len(dim_features)} {dim}-dimensional features")
        print(f"  Total persistence: {persistence_sum:.3f}")

    # Calculate correlations between particles
    n_steps = len(coordinates) // 3
    p1 = coordinates[:n_steps]
    p2 = coordinates[n_steps:2*n_steps]
    p3 = coordinates[2*n_steps:]

    corr_12 = np.mean([np.corrcoef(p1[:,i], p2[:,i])[0,1] for i in range(3)])
    corr_23 = np.mean([np.corrcoef(p2[:,i], p3[:,i])[0,1] for i in range(3)])
    corr_13 = np.mean([np.corrcoef(p1[:,i], p3[:,i])[0,1] for i in range(3)])

    print("\nParticle Correlations:")
    print(f"1-2: {abs(corr_12):.3f}")
    print(f"2-3: {abs(corr_23):.3f}")
    print(f"1-3: {abs(corr_13):.3f}")

def main():
    print("Comparing GHZ and W States")
    print("=" * 40)

    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    # Generate states
    t = np.linspace(0, 4*np.pi, 50)

    # Compare states at different noise levels
    for noise in [0.01, 0.05, 0.1]:
        print(f"\nNoise level: {noise:.2f}")
        print("=" * 40)

        # Generate and analyze GHZ state
        ghz_coords = generate_ghz_state(t, noise)
        analyze_state(ghz_coords, "GHZ", analyzer)

        # Generate and analyze W state
        w_coords = generate_w_state(t, noise)
        analyze_state(w_coords, "W", analyzer)

if __name__ == "__main__":
    main()