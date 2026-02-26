import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer

def analyze_quantum_state(name, coordinates, velocities):
    """Basic analysis of a quantum state"""
    print(f"\n{'-'*20} {name} State {'-'*20}")

    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    # Analyze patterns
    patterns = analyzer._analyze_emergent_patterns(coordinates, velocities)

    print(f"Found {len(patterns)} emergent patterns")
    for i, p in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"Complexity: {p.complexity:.3f}")
        print(f"Stability: {p.stability:.3f}")

    # Simple correlation analysis
    n_steps = len(coordinates) // 3
    particles = [
        coordinates[:n_steps],
        coordinates[n_steps:2*n_steps],
        coordinates[2*n_steps:]
    ]

    print("\nParticle Correlations:")
    for i in range(3):
        for j in range(i+1, 3):
            corr = np.mean([
                np.corrcoef(particles[i][:,d], particles[j][:,d])[0,1]
                for d in range(3)
            ])
            print(f"Particles {i+1}-{j+1}: {abs(corr):.3f}")

def main():
    print("Simple Quantum State Comparison")
    print("=" * 40)

    # Time points
    t = np.linspace(0, 4*np.pi, 50)
    dt = t[1] - t[0]

    # GHZ State (maximally entangled)
    ghz_coords = np.zeros((150, 3))  # 3 particles
    for i in range(3):
        start = i * 50
        end = (i + 1) * 50
        ghz_coords[start:end, 0] = np.sin(t)
        ghz_coords[start:end, 1] = np.cos(t)
        ghz_coords[start:end, 2] = (-1)**i * np.cos(2*t)

    ghz_vel = np.gradient(ghz_coords, dt, axis=0)

    # W State (distributed entanglement)
    w_coords = np.zeros((150, 3))  # 3 particles
    for i in range(3):
        start = i * 50
        end = (i + 1) * 50
        phase = 2*np.pi*i/3
        w_coords[start:end, 0] = np.sin(t + phase)
        w_coords[start:end, 1] = np.cos(t + phase)
        w_coords[start:end, 2] = np.sin(2*t + phase)

    w_vel = np.gradient(w_coords, dt, axis=0)

    # Add small quantum noise
    noise = 0.05
    ghz_coords += noise * np.random.randn(*ghz_coords.shape)
    w_coords += noise * np.random.randn(*w_coords.shape)

    # Analyze both states
    analyze_quantum_state("GHZ", ghz_coords, ghz_vel)
    analyze_quantum_state("W", w_coords, w_vel)

if __name__ == "__main__":
    main()