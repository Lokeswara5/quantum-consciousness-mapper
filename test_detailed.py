import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

def generate_quantum_states(n_steps=50):
    """Generate quantum-like states with varying coherence"""
    t = np.linspace(0, 4*np.pi, n_steps)
    states = []

    for i in range(n_steps):
        # Base oscillation with quantum phase
        phase = t[i]
        coherence = np.exp(-0.1 * t[i])  # Gradual decoherence

        # Generate quantum-like coordinates
        coordinates = np.array([
            [np.sin(phase) * coherence, np.cos(phase) * coherence, 0.5],
            [-np.sin(phase) * coherence, -np.cos(phase) * coherence, -0.5],
            [np.cos(2*phase) * coherence, np.sin(2*phase) * coherence, 0.0]
        ])

        # Add quantum fluctuations
        noise = 0.05 * np.random.randn(*coordinates.shape)
        coordinates += noise

        # Calculate velocities and accelerations
        if i > 0:
            velocity = (coordinates - prev_coords) / (t[1] - t[0])
            acceleration = (velocity - prev_vel) / (t[1] - t[0]) if i > 1 else np.zeros_like(coordinates)
        else:
            velocity = np.zeros_like(coordinates)
            acceleration = np.zeros_like(coordinates)

        # Create quantum state
        state = HyperDimensionalState(
            coordinates=coordinates,
            velocity=velocity,
            acceleration=acceleration,
            phase_space_density=coherence,
            topological_features=[],
            emergent_patterns=[]
        )
        states.append(state)

        # Store previous values
        prev_coords = coordinates.copy()
        prev_vel = velocity.copy()

    return states

def analyze_quantum_dynamics():
    print("Analyzing Quantum Consciousness Dynamics")
    print("-" * 40)

    # Initialize analyzer
    analyzer = HyperDimensionalAnalyzer(dimensions=3)

    # Generate quantum states
    states = generate_quantum_states()
    analyzer.state_history = states

    # Analyze final state for patterns
    final_state = states[-1]
    patterns = analyzer._analyze_emergent_patterns(
        final_state.coordinates,
        final_state.velocity
    )

    print("\nQuantum State Analysis:")
    print(f"Number of states: {len(states)}")
    print(f"Final phase space density: {final_state.phase_space_density:.3f}")

    print("\nEmergent Patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i+1}:")
        print(f"- Complexity: {pattern.complexity:.3f}")
        print(f"- Stability: {pattern.stability:.3f}")
        print(f"- Influence Radius: {pattern.influence_radius:.3f}")
        for pattern_id, strength in pattern.interaction_strength.items():
            print(f"- Interaction with {pattern_id}: {strength:.3f}")

    # Analyze topological features across time
    print("\nTopological Evolution:")
    for i in [0, len(states)//2, -1]:  # Start, middle, end
        features = analyzer._detect_topological_features(states[i].coordinates)
        print(f"\nTime step {i}:")
        dims = {f.dimension for f in features}
        for dim in sorted(dims):
            dim_features = [f for f in features if f.dimension == dim]
            persistence_sum = sum(f.persistence for f in dim_features)
            print(f"- {len(dim_features)} {dim}-dimensional features")
            print(f"  Total persistence: {persistence_sum:.3f}")

if __name__ == "__main__":
    analyze_quantum_dynamics()