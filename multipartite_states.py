import numpy as np
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

class MultipartiteStateAnalyzer:
    def __init__(self, dimensions=3):
        self.analyzer = HyperDimensionalAnalyzer(dimensions=dimensions)

    def generate_ghz_state(self, t, n_particles=3, noise_level=0.05):
        """
        Generate GHZ state: (|000⟩ + |111⟩)/√2
        Spatially represented as synchronized oscillations in superposition
        """
        n_steps = len(t)
        coordinates = np.zeros((n_steps * n_particles, 3))

        for i in range(n_particles):
            start_idx = i * n_steps
            end_idx = (i + 1) * n_steps

            # Each particle oscillates between two states
            coordinates[start_idx:end_idx, 0] = np.cos(t + i * 2*np.pi/n_particles)
            coordinates[start_idx:end_idx, 1] = np.sin(t + i * 2*np.pi/n_particles)
            coordinates[start_idx:end_idx, 2] = np.cos(2*t) * (-1)**i  # Alternating z-phase

        # Add quantum noise
        coordinates += noise_level * np.random.randn(*coordinates.shape)
        return coordinates

    def generate_w_state(self, t, n_particles=3, noise_level=0.05):
        """
        Generate W state: (|100⟩ + |010⟩ + |001⟩)/√3
        Spatially represented as excitations distributed among particles
        """
        n_steps = len(t)
        coordinates = np.zeros((n_steps * n_particles, 3))

        for i in range(n_particles):
            start_idx = i * n_steps
            end_idx = (i + 1) * n_steps

            # Each particle takes turns being in the excited state
            phase = 2*np.pi*i/n_particles
            coordinates[start_idx:end_idx, 0] = np.sin(t + phase)
            coordinates[start_idx:end_idx, 1] = np.cos(t + phase)
            coordinates[start_idx:end_idx, 2] = np.sin(2*t + phase)

        # Add quantum noise
        coordinates += noise_level * np.random.randn(*coordinates.shape)
        return coordinates

    def calculate_correlations(self, coordinates, n_particles):
        """Calculate quantum correlations between particles"""
        n_steps = len(coordinates) // n_particles
        particles = [coordinates[i*n_steps:(i+1)*n_steps] for i in range(n_particles)]

        correlations = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                # Calculate correlations for each dimension
                for dim in range(3):
                    corr = np.corrcoef(particles[i][:, dim], particles[j][:, dim])[0, 1]
                    correlations.append(abs(corr))

        return np.array(correlations)

    def analyze_state_properties(self, coordinates, velocities, state_type, n_particles=3):
        """Analyze quantum state properties"""
        print(f"\n{'-'*20} {state_type} State Analysis {'-'*20}")

        # Calculate quantum correlations
        correlations = self.calculate_correlations(coordinates, n_particles)
        print("\nQuantum Correlations:")
        print(f"Mean correlation: {np.mean(correlations):.3f}")
        print(f"Max correlation: {np.max(correlations):.3f}")
        print(f"Min correlation: {np.min(correlations):.3f}")

        # Analyze emergent patterns
        patterns = self.analyzer._analyze_emergent_patterns(coordinates, velocities)
        print(f"\nDetected {len(patterns)} emergent patterns:")

        for i, pattern in enumerate(patterns):
            print(f"\nPattern {i+1}:")
            print(f"- Complexity: {pattern.complexity:.3f}")
            print(f"- Stability: {pattern.stability:.3f}")
            print(f"- Influence Radius: {pattern.influence_radius:.3f}")
            print("- Interactions:")
            for k, v in pattern.interaction_strength.items():
                print(f"  * {k}: {v:.3f}")

        # Analyze topological features
        features = self.analyzer._detect_topological_features(coordinates)
        print("\nTopological Features:")
        dims = {f.dimension for f in features}
        for dim in sorted(dims):
            dim_features = [f for f in features if f.dimension == dim]
            persistence_sum = sum(f.persistence for f in dim_features)
            print(f"- {len(dim_features)} {dim}-dimensional features")
            print(f"  Total persistence: {persistence_sum:.3f}")

        # Calculate coherence properties
        coherence = self._calculate_coherence_metrics(coordinates, n_particles)
        print("\nQuantum Coherence Metrics:")
        print(f"- Global phase coherence: {coherence['global_phase']:.3f}")
        print(f"- Spatial coherence: {coherence['spatial']:.3f}")
        print(f"- Temporal stability: {coherence['temporal']:.3f}")

    def _calculate_coherence_metrics(self, coordinates, n_particles):
        """Calculate various quantum coherence metrics"""
        n_steps = len(coordinates) // n_particles

        # Calculate global phase coherence
        phase_diffs = []
        for i in range(n_particles):
            start_idx = i * n_steps
            end_idx = (i + 1) * n_steps
            phases = np.arctan2(coordinates[start_idx:end_idx, 1],
                              coordinates[start_idx:end_idx, 0])
            phase_diffs.extend([np.std(np.diff(phases))])
        global_phase = 1.0 / (1.0 + np.mean(phase_diffs))

        # Calculate spatial coherence
        spatial_corr = []
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                corr = np.corrcoef(coordinates[i*n_steps:(i+1)*n_steps].flatten(),
                                 coordinates[j*n_steps:(j+1)*n_steps].flatten())[0,1]
                spatial_corr.append(abs(corr))
        spatial_coherence = np.mean(spatial_corr)

        # Calculate temporal stability
        velocities = np.gradient(coordinates, axis=0)
        temporal_stability = 1.0 / (1.0 + np.std(velocities))

        return {
            'global_phase': float(global_phase),
            'spatial': float(spatial_coherence),
            'temporal': float(temporal_stability)
        }

def main():
    print("Multipartite Quantum State Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = MultipartiteStateAnalyzer()

    # Time evolution
    t = np.linspace(0, 4*np.pi, 50)

    # Analyze different noise levels
    noise_levels = [0.01, 0.05, 0.1]

    for noise in noise_levels:
        print(f"\nAnalyzing states with noise level {noise:.2f}")
        print("=" * 40)

        # Generate and analyze GHZ state
        coords_ghz = analyzer.generate_ghz_state(t, n_particles=3, noise_level=noise)
        velocities_ghz = np.gradient(coords_ghz, t[1]-t[0], axis=0)
        analyzer.analyze_state_properties(coords_ghz, velocities_ghz, "GHZ")

        # Generate and analyze W state
        coords_w = analyzer.generate_w_state(t, n_particles=3, noise_level=noise)
        velocities_w = np.gradient(coords_w, t[1]-t[0], axis=0)
        analyzer.analyze_state_properties(coords_w, velocities_w, "W")

if __name__ == "__main__":
    main()