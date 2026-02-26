import numpy as np
import matplotlib.pyplot as plt
from src.core.hyperdimensional_analyzer import HyperDimensionalAnalyzer, HyperDimensionalState

class EntanglementAnalyzer:
    def __init__(self, dimensions=3):
        self.analyzer = HyperDimensionalAnalyzer(dimensions=dimensions)

    def generate_entangled_states(self, n_steps=50, noise_levels=[0.01, 0.05, 0.1]):
        """Generate entangled states with different noise levels"""
        t = np.linspace(0, 4*np.pi, n_steps)
        states_by_noise = {}

        for noise in noise_levels:
            # Base entangled pair
            p1 = np.array([
                np.sin(t),
                np.cos(t),
                0.5 * np.sin(2*t)
            ]).T

            # Entangled counterpart with phase relationship
            p2 = np.array([
                -np.sin(t + np.pi/6),
                -np.cos(t + np.pi/6),
                -0.5 * np.sin(2*t + np.pi/6)
            ]).T

            # Add quantum noise
            coords = np.vstack([p1, p2])
            noisy_coords = coords + noise * np.random.randn(*coords.shape)

            # Calculate velocities
            velocities = np.gradient(noisy_coords, t[1]-t[0], axis=0)

            states_by_noise[noise] = (noisy_coords, velocities)

        return states_by_noise, t

    def analyze_entanglement_strength(self, coordinates):
        """Analyze the strength of entanglement through correlation"""
        n_points = len(coordinates) // 2
        p1 = coordinates[:n_points]
        p2 = coordinates[n_points:]

        # Calculate correlation between particle positions
        correlations = []
        for dim in range(3):
            corr = np.corrcoef(p1[:, dim], p2[:, dim])[0, 1]
            correlations.append(abs(corr))

        return {
            'x_correlation': correlations[0],
            'y_correlation': correlations[1],
            'z_correlation': correlations[2],
            'mean_correlation': np.mean(correlations)
        }

    def analyze_quantum_coherence(self, coordinates, velocities):
        """Analyze quantum coherence through pattern detection"""
        patterns = self.analyzer._analyze_emergent_patterns(coordinates, velocities)
        features = self.analyzer._detect_topological_features(coordinates)

        # Analyze pattern properties
        pattern_info = []
        for pattern in patterns:
            info = {
                'complexity': pattern.complexity,
                'stability': pattern.stability,
                'influence_radius': pattern.influence_radius,
                'interactions': pattern.interaction_strength
            }
            pattern_info.append(info)

        # Analyze topological features
        topo_info = {}
        for feature in features:
            dim = feature.dimension
            if dim not in topo_info:
                topo_info[dim] = {'count': 0, 'total_persistence': 0}
            topo_info[dim]['count'] += 1
            topo_info[dim]['total_persistence'] += feature.persistence

        return pattern_info, topo_info

    def analyze_phase_space(self, coordinates, velocities):
        """Analyze phase space properties"""
        # Calculate phase space density estimation
        n_points = len(coordinates)
        density = np.zeros(n_points)

        for i in range(n_points):
            # Use position and velocity to define phase space point
            phase_point = np.concatenate([coordinates[i], velocities[i]])
            # Calculate local density using nearest neighbors
            distances = np.linalg.norm(coordinates - coordinates[i], axis=1)
            density[i] = np.sum(np.exp(-distances))

        return {
            'mean_density': np.mean(density),
            'density_std': np.std(density),
            'max_density': np.max(density),
            'min_density': np.min(density)
        }

def main():
    print("Detailed Quantum Entanglement Analysis")
    print("=" * 40)

    # Initialize analyzer
    analyzer = EntanglementAnalyzer()

    # Generate states with different noise levels
    noise_levels = [0.01, 0.05, 0.1]
    states_by_noise, t = analyzer.generate_entangled_states(n_steps=50, noise_levels=noise_levels)

    # Analyze each noise level
    for noise in noise_levels:
        print(f"\nAnalyzing entangled state with noise level {noise:.2f}")
        print("-" * 40)

        coordinates, velocities = states_by_noise[noise]

        # Analyze entanglement strength
        ent_strength = analyzer.analyze_entanglement_strength(coordinates)
        print("\nEntanglement Correlations:")
        print(f"X-axis: {ent_strength['x_correlation']:.3f}")
        print(f"Y-axis: {ent_strength['y_correlation']:.3f}")
        print(f"Z-axis: {ent_strength['z_correlation']:.3f}")
        print(f"Mean correlation: {ent_strength['mean_correlation']:.3f}")

        # Analyze quantum coherence
        pattern_info, topo_info = analyzer.analyze_quantum_coherence(coordinates, velocities)

        print("\nQuantum Patterns:")
        for i, pattern in enumerate(pattern_info):
            print(f"\nPattern {i+1}:")
            print(f"- Complexity: {pattern['complexity']:.3f}")
            print(f"- Stability: {pattern['stability']:.3f}")
            print(f"- Influence Radius: {pattern['influence_radius']:.3f}")
            print("- Interactions:")
            for k, v in pattern['interactions'].items():
                print(f"  * {k}: {v:.3f}")

        print("\nTopological Features:")
        for dim in sorted(topo_info.keys()):
            info = topo_info[dim]
            print(f"- {info['count']} {dim}-dimensional features")
            print(f"  Total persistence: {info['total_persistence']:.3f}")

        # Analyze phase space
        phase_info = analyzer.analyze_phase_space(coordinates, velocities)
        print("\nPhase Space Properties:")
        print(f"- Mean Density: {phase_info['mean_density']:.3f}")
        print(f"- Density Std: {phase_info['density_std']:.3f}")
        print(f"- Max Density: {phase_info['max_density']:.3f}")
        print(f"- Min Density: {phase_info['min_density']:.3f}")

if __name__ == "__main__":
    main()