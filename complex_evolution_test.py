import numpy as np
import matplotlib.pyplot as plt
from src.core.evolution_tracker import EvolutionTracker
import seaborn as sns
from datetime import datetime
import pandas as pd
import os

class ComplexEvolutionTest:
    def __init__(self, dimensions=64):
        self.dimensions = dimensions
        self.tracker = EvolutionTracker(state_dim=dimensions)
        self.results = []
        # Create output directory
        os.makedirs('output', exist_ok=True)

    def generate_cyclic_state(self, t: float) -> np.ndarray:
        """Generate a state that follows a cyclic pattern"""
        base_state = np.sin(np.linspace(0, 4*np.pi, self.dimensions) + t)
        return base_state / np.linalg.norm(base_state)

    def generate_quantum_state(self, t: float) -> np.ndarray:
        """Generate a state with quantum-like behavior"""
        phase = np.exp(1j * t)
        state = np.array([np.cos(t), np.sin(t)] * (self.dimensions // 2))
        state = (state * phase).real
        return state / np.linalg.norm(state)

    def generate_chaotic_state(self, prev_state: np.ndarray) -> np.ndarray:
        """Generate a state using chaotic dynamics"""
        r = 3.9  # Chaos parameter
        state = r * prev_state * (1 - prev_state)
        return state / np.linalg.norm(state)

    def generate_emergent_state(self, t: float) -> np.ndarray:
        """Generate a state with emergent patterns"""
        # Combine multiple patterns
        wave1 = np.sin(np.linspace(0, 2*np.pi, self.dimensions) + t)
        wave2 = np.cos(np.linspace(0, 4*np.pi, self.dimensions) + 2*t)
        wave3 = np.sin(np.linspace(0, 6*np.pi, self.dimensions) + 3*t)

        state = wave1 + 0.5*wave2 + 0.25*wave3
        return state / np.linalg.norm(state)

    def plot_results(self):
        """Plot comprehensive analysis of results"""
        results_df = pd.DataFrame(self.results)

        # Create figure with smaller number of subplots
        fig = plt.figure(figsize=(15, 10))

        # 1. Quantum Properties Evolution
        ax1 = fig.add_subplot(221)
        for prop in ['coherence', 'entanglement', 'field_strength']:
            ax1.plot(results_df['iteration'], results_df[prop], label=prop)
        ax1.set_title('Quantum Properties Evolution')
        ax1.legend()

        # 2. Stability Score
        ax2 = fig.add_subplot(222)
        ax2.plot(results_df['iteration'], results_df['stability'], 'r-', label='Stability')
        ax2.set_title('System Stability')
        ax2.legend()

        # 3. State Space Trajectory (3D)
        ax3 = fig.add_subplot(223, projection='3d')
        scatter = ax3.scatter(results_df['state_x'],
                            results_df['state_y'],
                            results_df['state_z'],
                            c=range(len(results_df)),
                            cmap='viridis')
        ax3.set_title('State Space Trajectory')
        plt.colorbar(scatter, ax=ax3, label='Time')

        # 4. Event Timeline
        ax4 = fig.add_subplot(224)
        events_only = results_df[results_df['event_type'].notna()]
        if not events_only.empty:
            for idx, event_type in enumerate(events_only['event_type'].unique()):
                mask = events_only['event_type'] == event_type
                ax4.scatter(events_only[mask]['iteration'],
                           [idx] * sum(mask),
                           label=event_type,
                           s=50)
            ax4.set_yticks(range(len(events_only['event_type'].unique())))
            ax4.set_yticklabels(events_only['event_type'].unique())
        ax4.set_title('Event Timeline')
        ax4.set_xlabel('Iteration')

        plt.tight_layout()
        plt.savefig('output/evolution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

        # Create additional analysis plots
        fig2 = plt.figure(figsize=(15, 5))

        # 1. Phase Space
        ax1 = fig2.add_subplot(131)
        scatter = ax1.scatter(results_df['coherence'],
                            results_df['entanglement'],
                            c=results_df['field_strength'],
                            cmap='viridis')
        ax1.set_xlabel('Coherence')
        ax1.set_ylabel('Entanglement')
        ax1.set_title('Phase Space')
        plt.colorbar(scatter, ax=ax1, label='Field Strength')

        # 2. Stability Distribution
        ax2 = fig2.add_subplot(132)
        sns.histplot(data=results_df['stability'], bins=30, ax=ax2)
        ax2.set_title('Stability Distribution')

        # 3. Event Distribution
        ax3 = fig2.add_subplot(133)
        event_counts = results_df['event_type'].value_counts()
        if not event_counts.empty:
            event_counts.plot(kind='bar', ax=ax3)
            ax3.set_title('Event Distribution')

        plt.tight_layout()
        plt.savefig('output/evolution_analysis_2.png', dpi=300, bbox_inches='tight')
        plt.close()

        print("\nVisualizations saved to output/evolution_analysis.png and output/evolution_analysis_2.png")

    def run_test_scenario(self, iterations=100):
        """Run complex test scenario"""
        print("\nRunning Complex Evolution Test")
        print("==============================")

        # Initialize state
        current_state = np.random.rand(self.dimensions)
        current_state = current_state / np.linalg.norm(current_state)

        for i in range(iterations):
            # Progress indicator
            if i % 10 == 0:
                print(f"Iteration {i}/{iterations}")

            # Generate different types of states based on phase
            phase = i / iterations
            if phase < 0.25:
                # Cyclic phase
                new_state = self.generate_cyclic_state(i * 0.1)
            elif phase < 0.5:
                # Quantum phase
                new_state = self.generate_quantum_state(i * 0.1)
            elif phase < 0.75:
                # Chaotic phase
                new_state = self.generate_chaotic_state(current_state)
            else:
                # Emergent phase
                new_state = self.generate_emergent_state(i * 0.1)

            # Track state
            state_result = self.tracker.track_state(new_state)

            # Record results
            result = {
                'iteration': i,
                'coherence': state_result.quantum_properties['coherence'],
                'entanglement': state_result.quantum_properties['entanglement'],
                'field_strength': state_result.quantum_properties['field_strength'],
                'stability': state_result.stability_score,
                'topology': state_result.topology_signature,
                'state_x': new_state[0],
                'state_y': new_state[1],
                'state_z': new_state[2],
                'event_type': None
            }

            # Record events
            if len(self.tracker.event_history) > 0 and \
               len(self.tracker.event_history) > len(self.results):
                result['event_type'] = self.tracker.event_history[-1].event_type

            self.results.append(result)
            current_state = new_state

        # Get final summary
        summary = self.tracker.get_evolution_summary()

        print("\nTest Complete!")
        print("\nEvolution Summary:")
        print(f"Total States: {summary['total_states']}")
        print(f"Total Events: {summary['total_events']}")
        print(f"Average Stability: {summary['average_stability']:.3f}")
        print("\nEvent Distribution:")
        for event_type, count in summary['event_distribution'].items():
            print(f"{event_type}: {count}")

        # Generate visualizations
        print("\nGenerating visualizations...")
        self.plot_results()

if __name__ == "__main__":
    # Run test with different scenarios
    test = ComplexEvolutionTest(dimensions=64)
    test.run_test_scenario(iterations=200)