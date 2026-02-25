import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
from scipy.stats import entropy
import seaborn as sns
import os

class StabilityDropAnalyzer:
    def __init__(self):
        self.output_dir = 'output/stability_drops'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')

    def find_stability_drops(self, df, threshold=-0.1, window=5):
        """Identify and analyze stability drops"""
        # Calculate stability changes
        stability_diff = df['stability'].diff()

        # Find drops exceeding threshold
        drops = []
        for i in range(len(df)):
            if stability_diff.iloc[i] < threshold:
                # Extract window around drop
                start_idx = max(0, i - window)
                end_idx = min(len(df), i + window + 1)

                drop_info = {
                    'index': i,
                    'magnitude': stability_diff.iloc[i],
                    'window': df.iloc[start_idx:end_idx].copy(),
                    'pre_drop': df.iloc[start_idx:i].copy(),
                    'post_drop': df.iloc[i:end_idx].copy()
                }
                drops.append(drop_info)

        return drops

    def analyze_drop_characteristics(self, drop_info):
        """Analyze detailed characteristics of a stability drop"""
        window_data = drop_info['window']

        # Calculate recovery time
        stability_threshold = 0.9 * window_data['stability'].iloc[0]
        recovery_time = None
        for i, val in enumerate(window_data['stability'].iloc[1:]):
            if val >= stability_threshold:
                recovery_time = i
                break

        # Calculate quantum metrics during drop
        quantum_changes = {
            'coherence_change': window_data['coherence'].max() - window_data['coherence'].min(),
            'entanglement_change': window_data['entanglement'].max() - window_data['entanglement'].min(),
            'field_strength_change': window_data['field_strength'].max() - window_data['field_strength'].min()
        }

        # Calculate state space displacement
        state_displacement = np.sqrt(
            (window_data['state_x'].iloc[-1] - window_data['state_x'].iloc[0])**2 +
            (window_data['state_y'].iloc[-1] - window_data['state_y'].iloc[0])**2 +
            (window_data['state_z'].iloc[-1] - window_data['state_z'].iloc[0])**2
        )

        return {
            'recovery_time': recovery_time,
            'quantum_changes': quantum_changes,
            'state_displacement': state_displacement,
            'min_stability': window_data['stability'].min(),
            'stability_loss': window_data['stability'].iloc[0] - window_data['stability'].min()
        }

    def visualize_drop(self, drop_info, drop_num, characteristics):
        """Create detailed visualization for a stability drop"""
        fig = plt.figure(figsize=(20, 10))

        # 1. Stability and Quantum Properties
        ax1 = fig.add_subplot(231)
        window_data = drop_info['window']
        mid_point = len(drop_info['pre_drop'])

        # Plot with centered time axis
        time_axis = range(-mid_point, len(window_data) - mid_point)
        ax1.plot(time_axis, window_data['stability'], 'b-', label='Stability')
        ax1.plot(time_axis, window_data['coherence'], 'r-', label='Coherence')
        ax1.plot(time_axis, window_data['entanglement'], 'g-', label='Entanglement')
        ax1.axvline(x=0, color='white', linestyle='--', alpha=0.5)
        ax1.set_title(f'Drop {drop_num}: Quantum Properties')
        ax1.legend()

        # 2. State Space Trajectory
        ax2 = fig.add_subplot(232, projection='3d')
        ax2.plot(window_data['state_x'], window_data['state_y'], window_data['state_z'])
        ax2.scatter(window_data['state_x'].iloc[mid_point],
                   window_data['state_y'].iloc[mid_point],
                   window_data['state_z'].iloc[mid_point],
                   color='red', s=100, label='Drop Point')
        ax2.set_title('State Space Trajectory')

        # 3. Phase Space
        ax3 = fig.add_subplot(233)
        scatter = ax3.scatter(window_data['stability'],
                            window_data['coherence'],
                            c=range(len(window_data)),
                            cmap='viridis')
        plt.colorbar(scatter, label='Time')
        ax3.set_xlabel('Stability')
        ax3.set_ylabel('Coherence')
        ax3.set_title('Phase Space Evolution')

        # 4. Recovery Analysis
        ax4 = fig.add_subplot(234)
        stability_values = window_data['stability']
        recovery_profile = (stability_values - stability_values.min()) / \
                         (stability_values.iloc[0] - stability_values.min())
        ax4.plot(time_axis, recovery_profile, 'g-')
        ax4.set_title('Recovery Profile')
        ax4.set_ylabel('Recovery Progress')
        ax4.set_xlabel('Time Steps')

        # 5. Quantum Correlations
        ax5 = fig.add_subplot(235)
        correlations = window_data[['stability', 'coherence', 'entanglement']].corr()
        sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax5)
        ax5.set_title('Property Correlations')

        # 6. Key Metrics
        ax6 = fig.add_subplot(236)
        ax6.axis('off')
        metrics_text = (
            f"Drop Characteristics:\n"
            f"Recovery Time: {characteristics['recovery_time']} steps\n"
            f"Stability Loss: {characteristics['stability_loss']:.3f}\n"
            f"Min Stability: {characteristics['min_stability']:.3f}\n"
            f"State Displacement: {characteristics['state_displacement']:.3f}\n"
            f"\nQuantum Changes:\n"
            f"Coherence: {characteristics['quantum_changes']['coherence_change']:.3f}\n"
            f"Entanglement: {characteristics['quantum_changes']['entanglement_change']:.3f}\n"
            f"Field Strength: {characteristics['quantum_changes']['field_strength_change']:.3f}"
        )
        ax6.text(0.1, 0.9, metrics_text, fontsize=10, va='top')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/drop_{drop_num}_analysis.png')
        plt.close()

    def analyze_all_drops(self, df):
        """Analyze all stability drops in the data"""
        print("\nAnalyzing Individual Stability Drops")
        print("==================================")

        # Find all drops
        drops = self.find_stability_drops(df)
        print(f"\nFound {len(drops)} significant stability drops")

        # Analyze each drop
        drop_characteristics = []
        for i, drop in enumerate(drops, 1):
            print(f"\nAnalyzing drop {i}:")

            # Calculate characteristics
            characteristics = self.analyze_drop_characteristics(drop)
            drop_characteristics.append(characteristics)

            # Print summary
            print(f"Time index: {drop['index']}")
            print(f"Magnitude: {drop['magnitude']:.3f}")
            print(f"Recovery time: {characteristics['recovery_time']} steps")
            print(f"Stability loss: {characteristics['stability_loss']:.3f}")

            # Create visualization
            self.visualize_drop(drop, i, characteristics)
            print(f"Visualization saved as drop_{i}_analysis.png")

        # Create summary visualization
        self.create_drops_summary(drops, drop_characteristics)

        return drops, drop_characteristics

    def create_drops_summary(self, drops, characteristics):
        """Create summary visualization of all drops"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Drop Magnitudes
        ax1 = fig.add_subplot(221)
        magnitudes = [drop['magnitude'] for drop in drops]
        ax1.bar(range(1, len(drops) + 1), magnitudes)
        ax1.set_title('Drop Magnitudes')
        ax1.set_xlabel('Drop Number')
        ax1.set_ylabel('Magnitude')

        # 2. Recovery Times
        ax2 = fig.add_subplot(222)
        recovery_times = [c['recovery_time'] for c in characteristics]
        ax2.bar(range(1, len(drops) + 1), recovery_times)
        ax2.set_title('Recovery Times')
        ax2.set_xlabel('Drop Number')
        ax2.set_ylabel('Steps to Recovery')

        # 3. Quantum Changes
        ax3 = fig.add_subplot(223)
        quantum_changes = pd.DataFrame([c['quantum_changes'] for c in characteristics])
        quantum_changes.plot(kind='bar', ax=ax3)
        ax3.set_title('Quantum Property Changes')
        ax3.set_xlabel('Drop Number')
        ax3.set_ylabel('Change Magnitude')
        plt.xticks(rotation=45)

        # 4. State Displacements
        ax4 = fig.add_subplot(224)
        displacements = [c['state_displacement'] for c in characteristics]
        ax4.bar(range(1, len(drops) + 1), displacements)
        ax4.set_title('State Space Displacements')
        ax4.set_xlabel('Drop Number')
        ax4.set_ylabel('Displacement')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/drops_summary.png')
        plt.close()

if __name__ == "__main__":
    from complex_evolution_test import ComplexEvolutionTest

    # Run evolution test
    print("Running evolution test...")
    test = ComplexEvolutionTest(dimensions=64)
    test.run_test_scenario(iterations=200)

    # Analyze stability drops
    analyzer = StabilityDropAnalyzer()
    analyzer.analyze_all_drops(pd.DataFrame(test.results))