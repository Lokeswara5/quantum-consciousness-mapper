import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import os
import json
from datetime import datetime

class ResultsAnalyzer:
    def __init__(self, output_dir='output'):
        self.output_dir = output_dir
        plt.style.use('dark_background')

    def load_metrics_data(self):
        """Extract metrics data from file names to reconstruct the time series"""
        metrics_files = [f for f in os.listdir(self.output_dir) if f.startswith('visualization_metrics_')]
        data = []

        for file in metrics_files:
            iteration = int(file.split('_')[-1].split('.')[0])
            # Extract metrics from PNG using simple image analysis
            img = plt.imread(os.path.join(self.output_dir, file))
            # Get the mean values of the RGB channels as a simple metric
            metrics = np.mean(img, axis=(0, 1))
            data.append({
                'iteration': iteration,
                'coherence': metrics[0],
                'entanglement': metrics[1],
                'field_strength': metrics[2]
            })

        return pd.DataFrame(data).sort_values('iteration')

    def analyze_quantum_patterns(self, data):
        """Analyze quantum patterns in the data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 1. Correlation Analysis
        correlation_matrix = data[['coherence', 'entanglement', 'field_strength']].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Quantum Property Correlations')

        # 2. Phase Space Analysis
        axes[0, 1].scatter(data['coherence'], data['entanglement'],
                         c=data['field_strength'], cmap='viridis')
        axes[0, 1].set_xlabel('Coherence')
        axes[0, 1].set_ylabel('Entanglement')
        axes[0, 1].set_title('Quantum Phase Space')

        # 3. Time Series Analysis
        data[['coherence', 'entanglement', 'field_strength']].plot(ax=axes[1, 0])
        axes[1, 0].set_title('Quantum Properties Evolution')
        axes[1, 0].set_xlabel('Measurement Index')
        axes[1, 0].set_ylabel('Value')

        # 4. Distribution Analysis
        for col in ['coherence', 'entanglement', 'field_strength']:
            sns.kdeplot(data[col], ax=axes[1, 1], label=col)
        axes[1, 1].set_title('Property Distributions')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quantum_pattern_analysis.png'))
        plt.close()

        return correlation_matrix

    def analyze_stability_patterns(self, data):
        """Analyze stability and fluctuation patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 1. Rolling Statistics
        window = 5
        rolling_mean = data[['coherence', 'entanglement', 'field_strength']].rolling(window).mean()
        rolling_std = data[['coherence', 'entanglement', 'field_strength']].rolling(window).std()

        rolling_mean.plot(ax=axes[0, 0])
        axes[0, 0].set_title(f'Rolling Mean (window={window})')
        axes[0, 0].set_xlabel('Measurement Index')
        axes[0, 0].set_ylabel('Value')

        rolling_std.plot(ax=axes[0, 1])
        axes[0, 1].set_title(f'Rolling Standard Deviation (window={window})')
        axes[0, 1].set_xlabel('Measurement Index')
        axes[0, 1].set_ylabel('Value')

        # 2. Stability Analysis
        stability_metrics = pd.DataFrame({
            'mean': data[['coherence', 'entanglement', 'field_strength']].mean(),
            'std': data[['coherence', 'entanglement', 'field_strength']].std(),
            'cv': data[['coherence', 'entanglement', 'field_strength']].std() /
                  data[['coherence', 'entanglement', 'field_strength']].mean()
        })

        stability_metrics.plot(kind='bar', ax=axes[1, 0])
        axes[1, 0].set_title('Stability Metrics')
        axes[1, 0].set_ylabel('Value')

        # 3. Peak Analysis
        for col in ['coherence', 'entanglement', 'field_strength']:
            peaks, _ = find_peaks(data[col])
            axes[1, 1].plot(data.index, data[col], label=f'{col} peaks')
            axes[1, 1].plot(peaks, data[col].iloc[peaks], "x")
        axes[1, 1].set_title('Peak Detection')
        axes[1, 1].legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'stability_analysis.png'))
        plt.close()

        return stability_metrics

    def analyze_emergent_patterns(self, data):
        """Analyze emergent patterns and transitions"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))

        # 1. Phase Transitions
        phase_space = np.column_stack([
            data['coherence'],
            data['entanglement'],
            data['field_strength']
        ])

        # Calculate phase space velocity
        velocity = np.diff(phase_space, axis=0)
        speed = np.linalg.norm(velocity, axis=1)

        axes[0, 0].plot(speed)
        axes[0, 0].set_title('Phase Space Velocity')
        axes[0, 0].set_xlabel('Time Step')
        axes[0, 0].set_ylabel('Velocity Magnitude')

        # 2. State Space Trajectory
        axes[0, 1].scatter(data['coherence'], data['entanglement'],
                         c=data.index, cmap='viridis')
        axes[0, 1].set_title('State Space Trajectory')
        axes[0, 1].set_xlabel('Coherence')
        axes[0, 1].set_ylabel('Entanglement')

        # 3. Recurrence Analysis
        distance_matrix = np.zeros((len(data), len(data)))
        for i in range(len(data)):
            for j in range(len(data)):
                distance_matrix[i, j] = np.linalg.norm(phase_space[i] - phase_space[j])

        sns.heatmap(distance_matrix, ax=axes[1, 0], cmap='viridis')
        axes[1, 0].set_title('Recurrence Plot')

        # 4. Entropy Evolution
        window = 5
        entropy = []
        for i in range(window, len(data)):
            window_data = phase_space[i-window:i]
            # Calculate approximate entropy using std dev as a proxy
            entropy.append(np.std(window_data))

        axes[1, 1].plot(range(window, len(data)), entropy)
        axes[1, 1].set_title('Local Entropy Evolution')
        axes[1, 1].set_xlabel('Time Step')
        axes[1, 1].set_ylabel('Approximate Entropy')

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'emergent_pattern_analysis.png'))
        plt.close()

        return {
            'average_speed': np.mean(speed),
            'max_speed': np.max(speed),
            'average_entropy': np.mean(entropy),
            'entropy_variation': np.std(entropy)
        }

    def generate_analysis_report(self):
        """Generate comprehensive analysis report"""
        print("\n=== Starting Detailed Analysis ===\n")

        # Load and analyze data
        print("Loading metrics data...")
        data = self.load_metrics_data()

        print("\nAnalyzing quantum patterns...")
        correlation_matrix = self.analyze_quantum_patterns(data)
        print("\nQuantum Property Correlations:")
        print(correlation_matrix)

        print("\nAnalyzing stability patterns...")
        stability_metrics = self.analyze_stability_patterns(data)
        print("\nStability Metrics:")
        print(stability_metrics)

        print("\nAnalyzing emergent patterns...")
        emergent_metrics = self.analyze_emergent_patterns(data)
        print("\nEmergent Pattern Metrics:")
        for key, value in emergent_metrics.items():
            print(f"{key}: {value:.4f}")

        # Save analysis results
        analysis_results = {
            'correlation_matrix': correlation_matrix.to_dict(),
            'stability_metrics': stability_metrics.to_dict(),
            'emergent_metrics': emergent_metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }

        with open(os.path.join(self.output_dir, 'analysis_results.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)

        print("\nAnalysis complete! Results saved to:")
        print(f"1. {self.output_dir}/quantum_pattern_analysis.png")
        print(f"2. {self.output_dir}/stability_analysis.png")
        print(f"3. {self.output_dir}/emergent_pattern_analysis.png")
        print(f"4. {self.output_dir}/analysis_results.json")

if __name__ == "__main__":
    analyzer = ResultsAnalyzer()
    analyzer.generate_analysis_report()