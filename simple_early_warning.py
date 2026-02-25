import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import seaborn as sns
import os

class EarlyWarningAnalyzer:
    def __init__(self):
        self.output_dir = 'output/warning_patterns'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')
        self.warning_window = 20

    def analyze_window(self, window_data):
        """Analyze a pre-drop window for warning patterns"""
        stability = window_data['stability'].values
        coherence = window_data['coherence'].values
        entanglement = window_data['entanglement'].values
        time_steps = np.arange(len(stability))

        # Calculate warning indicators
        variance = pd.Series(stability).rolling(5).var().fillna(0)
        autocorr = pd.Series(stability).rolling(5).apply(
            lambda x: x.autocorr() if len(x.dropna()) > 1 else 0
        ).fillna(0)
        delta = np.abs(np.gradient(stability))

        # Calculate warning score
        warning_score = (variance / variance.max() +
                        autocorr +
                        delta / delta.max()) / 3

        return {
            'stability': stability,
            'coherence': coherence,
            'entanglement': entanglement,
            'variance': variance,
            'autocorr': autocorr,
            'delta': delta,
            'warning_score': warning_score,
            'time_steps': time_steps
        }

    def plot_analysis(self, metrics, drop_num):
        """Create visualization of warning patterns"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Early Warning Analysis - Drop {drop_num}')

        # 1. System Properties
        ax = axes[0, 0]
        ax.plot(metrics['time_steps'], metrics['stability'],
               'b-', label='Stability')
        ax.plot(metrics['time_steps'], metrics['coherence'],
               'r-', label='Coherence')
        ax.plot(metrics['time_steps'], metrics['entanglement'],
               'g-', label='Entanglement')
        ax.set_title('System Properties')
        ax.set_xlabel('Time Steps Before Drop')
        ax.legend()

        # 2. Warning Indicators
        ax = axes[0, 1]
        ax.plot(metrics['time_steps'], metrics['variance'],
               label='Variance')
        ax.plot(metrics['time_steps'], metrics['autocorr'],
               label='Autocorrelation')
        ax.plot(metrics['time_steps'], metrics['delta'],
               label='Rate of Change')
        ax.set_title('Warning Indicators')
        ax.legend()

        # 3. Warning Score
        ax = axes[1, 0]
        ax.plot(metrics['time_steps'], metrics['warning_score'],
               'r-', label='Warning Score')
        ax.axhline(y=0.7, color='y', linestyle='--',
                  label='Warning Threshold')
        ax.set_title('Early Warning Score')
        ax.legend()

        # 4. Phase Space
        ax = axes[1, 1]
        ax.scatter(metrics['stability'][:-1], metrics['stability'][1:],
                  c=metrics['time_steps'][:-1], cmap='viridis')
        ax.set_xlabel('Stability(t)')
        ax.set_ylabel('Stability(t+1)')
        ax.set_title('Phase Space')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/early_warning_{drop_num}.png')
        plt.close()

    def create_summary(self, all_metrics):
        """Create summary of warning patterns across drops"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Early Warning Patterns Summary')

        # 1. Warning Score Comparison
        ax = axes[0, 0]
        for i, metrics in enumerate(all_metrics):
            ax.plot(metrics['warning_score'],
                   label=f'Drop {i+1}')
        ax.axhline(y=0.7, color='y', linestyle='--',
                  label='Warning Threshold')
        ax.set_title('Warning Score Comparison')
        ax.legend()

        # 2. System Stability
        ax = axes[0, 1]
        for i, metrics in enumerate(all_metrics):
            ax.plot(metrics['stability'],
                   label=f'Drop {i+1}')
        ax.set_title('Stability Patterns')
        ax.legend()

        # 3. Warning Duration
        ax = axes[1, 0]
        warning_durations = [
            np.sum(m['warning_score'] > 0.7) / len(m['warning_score'])
            for m in all_metrics
        ]
        ax.bar(range(1, len(warning_durations) + 1),
               warning_durations)
        ax.set_title('Warning Duration Ratio')
        ax.set_xlabel('Drop Number')
        ax.set_ylabel('Ratio of Time in Warning State')

        # 4. Warning Intensity
        ax = axes[1, 1]
        warning_max = [m['warning_score'].max() for m in all_metrics]
        warning_mean = [m['warning_score'].mean() for m in all_metrics]
        x = range(1, len(warning_max) + 1)
        ax.bar(x, warning_max, alpha=0.5, label='Max Warning')
        ax.bar(x, warning_mean, alpha=0.5, label='Mean Warning')
        ax.set_title('Warning Intensity')
        ax.set_xlabel('Drop Number')
        ax.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/warning_summary.png')
        plt.close()

    def run_analysis(self, df, drops):
        """Run early warning analysis for all drops"""
        print("\nAnalyzing Early Warning Patterns")
        print("==============================")

        all_metrics = []

        for i, drop in enumerate(drops, 1):
            print(f"\nAnalyzing drop {i}:")

            # Extract pre-drop window
            start_idx = max(0, drop['index'] - self.warning_window)
            window_data = df.iloc[start_idx:drop['index']].copy()

            # Analyze warning patterns
            metrics = self.analyze_window(window_data)
            all_metrics.append(metrics)

            # Plot analysis
            self.plot_analysis(metrics, i)

            # Print summary
            warning_duration = np.sum(metrics['warning_score'] > 0.7) / len(metrics['warning_score'])
            print(f"Warning pattern analysis completed:")
            print(f"- Maximum warning score: {metrics['warning_score'].max():.3f}")
            print(f"- Warning duration ratio: {warning_duration:.3f}")
            print(f"- Analysis saved as early_warning_{i}.png")

        # Create summary
        print("\nGenerating summary analysis...")
        self.create_summary(all_metrics)
        print("Summary saved as warning_summary.png")

        return all_metrics

if __name__ == "__main__":
    from complex_evolution_test import ComplexEvolutionTest
    from stability_drop_analyzer import StabilityDropAnalyzer

    # Run evolution test
    print("Running evolution test...")
    test = ComplexEvolutionTest(dimensions=64)
    test.run_test_scenario(iterations=200)

    # Find stability drops
    drop_analyzer = StabilityDropAnalyzer()
    drops = drop_analyzer.find_stability_drops(pd.DataFrame(test.results))

    # Analyze warning patterns
    warning_analyzer = EarlyWarningAnalyzer()
    warning_analyzer.run_analysis(pd.DataFrame(test.results), drops)