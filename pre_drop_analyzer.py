import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

class PreDropAnalyzer:
    def __init__(self):
        self.output_dir = 'output/pre_drop_analysis'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')
        self.pre_window = 20  # Analysis window before drop

    def extract_pre_drop_windows(self, df, drops, window_size=20):
        """Extract windows of data before each stability drop"""
        pre_drop_windows = []
        for drop in drops:
            start_idx = max(0, drop['index'] - window_size)
            window = df.iloc[start_idx:drop['index']].copy()
            window['time_to_drop'] = range(-len(window), 0)
            pre_drop_windows.append(window)
        return pre_drop_windows

    def analyze_early_warnings(self, window_data):
        """Analyze early warning signals in pre-drop window"""
        # Calculate statistical indicators
        rolling = window_data['stability'].rolling(window=5)

        metrics = {
            'variance': rolling.var(),
            'autocorrelation': rolling.apply(lambda x: pd.Series(x).autocorr()),
            'skewness': rolling.skew(),
            'kurtosis': rolling.kurt(),
            'detrended_fluctuation': self._calculate_dfa(window_data['stability'])
        }

        # Calculate rate of change
        metrics['stability_roc'] = window_data['stability'].diff()

        return pd.DataFrame(metrics)

    def _calculate_dfa(self, data, scales=None):
        """Calculate Detrended Fluctuation Analysis"""
        if scales is None:
            scales = np.logspace(1, len(data)//4, 8).astype(int)

        # Calculate profile
        profile = np.cumsum(data - data.mean())
        dfa = np.zeros(len(scales))

        for i, scale in enumerate(scales):
            # Calculate local trends
            segments = len(profile) // scale
            fluctuations = np.zeros(segments)

            for j in range(segments):
                segment = profile[j*scale:(j+1)*scale]
                time = np.arange(len(segment))
                coef = np.polyfit(time, segment, 1)
                trend = np.polyval(coef, time)
                fluctuations[j] = np.sqrt(np.mean((segment - trend)**2))

            dfa[i] = np.mean(fluctuations)

        return dfa.mean()

    def analyze_frequency_components(self, window_data):
        """Analyze frequency components before drops"""
        # Perform FFT
        signal = window_data['stability'].values
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal))

        # Get dominant frequencies
        main_freqs = np.abs(fft_vals[:len(freqs)//2])
        freq_vals = freqs[:len(freqs)//2]

        return freq_vals, main_freqs

    def calculate_complexity_metrics(self, window_data):
        """Calculate complexity metrics for the pre-drop window"""
        # Sample Entropy
        def sample_entropy(data, m=2, r=0.2):
            N = len(data)
            r = r * np.std(data)

            def count_matches(template):
                count = 0
                for i in range(N - len(template) + 1):
                    if np.all(np.abs(data[i:i+len(template)] - template) < r):
                        count += 1
                return count - 1  # Subtract self-match

            # Count matches for m and m+1 length templates
            matches_m = sum(count_matches(data[i:i+m]) for i in range(N-m+1))
            matches_m1 = sum(count_matches(data[i:i+m+1]) for i in range(N-m))

            return -np.log(matches_m1 / matches_m) if matches_m > 0 else np.inf

        metrics = {
            'sample_entropy': sample_entropy(window_data['stability']),
            'correlation_dimension': self._correlation_dimension(window_data),
            'lyapunov_exp': self._largest_lyapunov(window_data['stability'])
        }

        return metrics

    def _correlation_dimension(self, data, max_dim=10):
        """Calculate correlation dimension"""
        # Simplified implementation
        points = data[['stability', 'coherence', 'entanglement']].values
        dists = np.sqrt(np.sum((points[:, None] - points) ** 2, axis=2))
        r = np.logspace(-2, 2, 20)
        C = np.array([np.sum(dists < ri) for ri in r]) / len(points)**2
        valid = (C > 0) & (C < 1)
        if np.sum(valid) > 1:
            slope = np.polyfit(np.log(r[valid]), np.log(C[valid]), 1)[0]
            return slope
        return np.nan

    def _largest_lyapunov(self, data, tau=1, m=2):
        """Calculate largest Lyapunov exponent"""
        # Simplified implementation
        N = len(data)
        d0 = np.mean([np.abs(data[i] - data[i-tau]) for i in range(tau, N)])
        d1 = np.mean([np.abs(data[i] - data[i-2*tau]) for i in range(2*tau, N)])
        if d0 > 0:
            return np.log(d1/d0) / tau
        return np.nan

    def analyze_quantum_correlations(self, window_data):
        """Analyze quantum property correlations before drops"""
        quantum_props = ['coherence', 'entanglement', 'field_strength']
        correlations = window_data[['stability'] + quantum_props].corr()
        return correlations

    def plot_pre_drop_analysis(self, window_data, drop_num, early_warnings, freqs, main_freqs, complexity, correlations):
        """Create comprehensive visualization of pre-drop analysis"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Time Series and Early Warnings
        ax1 = fig.add_subplot(231)
        for col in early_warnings.columns:
            normalized = StandardScaler().fit_transform(early_warnings[col].values.reshape(-1, 1))
            ax1.plot(early_warnings.index, normalized, label=col)
        ax1.set_title('Early Warning Signals')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.set_xlabel('Time Steps Before Drop')

        # 2. Frequency Analysis
        ax2 = fig.add_subplot(232)
        ax2.plot(freqs, main_freqs)
        ax2.set_title('Frequency Components')
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Amplitude')

        # 3. Phase Space
        ax3 = fig.add_subplot(233)
        scatter = ax3.scatter(window_data['stability'],
                            window_data['coherence'],
                            c=window_data['time_to_drop'],
                            cmap='viridis')
        plt.colorbar(scatter, label='Time to Drop')
        ax3.set_title('Pre-Drop Phase Space')
        ax3.set_xlabel('Stability')
        ax3.set_ylabel('Coherence')

        # 4. Complexity Metrics
        ax4 = fig.add_subplot(234)
        complexity_vals = list(complexity.values())
        complexity_names = list(complexity.keys())
        ax4.bar(complexity_names, complexity_vals)
        plt.xticks(rotation=45)
        ax4.set_title('Complexity Metrics')

        # 5. Quantum Correlations
        ax5 = fig.add_subplot(235)
        sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax5)
        ax5.set_title('Property Correlations')

        # 6. Combined Indicators
        ax6 = fig.add_subplot(236)
        # Plot stability with quantum properties
        ax6.plot(window_data['time_to_drop'], window_data['stability'], 'b-', label='Stability')
        ax6.plot(window_data['time_to_drop'], window_data['coherence'], 'r-', label='Coherence')
        ax6.plot(window_data['time_to_drop'], window_data['entanglement'], 'g-', label='Entanglement')
        ax6.set_title('System Properties')
        ax6.set_xlabel('Time Steps Before Drop')
        ax6.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pre_drop_{drop_num}_analysis.png')
        plt.close()

    def create_comparative_analysis(self, all_windows):
        """Create comparative analysis of all pre-drop periods"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Average Trajectories
        ax1 = fig.add_subplot(221)
        for i, window in enumerate(all_windows):
            ax1.plot(window['time_to_drop'], window['stability'],
                    alpha=0.3, label=f'Drop {i+1}')
        # Plot average
        avg_stability = pd.concat([w['stability'] for w in all_windows], axis=1).mean(axis=1)
        ax1.plot(window['time_to_drop'], avg_stability, 'r-',
                label='Average', linewidth=2)
        ax1.set_title('Stability Trajectories Before Drops')
        ax1.legend()

        # 2. Variance Comparison
        ax2 = fig.add_subplot(222)
        variances = [w['stability'].var() for w in all_windows]
        ax2.bar(range(1, len(variances) + 1), variances)
        ax2.set_title('Stability Variance Before Each Drop')
        ax2.set_xlabel('Drop Number')
        ax2.set_ylabel('Variance')

        # 3. Cross-correlation Analysis
        ax3 = fig.add_subplot(223)
        correlations = []
        for window in all_windows:
            corr = window[['stability', 'coherence', 'entanglement']].corr()['stability']
            correlations.append(corr)
        correlations_df = pd.DataFrame(correlations)
        sns.heatmap(correlations_df, annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title('Property Correlations Across Drops')

        # 4. Early Warning Comparison
        ax4 = fig.add_subplot(224)
        warning_metrics = []
        for window in all_windows:
            metrics = self.analyze_early_warnings(window)
            warning_metrics.append(metrics.mean())
        warning_df = pd.DataFrame(warning_metrics)
        warning_df.plot(kind='bar', ax=ax4)
        ax4.set_title('Early Warning Metrics Comparison')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_analysis.png')
        plt.close()

    def run_analysis(self, df, drops):
        """Run complete pre-drop analysis"""
        print("\nAnalyzing Pre-Drop Conditions")
        print("===========================")

        # Extract pre-drop windows
        pre_drop_windows = self.extract_pre_drop_windows(df, drops)
        print(f"\nAnalyzing {len(drops)} pre-drop periods")

        # Analyze each pre-drop window
        for i, window in enumerate(pre_drop_windows, 1):
            print(f"\nAnalyzing pre-drop period {i}:")

            # Calculate early warning signals
            early_warnings = self.analyze_early_warnings(window)
            print("Early warning signals calculated")

            # Analyze frequency components
            freqs, main_freqs = self.analyze_frequency_components(window)
            print("Frequency analysis completed")

            # Calculate complexity metrics
            complexity = self.calculate_complexity_metrics(window)
            print("Complexity metrics calculated:")
            for metric, value in complexity.items():
                print(f"{metric}: {value:.3f}")

            # Analyze quantum correlations
            correlations = self.analyze_quantum_correlations(window)
            print("\nQuantum correlations analyzed")

            # Create visualization
            self.plot_pre_drop_analysis(window, i, early_warnings, freqs, main_freqs,
                                      complexity, correlations)
            print(f"Visualization saved as pre_drop_{i}_analysis.png")

        # Create comparative analysis
        print("\nGenerating comparative analysis...")
        self.create_comparative_analysis(pre_drop_windows)
        print("Comparative analysis saved as comparative_analysis.png")

        return pre_drop_windows

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

    # Analyze pre-drop conditions
    pre_analyzer = PreDropAnalyzer()
    pre_analyzer.run_analysis(pd.DataFrame(test.results), drops)