import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pywt
import os

class WarningPatternAnalyzer:
    def __init__(self):
        self.output_dir = 'output/warning_patterns'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')
        self.warning_window = 20  # Analysis window size

    def analyze_critical_slowing(self, data):
        """Analyze critical slowing down indicators"""
        # Calculate rolling statistics
        window = 5
        rolling = pd.Series(data).rolling(window=window)

        metrics = {
            'variance': rolling.var().values,
            'autocorr': rolling.apply(lambda x: pd.Series(x).autocorr()).values,
            'skewness': rolling.skew().values,
            'kurtosis': rolling.kurt().values
        }

        return pd.DataFrame(metrics)

    def analyze_flickering(self, data):
        """Analyze flickering between states"""
        # Calculate state transitions
        mean_state = np.mean(data)
        state_changes = np.diff(data > mean_state)

        # Analyze transition patterns
        transition_points = np.where(state_changes)[0]
        intervals = np.diff(transition_points)

        return {
            'transition_points': transition_points,
            'intervals': intervals,
            'frequency': len(transition_points) / len(data)
        }

    def analyze_wavelets(self, data):
        """Perform wavelet analysis to detect pattern changes"""
        # Perform continuous wavelet transform
        wavelet = 'cmor1.5-1.0'
        scales = np.arange(1, min(len(data)//2, 20))
        coefficients, frequencies = pywt.cwt(data, scales, wavelet)

        # Calculate wavelet power
        power = np.abs(coefficients) ** 2

        return {
            'coefficients': coefficients,
            'frequencies': frequencies,
            'power': power
        }

    def analyze_frequency_changes(self, data):
        """Analyze changes in frequency components"""
        # Perform short-time Fourier transform
        window = min(10, len(data)//4)
        frequencies, times, spectrogram = signal.spectrogram(
            data, nperseg=window, noverlap=window//2
        )

        return {
            'frequencies': frequencies,
            'times': times,
            'spectrogram': spectrogram
        }

    def detect_correlation_patterns(self, data_dict):
        """Analyze correlation patterns between different metrics"""
        # Create correlation matrix
        df = pd.DataFrame(data_dict)
        correlations = df.corr()

        # Calculate time-lagged correlations
        max_lag = min(5, len(df)//4)
        lagged_corr = {}

        for col1 in df.columns:
            for col2 in df.columns:
                if col1 != col2:
                    ccf = np.correlate(
                        df[col1] - df[col1].mean(),
                        df[col2] - df[col2].mean(),
                        mode='full'
                    )
                    lags = np.arange(-max_lag, max_lag + 1)
                    ccf = ccf[len(ccf)//2 - max_lag:len(ccf)//2 + max_lag + 1]
                    lagged_corr[f"{col1}-{col2}"] = (lags, ccf)

        return correlations, lagged_corr

    def plot_warning_analysis(self, window_data, drop_num):
        """Create comprehensive visualization of warning patterns"""
        # Extract stability and quantum properties
        stability = window_data['stability'].values
        coherence = window_data['coherence'].values
        entanglement = window_data['entanglement'].values
        time_to_drop = window_data['time_to_drop'].values

        # Perform analyses
        critical_metrics = self.analyze_critical_slowing(stability)
        flickering = self.analyze_flickering(stability)
        wavelets = self.analyze_wavelets(stability)
        freq_changes = self.analyze_frequency_changes(stability)

        # Create visualization
        fig = plt.figure(figsize=(20, 15))

        # 1. Time Series with Critical Metrics
        ax1 = fig.add_subplot(331)
        ax1.plot(time_to_drop, stability, 'b-', label='Stability')
        ax1.plot(time_to_drop, coherence, 'r-', label='Coherence')
        ax1.plot(time_to_drop, entanglement, 'g-', label='Entanglement')
        ax1.set_title('System Properties')
        ax1.legend()

        # 2. Critical Slowing Indicators
        ax2 = fig.add_subplot(332)
        for col in critical_metrics.columns:
            normalized = StandardScaler().fit_transform(
                critical_metrics[col].values.reshape(-1, 1)
            )
            ax2.plot(time_to_drop[5:], normalized.flatten(), label=col)
        ax2.set_title('Critical Slowing Indicators')
        ax2.legend()

        # 3. State Flickering
        ax3 = fig.add_subplot(333)
        ax3.plot(time_to_drop[1:], np.diff(stability), 'b-')
        ax3.scatter(time_to_drop[flickering['transition_points']],
                   np.zeros_like(flickering['transition_points']),
                   c='r', label='Transitions')
        ax3.set_title(f'State Transitions (Frequency: {flickering["frequency"]:.3f})')
        ax3.legend()

        # 4. Wavelet Analysis
        ax4 = fig.add_subplot(334)
        im = ax4.imshow(wavelets['power'], aspect='auto', cmap='viridis',
                       extent=[time_to_drop[0], time_to_drop[-1],
                              1, len(wavelets['frequencies'])])
        plt.colorbar(im, ax=ax4)
        ax4.set_title('Wavelet Power Spectrum')
        ax4.set_ylabel('Scale')

        # 5. Frequency Changes
        ax5 = fig.add_subplot(335)
        im = ax5.pcolormesh(freq_changes['times'], freq_changes['frequencies'],
                           freq_changes['spectrogram'], shading='gouraud')
        plt.colorbar(im, ax=ax5)
        ax5.set_title('Spectrogram')
        ax5.set_ylabel('Frequency')

        # 6. Phase Space
        ax6 = fig.add_subplot(336)
        ax6.scatter(stability[:-1], stability[1:], c=time_to_drop[1:],
                   cmap='viridis')
        ax6.set_xlabel('Stability(t)')
        ax6.set_ylabel('Stability(t+1)')
        ax6.set_title('Phase Space Reconstruction')

        # 7. Correlation Analysis
        correlations, lagged_corr = self.detect_correlation_patterns({
            'stability': stability,
            'coherence': coherence,
            'entanglement': entanglement
        })

        ax7 = fig.add_subplot(337)
        sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax7)
        ax7.set_title('Property Correlations')

        # 8. Lagged Correlations
        ax8 = fig.add_subplot(338)
        for key, (lags, ccf) in lagged_corr.items():
            ax8.plot(lags, ccf, label=key)
        ax8.set_title('Lagged Correlations')
        ax8.legend()

        # 9. Warning Score
        ax9 = fig.add_subplot(339)
        # Calculate combined warning score
        warning_score = (
            StandardScaler().fit_transform(critical_metrics['variance'].reshape(-1, 1)) +
            StandardScaler().fit_transform(critical_metrics['autocorr'].reshape(-1, 1)) +
            StandardScaler().fit_transform(critical_metrics['skewness'].reshape(-1, 1))
        ).flatten()

        ax9.plot(time_to_drop[5:], warning_score, 'r-')
        ax9.axhline(y=2.0, color='y', linestyle='--', label='Warning Threshold')
        ax9.set_title('Combined Warning Score')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/warning_analysis_drop_{drop_num}.png')
        plt.close()

        return {
            'critical_metrics': critical_metrics,
            'flickering': flickering,
            'warning_score': warning_score
        }

    def create_comparative_warning_analysis(self, all_results, drops):
        """Create comparative analysis of warning patterns across drops"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Warning Score Comparison
        ax1 = fig.add_subplot(221)
        for i, results in enumerate(all_results):
            ax1.plot(results['warning_score'], label=f'Drop {i+1}')
        ax1.axhline(y=2.0, color='y', linestyle='--', label='Warning Threshold')
        ax1.set_title('Warning Score Evolution')
        ax1.legend()

        # 2. Critical Metrics Comparison
        ax2 = fig.add_subplot(222)
        metrics_comparison = []
        for results in all_results:
            metrics = results['critical_metrics'].mean()
            metrics_comparison.append(metrics)
        metrics_df = pd.DataFrame(metrics_comparison)

        sns.heatmap(metrics_df, annot=True, cmap='viridis', ax=ax2)
        ax2.set_title('Critical Metrics Comparison')
        ax2.set_yticklabels([f'Drop {i+1}' for i in range(len(all_results))])

        # 3. Flickering Analysis
        ax3 = fig.add_subplot(223)
        frequencies = [r['flickering']['frequency'] for r in all_results]
        ax3.bar(range(1, len(frequencies) + 1), frequencies)
        ax3.set_title('State Transition Frequencies')
        ax3.set_xlabel('Drop Number')
        ax3.set_ylabel('Transition Frequency')

        # 4. Warning Pattern Timeline
        ax4 = fig.add_subplot(224)
        timeline = np.zeros(max([d['index'] for d in drops]) + 1)
        for i, results in enumerate(all_results):
            warning_region = np.where(results['warning_score'] > 2.0)[0]
            if len(warning_region) > 0:
                start_idx = drops[i]['index'] - len(results['warning_score']) + warning_region[0]
                end_idx = drops[i]['index']
                timeline[start_idx:end_idx] = 1

        ax4.plot(timeline, 'r-')
        ax4.set_title('Warning Regions Timeline')
        ax4.set_xlabel('Time Step')
        ax4.set_ylabel('Warning Active')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_warning_analysis.png')
        plt.close()

    def run_analysis(self, df, drops):
        """Run complete warning pattern analysis"""
        print("\nAnalyzing Warning Patterns")
        print("========================")

        all_results = []

        # Analyze each drop
        for i, drop in enumerate(drops, 1):
            print(f"\nAnalyzing warning patterns for drop {i}:")

            # Extract pre-drop window
            start_idx = max(0, drop['index'] - self.warning_window)
            window_data = df.iloc[start_idx:drop['index']].copy()
            window_data['time_to_drop'] = range(-len(window_data), 0)

            # Analyze warning patterns
            results = self.plot_warning_analysis(window_data, i)
            all_results.append(results)

            # Print summary
            print("Warning pattern analysis completed:")
            print(f"- Critical slowing indicators detected: {np.any(results['warning_score'] > 2.0)}")
            print(f"- State transitions frequency: {results['flickering']['frequency']:.3f}")
            print(f"- Maximum warning score: {np.max(results['warning_score']):.3f}")

        # Create comparative analysis
        print("\nGenerating comparative analysis...")
        self.create_comparative_warning_analysis(all_results, drops)
        print("Comparative analysis saved as comparative_warning_analysis.png")

        return all_results

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
    warning_analyzer = WarningPatternAnalyzer()
    warning_analyzer.run_analysis(pd.DataFrame(test.results), drops)