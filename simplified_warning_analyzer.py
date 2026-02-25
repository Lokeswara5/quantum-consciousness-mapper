import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal, stats
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

class WarningPatternAnalyzer:
    def __init__(self):
        self.output_dir = 'output/warning_patterns'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')
        self.warning_window = 20

    def analyze_critical_indicators(self, data):
        """Analyze critical transition indicators"""
        window = 5
        indicators = pd.DataFrame()

        # Calculate rolling statistics
        rolling_data = pd.Series(data).rolling(window=window)
        indicators['variance'] = rolling_data.var()
        indicators['autocorr'] = rolling_data.apply(
            lambda x: x.autocorr() if len(x.dropna()) > 1 else np.nan
        )
        indicators['skewness'] = rolling_data.skew()

        # Fill NaN values
        indicators = indicators.fillna(method='bfill')
        return indicators

    def analyze_frequency_domain(self, data):
        """Analyze frequency domain characteristics"""
        # Calculate power spectrum
        freqs, psd = signal.welch(data)

        # Find dominant frequencies
        peak_freqs = freqs[signal.find_peaks(psd)[0]]
        peak_powers = psd[signal.find_peaks(psd)[0]]

        return {
            'freqs': freqs,
            'psd': psd,
            'peak_freqs': peak_freqs,
            'peak_powers': peak_powers
        }

    def analyze_state_transitions(self, data):
        """Analyze state transition patterns"""
        # Calculate state changes
        mean_state = np.mean(data)
        states = data > mean_state
        transitions = np.diff(states.astype(int))

        return {
            'transitions': transitions,
            'frequency': np.sum(np.abs(transitions)) / len(data),
            'transition_points': np.where(transitions != 0)[0]
        }

    def calculate_warning_score(self, indicators):
        """Calculate combined warning score"""
        normalized = StandardScaler().fit_transform(indicators)
        warning_score = np.mean(normalized, axis=1)
        return warning_score

    def plot_warning_analysis(self, window_data, drop_num):
        """Create visualization of warning patterns"""
        fig = plt.figure(figsize=(20, 12))

        # Extract data
        stability = window_data['stability'].values
        coherence = window_data['coherence'].values
        entanglement = window_data['entanglement'].values
        time_to_drop = window_data['time_to_drop'].values

        # 1. System Properties
        ax1 = fig.add_subplot(231)
        ax1.plot(time_to_drop, stability, 'b-', label='Stability')
        ax1.plot(time_to_drop, coherence, 'r-', label='Coherence')
        ax1.plot(time_to_drop, entanglement, 'g-', label='Entanglement')
        ax1.set_title('System Properties')
        ax1.legend()
        ax1.set_xlabel('Time Steps to Drop')

        # 2. Critical Indicators
        indicators = self.analyze_critical_indicators(stability)
        ax2 = fig.add_subplot(232)
        for col in indicators.columns:
            normalized = StandardScaler().fit_transform(
                indicators[col].values.reshape(-1, 1)
            )
            ax2.plot(time_to_drop[window.rolling_window-1:],
                    normalized, label=col)
        ax2.set_title('Critical Indicators')
        ax2.legend()
        ax2.set_xlabel('Time Steps to Drop')

        # 3. State Space
        ax3 = fig.add_subplot(233)
        ax3.scatter(stability[:-1], stability[1:], c=range(len(stability)-1),
                   cmap='viridis')
        ax3.set_xlabel('Stability(t)')
        ax3.set_ylabel('Stability(t+1)')
        ax3.set_title('State Space')

        # 4. Frequency Analysis
        freq_analysis = self.analyze_frequency_domain(stability)
        ax4 = fig.add_subplot(234)
        ax4.plot(freq_analysis['freqs'], freq_analysis['psd'])
        ax4.scatter(freq_analysis['peak_freqs'],
                   freq_analysis['peak_powers'],
                   color='red', label='Peak Frequencies')
        ax4.set_title('Frequency Analysis')
        ax4.set_xlabel('Frequency')
        ax4.set_ylabel('Power')
        ax4.legend()

        # 5. State Transitions
        transitions = self.analyze_state_transitions(stability)
        ax5 = fig.add_subplot(235)
        ax5.plot(time_to_drop[1:], transitions['transitions'], 'b-')
        ax5.scatter(time_to_drop[1:][transitions['transition_points']],
                   transitions['transitions'][transitions['transition_points']],
                   color='red', label='Transitions')
        ax5.set_title(f'State Transitions (Freq: {transitions["frequency"]:.3f})')
        ax5.legend()
        ax5.set_xlabel('Time Steps to Drop')

        # 6. Warning Score
        warning_score = self.calculate_warning_score(indicators)
        ax6 = fig.add_subplot(236)
        ax6.plot(time_to_drop[window.rolling_window-1:], warning_score, 'r-')
        ax6.axhline(y=1.5, color='y', linestyle='--',
                   label='Warning Threshold')
        ax6.set_title('Warning Score')
        ax6.set_xlabel('Time Steps to Drop')
        ax6.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/warning_analysis_drop_{drop_num}.png')
        plt.close()

        return {
            'indicators': indicators,
            'transitions': transitions,
            'warning_score': warning_score
        }

    def create_summary_analysis(self, all_results):
        """Create summary of warning patterns across all drops"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Warning Score Comparison
        ax1 = fig.add_subplot(221)
        for i, results in enumerate(all_results):
            ax1.plot(results['warning_score'],
                    label=f'Drop {i+1}')
        ax1.axhline(y=1.5, color='y', linestyle='--',
                   label='Warning Threshold')
        ax1.set_title('Warning Score Comparison')
        ax1.legend()

        # 2. Indicator Statistics
        ax2 = fig.add_subplot(222)
        indicator_stats = []
        for results in all_results:
            stats = results['indicators'].mean()
            indicator_stats.append(stats)
        indicator_df = pd.DataFrame(indicator_stats)
        sns.heatmap(indicator_df, annot=True, cmap='viridis', ax=ax2)
        ax2.set_title('Critical Indicator Statistics')
        ax2.set_yticklabels([f'Drop {i+1}'
                            for i in range(len(all_results))])

        # 3. Transition Frequencies
        ax3 = fig.add_subplot(223)
        frequencies = [r['transitions']['frequency']
                      for r in all_results]
        ax3.bar(range(1, len(frequencies) + 1), frequencies)
        ax3.set_title('State Transition Frequencies')
        ax3.set_xlabel('Drop Number')
        ax3.set_ylabel('Transition Frequency')

        # 4. Warning Pattern Summary
        ax4 = fig.add_subplot(224)
        warning_levels = []
        for results in all_results:
            warning_levels.append(np.mean(
                results['warning_score'] > 1.5
            ))
        ax4.bar(range(1, len(warning_levels) + 1),
                warning_levels)
        ax4.set_title('Warning Level Summary')
        ax4.set_xlabel('Drop Number')
        ax4.set_ylabel('Warning Duration Ratio')

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/warning_summary.png')
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
            print(f"- Critical indicators detected: {np.any(results['warning_score'] > 1.5)}")
            print(f"- Transition frequency: {results['transitions']['frequency']:.3f}")
            print(f"- Maximum warning score: {np.max(results['warning_score']):.3f}")

        # Create summary analysis
        print("\nGenerating summary analysis...")
        self.create_summary_analysis(all_results)
        print("Summary analysis saved as warning_summary.png")

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