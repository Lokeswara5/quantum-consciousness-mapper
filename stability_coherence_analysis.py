import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import os

class StabilityCoherenceAnalyzer:
    def __init__(self):
        self.output_dir = 'output'
        os.makedirs(self.output_dir, exist_ok=True)
        plt.style.use('dark_background')

    def load_results(self, results_file):
        """Load results from previous run"""
        return pd.DataFrame(results_file)

    def analyze_correlation(self, df):
        """Analyze correlation between stability and coherence"""
        # Calculate correlation
        correlation = stats.pearsonr(df['stability'], df['coherence'])

        # Create correlation plot
        plt.figure(figsize=(12, 8))

        # Main scatter plot
        plt.scatter(df['stability'], df['coherence'],
                   c=df['entanglement'], cmap='viridis', alpha=0.6)
        plt.colorbar(label='Entanglement')

        # Add trend line
        z = np.polyfit(df['stability'], df['coherence'], 1)
        p = np.poly1d(z)
        plt.plot(df['stability'], p(df['stability']), "r--", alpha=0.8)

        plt.xlabel('Stability')
        plt.ylabel('Coherence')
        plt.title(f'Stability vs Coherence (r={correlation[0]:.3f}, p={correlation[1]:.3f})')

        plt.savefig(f'{self.output_dir}/stability_coherence_correlation.png')
        plt.close()

        return correlation

    def analyze_stability_drops(self, df):
        """Analyze stability drops and their relationship with coherence"""
        # Identify stability drops
        stability_diff = df['stability'].diff()
        drop_threshold = -0.1  # Consider drops larger than 0.1 as significant
        drops = stability_diff < drop_threshold

        # Create timeline plot
        plt.figure(figsize=(15, 10))

        # Plot stability and coherence
        plt.subplot(211)
        plt.plot(df['stability'], label='Stability', color='blue', alpha=0.7)
        plt.plot(df['coherence'], label='Coherence', color='red', alpha=0.7)

        # Highlight stability drops
        drop_points = df[drops].index
        plt.scatter(drop_points, df.loc[drop_points, 'stability'],
                   color='yellow', s=100, label='Stability Drops')

        plt.legend()
        plt.title('Stability and Coherence Timeline with Drops')

        # Plot drop analysis
        plt.subplot(212)
        drop_coherence = df.loc[drop_points, 'coherence']
        non_drop_coherence = df.loc[~drops, 'coherence']

        # Create violin plot comparing coherence during drops vs normal
        data = [drop_coherence, non_drop_coherence]
        labels = ['During Drops', 'Normal States']

        violin = plt.violinplot(data, showmeans=True)
        plt.xticks([1, 2], labels)
        plt.ylabel('Coherence')
        plt.title('Coherence Distribution During Stability Drops vs Normal States')

        # Customize violin plot colors
        for pc in violin['bodies']:
            pc.set_facecolor('lightblue')
            pc.set_alpha(0.7)

        plt.savefig(f'{self.output_dir}/stability_drops_analysis.png')
        plt.close()

        # Calculate statistics
        stats_dict = {
            'num_drops': len(drop_points),
            'avg_drop_coherence': drop_coherence.mean(),
            'avg_normal_coherence': non_drop_coherence.mean(),
            'coherence_diff': drop_coherence.mean() - non_drop_coherence.mean(),
            'ttest': stats.ttest_ind(drop_coherence, non_drop_coherence)
        }

        return stats_dict

    def analyze_phase_space(self, df):
        """Analyze phase space relationship between stability and coherence"""
        plt.figure(figsize=(12, 8))

        # Create phase space plot
        scatter = plt.scatter(df['stability'], df['coherence'],
                            c=df.index, cmap='viridis',
                            s=50, alpha=0.6)
        plt.colorbar(scatter, label='Time')

        # Add arrows to show evolution
        step = 5  # Plot arrow every 5 points
        for i in range(0, len(df)-step, step):
            plt.arrow(df['stability'].iloc[i], df['coherence'].iloc[i],
                     df['stability'].iloc[i+step] - df['stability'].iloc[i],
                     df['coherence'].iloc[i+step] - df['coherence'].iloc[i],
                     head_width=0.02, head_length=0.02, fc='w', ec='w', alpha=0.3)

        plt.xlabel('Stability')
        plt.ylabel('Coherence')
        plt.title('Stability-Coherence Phase Space Evolution')

        plt.savefig(f'{self.output_dir}/stability_coherence_phase_space.png')
        plt.close()

    def analyze_cross_correlation(self, df):
        """Analyze cross-correlation between stability and coherence"""
        plt.figure(figsize=(12, 6))

        # Calculate cross-correlation
        cross_corr = np.correlate(
            (df['stability'] - df['stability'].mean()) / df['stability'].std(),
            (df['coherence'] - df['coherence'].mean()) / df['coherence'].std(),
            mode='full'
        )

        # Calculate lag values
        lags = np.arange(-(len(df)-1), len(df))

        # Plot cross-correlation
        plt.plot(lags, cross_corr)
        plt.axvline(x=0, color='r', linestyle='--', alpha=0.5)
        plt.xlabel('Lag')
        plt.ylabel('Cross-correlation')
        plt.title('Stability-Coherence Cross-correlation')

        # Find peak correlation and lag
        peak_idx = np.argmax(np.abs(cross_corr))
        peak_lag = lags[peak_idx]
        peak_corr = cross_corr[peak_idx]

        plt.scatter(peak_lag, peak_corr, color='red', s=100,
                   label=f'Peak at lag={peak_lag}')
        plt.legend()

        plt.savefig(f'{self.output_dir}/stability_coherence_cross_correlation.png')
        plt.close()

        return {'peak_lag': peak_lag, 'peak_correlation': peak_corr}

    def run_analysis(self, results):
        """Run complete stability-coherence analysis"""
        print("\nAnalyzing Stability-Coherence Relationship")
        print("=========================================")

        df = pd.DataFrame(results)

        # 1. Correlation Analysis
        correlation = self.analyze_correlation(df)
        print(f"\nCorrelation Analysis:")
        print(f"Pearson correlation: {correlation[0]:.3f}")
        print(f"P-value: {correlation[1]:.3f}")

        # 2. Stability Drops Analysis
        drop_stats = self.analyze_stability_drops(df)
        print(f"\nStability Drops Analysis:")
        print(f"Number of significant drops: {drop_stats['num_drops']}")
        print(f"Average coherence during drops: {drop_stats['avg_drop_coherence']:.3f}")
        print(f"Average coherence during normal states: {drop_stats['avg_normal_coherence']:.3f}")
        print(f"Coherence difference: {drop_stats['coherence_diff']:.3f}")
        print(f"T-test p-value: {drop_stats['ttest'].pvalue:.3f}")

        # 3. Phase Space Analysis
        self.analyze_phase_space(df)
        print("\nPhase space analysis visualization saved")

        # 4. Cross-correlation Analysis
        cross_corr = self.analyze_cross_correlation(df)
        print(f"\nCross-correlation Analysis:")
        print(f"Peak lag: {cross_corr['peak_lag']}")
        print(f"Peak correlation: {cross_corr['peak_correlation']:.3f}")

        print("\nAnalysis complete! Visualizations saved to output directory.")

if __name__ == "__main__":
    # Load and analyze results
    analyzer = StabilityCoherenceAnalyzer()

    # Example usage (you'll need to provide actual results)
    # analyzer.run_analysis(results)