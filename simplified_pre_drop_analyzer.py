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
        # Calculate rolling statistics
        stability = window_data['stability'].values
        n = len(stability)
        window = 5

        warnings = {
            'variance': [],
            'autocorr': [],
            'trend': []
        }

        for i in range(window, n):
            segment = stability[i-window:i]
            warnings['variance'].append(np.var(segment))
            warnings['autocorr'].append(np.corrcoef(segment[:-1], segment[1:])[0,1])
            warnings['trend'].append(np.polyfit(range(window), segment, 1)[0])

        return pd.DataFrame(warnings)

    def analyze_quantum_state(self, window_data):
        """Analyze quantum properties before drop"""
        properties = {
            'coherence_mean': window_data['coherence'].mean(),
            'coherence_std': window_data['coherence'].std(),
            'entanglement_mean': window_data['entanglement'].mean(),
            'entanglement_std': window_data['entanglement'].std(),
            'field_strength_mean': window_data['field_strength'].mean(),
            'field_strength_std': window_data['field_strength'].std()
        }

        # Calculate correlations
        correlations = window_data[['stability', 'coherence', 'entanglement', 'field_strength']].corr()

        return properties, correlations

    def analyze_stability_trends(self, window_data):
        """Analyze stability trends before drop"""
        stability = window_data['stability'].values
        time = np.arange(len(stability))

        # Fit linear trend
        slope, intercept = np.polyfit(time, stability, 1)

        # Calculate fluctuations around trend
        trend = slope * time + intercept
        fluctuations = stability - trend

        return {
            'slope': slope,
            'fluctuation_std': np.std(fluctuations),
            'acceleration': np.gradient(np.gradient(stability))
        }

    def plot_pre_drop_analysis(self, window_data, warnings, quantum_props, correlations, trends, drop_num):
        """Create comprehensive visualization of pre-drop analysis"""
        fig = plt.figure(figsize=(20, 12))

        # 1. Main Time Series
        ax1 = fig.add_subplot(231)
        time = window_data['time_to_drop']
        ax1.plot(time, window_data['stability'], 'b-', label='Stability')
        ax1.plot(time, window_data['coherence'], 'r-', label='Coherence')
        ax1.plot(time, window_data['entanglement'], 'g-', label='Entanglement')
        ax1.set_title('System Properties Before Drop')
        ax1.set_xlabel('Time Steps to Drop')
        ax1.legend()

        # 2. Early Warning Signals
        ax2 = fig.add_subplot(232)
        for col in warnings.columns:
            normalized = StandardScaler().fit_transform(warnings[col].values.reshape(-1, 1))
            ax2.plot(time[5:], normalized, label=col)
        ax2.set_title('Early Warning Signals')
        ax2.legend()
        ax2.set_xlabel('Time Steps to Drop')

        # 3. Phase Space
        ax3 = fig.add_subplot(233)
        scatter = ax3.scatter(window_data['stability'],
                            window_data['coherence'],
                            c=time,
                            cmap='viridis')
        plt.colorbar(scatter, label='Time to Drop')
        ax3.set_title('Phase Space Evolution')
        ax3.set_xlabel('Stability')
        ax3.set_ylabel('Coherence')

        # 4. Quantum Properties
        ax4 = fig.add_subplot(234)
        quantum_means = [quantum_props[k] for k in quantum_props.keys() if 'mean' in k]
        quantum_stds = [quantum_props[k] for k in quantum_props.keys() if 'std' in k]
        labels = ['Coherence', 'Entanglement', 'Field Strength']
        x = np.arange(len(labels))
        width = 0.35
        ax4.bar(x - width/2, quantum_means, width, label='Mean')
        ax4.bar(x + width/2, quantum_stds, width, label='Std')
        ax4.set_xticks(x)
        ax4.set_xticklabels(labels)
        ax4.set_title('Quantum Properties Statistics')
        ax4.legend()

        # 5. Correlation Matrix
        ax5 = fig.add_subplot(235)
        sns.heatmap(correlations, annot=True, cmap='coolwarm', ax=ax5)
        ax5.set_title('Property Correlations')

        # 6. Stability Analysis
        ax6 = fig.add_subplot(236)
        time = np.arange(len(window_data))
        stability = window_data['stability'].values
        trend = trends['slope'] * time + np.mean(stability)
        ax6.plot(time, stability, 'b-', label='Stability')
        ax6.plot(time, trend, 'r--', label=f'Trend (slope={trends["slope"]:.3f})')
        ax6.set_title('Stability Trend Analysis')
        ax6.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/pre_drop_{drop_num}_analysis.png')
        plt.close()

    def create_comparative_summary(self, all_windows, all_warnings, all_quantum_props, all_trends):
        """Create comparative analysis of all pre-drop periods"""
        fig = plt.figure(figsize=(15, 10))

        # 1. Average Trajectories
        ax1 = fig.add_subplot(221)
        for i, window in enumerate(all_windows):
            ax1.plot(window['time_to_drop'], window['stability'],
                    alpha=0.3, label=f'Drop {i+1}')
        avg_stability = pd.concat([w.set_index('time_to_drop')['stability']
                                 for w in all_windows], axis=1).mean(axis=1)
        ax1.plot(avg_stability.index, avg_stability.values, 'r-',
                label='Average', linewidth=2)
        ax1.set_title('Stability Trajectories')
        ax1.legend()

        # 2. Warning Signal Comparison
        ax2 = fig.add_subplot(222)
        warning_stats = []
        for warnings in all_warnings:
            warning_stats.append([w.mean() for w in warnings.values()])
        warning_stats = np.array(warning_stats)
        sns.heatmap(warning_stats, xticklabels=all_warnings[0].columns,
                   yticklabels=[f'Drop {i+1}' for i in range(len(all_warnings))],
                   ax=ax2, cmap='viridis')
        ax2.set_title('Early Warning Signal Strength')

        # 3. Quantum Property Comparison
        ax3 = fig.add_subplot(223)
        quantum_data = []
        for props in all_quantum_props:
            quantum_data.append([props[k] for k in props.keys() if 'mean' in k])
        quantum_data = np.array(quantum_data)
        sns.heatmap(quantum_data,
                   xticklabels=['Coherence', 'Entanglement', 'Field'],
                   yticklabels=[f'Drop {i+1}' for i in range(len(all_quantum_props))],
                   ax=ax3, cmap='viridis')
        ax3.set_title('Quantum Properties Comparison')

        # 4. Trend Analysis
        ax4 = fig.add_subplot(224)
        slopes = [t['slope'] for t in all_trends]
        fluct = [t['fluctuation_std'] for t in all_trends]
        x = np.arange(len(slopes))
        ax4.bar(x - 0.2, slopes, 0.4, label='Trend Slope')
        ax4.bar(x + 0.2, fluct, 0.4, label='Fluctuation Std')
        ax4.set_xticks(x)
        ax4.set_xticklabels([f'Drop {i+1}' for i in range(len(slopes))])
        ax4.set_title('Stability Trends')
        ax4.legend()

        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/comparative_summary.png')
        plt.close()

    def run_analysis(self, df, drops):
        """Run complete pre-drop analysis"""
        print("\nAnalyzing Pre-Drop Conditions")
        print("===========================")

        # Extract pre-drop windows
        pre_drop_windows = self.extract_pre_drop_windows(df, drops)
        print(f"\nAnalyzing {len(drops)} pre-drop periods")

        all_warnings = []
        all_quantum_props = []
        all_trends = []

        # Analyze each pre-drop window
        for i, window in enumerate(pre_drop_windows, 1):
            print(f"\nAnalyzing pre-drop period {i}:")

            # Calculate early warning signals
            warnings = self.analyze_early_warnings(window)
            all_warnings.append(warnings)
            print("Early warning signals calculated")

            # Analyze quantum properties
            quantum_props, correlations = self.analyze_quantum_state(window)
            all_quantum_props.append(quantum_props)
            print("Quantum properties analyzed")

            # Analyze stability trends
            trends = self.analyze_stability_trends(window)
            all_trends.append(trends)
            print("Stability trends analyzed")

            # Create visualization
            self.plot_pre_drop_analysis(window, warnings, quantum_props,
                                      correlations, trends, i)
            print(f"Visualization saved as pre_drop_{i}_analysis.png")

        # Create comparative summary
        print("\nGenerating comparative analysis...")
        self.create_comparative_summary(pre_drop_windows, all_warnings,
                                      all_quantum_props, all_trends)
        print("Comparative analysis saved as comparative_summary.png")

        return pre_drop_windows, all_warnings, all_quantum_props, all_trends

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