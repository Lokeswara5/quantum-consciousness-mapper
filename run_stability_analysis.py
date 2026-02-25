from complex_evolution_test import ComplexEvolutionTest
from stability_coherence_analysis import StabilityCoherenceAnalyzer

def run_analysis():
    # Run evolution test
    print("Running evolution test...")
    test = ComplexEvolutionTest(dimensions=64)
    test.run_test_scenario(iterations=200)

    # Analyze stability-coherence relationship
    print("\nAnalyzing stability-coherence relationship...")
    analyzer = StabilityCoherenceAnalyzer()
    analyzer.run_analysis(test.results)

if __name__ == "__main__":
    run_analysis()