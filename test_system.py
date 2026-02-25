import numpy as np
import time
import psutil
import os
from datetime import datetime
from src.core.advanced_analysis import AdvancedConsciousnessAnalyzer
from src.visualization.consciousness_visualizer import ConsciousnessVisualizer
from src.monitoring.realtime_monitor import RealtimeMonitor
from src.prediction.advanced_predictor import ConsciousnessPredictor

class SystemTester:
    def __init__(self):
        self.analyzer = AdvancedConsciousnessAnalyzer(dimensions=64)  # Reduced dimensions for laptop
        self.visualizer = ConsciousnessVisualizer()
        self.monitor = RealtimeMonitor(update_interval=0.5)  # Slower updates for performance
        self.predictor = ConsciousnessPredictor(state_dim=64)

    def monitor_system_resources(self):
        """Monitor system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        return cpu_percent, memory

    def run_test(self, duration_seconds=30):
        print("\n=== Starting System Test ===")
        print(f"Test duration: {duration_seconds} seconds")
        print("\nInitializing components...")

        start_time = time.time()
        last_print = start_time
        test_iterations = 0

        try:
            while time.time() - start_time < duration_seconds:
                test_iterations += 1
                current_time = time.time()

                # Generate test state
                test_state = np.random.rand(64)

                # Run analysis
                analysis_start = time.time()
                analysis_results = self.analyzer.analyze_consciousness_state(test_state)
                analysis_time = time.time() - analysis_start

                # Run prediction
                prediction_start = time.time()
                prediction_results = self.predictor.predict_consciousness_evolution(
                    test_state, timesteps=5
                )
                prediction_time = time.time() - prediction_start

                # Get system metrics
                cpu_percent, memory_mb = self.monitor_system_resources()

                # Print status every 5 seconds
                if current_time - last_print >= 5:
                    print(f"\n--- Status at {datetime.now().strftime('%H:%M:%S')} ---")
                    print(f"Iterations completed: {test_iterations}")
                    print(f"Analysis time: {analysis_time:.3f} seconds")
                    print(f"Prediction time: {prediction_time:.3f} seconds")
                    print(f"CPU Usage: {cpu_percent}%")
                    print(f"Memory Usage: {memory_mb:.1f} MB")

                    # Print some analysis results
                    if 'quantum' in analysis_results:
                        print("\nQuantum Metrics:")
                        for key, value in analysis_results['quantum'].items():
                            if isinstance(value, (int, float)):
                                print(f"  {key}: {value:.3f}")

                    last_print = current_time

                # Small delay to prevent overwhelming the system
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nTest interrupted by user")
        except Exception as e:
            print(f"\nTest error: {e}")
        finally:
            end_time = time.time()
            total_time = end_time - start_time

            print("\n=== Test Summary ===")
            print(f"Total runtime: {total_time:.1f} seconds")
            print(f"Total iterations: {test_iterations}")
            print(f"Average time per iteration: {total_time/test_iterations:.3f} seconds")
            print(f"Final CPU Usage: {psutil.cpu_percent()}%")
            print(f"Final Memory Usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    print("Quantum Consciousness Mapper System Test")
    print("---------------------------------------")
    print("This test will run the system components and monitor performance.")
    print("Press Ctrl+C to stop the test at any time.")

    input("\nPress Enter to start the test...")

    tester = SystemTester()
    tester.run_test(duration_seconds=30)