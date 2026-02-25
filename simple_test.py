import numpy as np
import time
import psutil
import os
from datetime import datetime

class SimpleSystemTester:
    def __init__(self, dimensions=64):
        self.dimensions = dimensions
        self.quantum_threshold = 0.7
        self.chaos_threshold = 0.6

    def analyze_quantum_properties(self, state_vector):
        """Simplified quantum analysis"""
        # Calculate basic quantum properties
        amplitude = np.abs(np.fft.fft(state_vector))
        coherence = np.mean(amplitude) / np.max(amplitude)
        entanglement = np.mean(np.outer(state_vector, state_vector))

        return {
            'coherence': coherence,
            'entanglement': entanglement,
            'amplitude': amplitude
        }

    def analyze_chaotic_properties(self, state_vector):
        """Simplified chaos analysis"""
        # Calculate basic chaotic properties
        lyapunov = np.mean(np.diff(state_vector))
        entropy = -np.sum(state_vector**2 * np.log(state_vector**2 + 1e-10))

        return {
            'lyapunov': lyapunov,
            'entropy': entropy,
            'chaos_degree': abs(lyapunov) * entropy
        }

    def predict_next_state(self, current_state):
        """Simple prediction"""
        # Apply simple evolution
        next_state = current_state + np.random.normal(0, 0.1, self.dimensions)
        next_state = next_state / np.linalg.norm(next_state)

        return next_state

    def monitor_system_resources(self):
        """Monitor system resource usage"""
        cpu_percent = psutil.cpu_percent()
        memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        return cpu_percent, memory

    def run_test(self, duration_seconds=30):
        print("\n=== Starting Simple System Test ===")
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
                test_state = np.random.rand(self.dimensions)
                test_state = test_state / np.linalg.norm(test_state)

                # Run analysis
                analysis_start = time.time()
                quantum_results = self.analyze_quantum_properties(test_state)
                chaos_results = self.analyze_chaotic_properties(test_state)
                analysis_time = time.time() - analysis_start

                # Run prediction
                prediction_start = time.time()
                next_state = self.predict_next_state(test_state)
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

                    print("\nQuantum Metrics:")
                    print(f"  Coherence: {quantum_results['coherence']:.3f}")
                    print(f"  Entanglement: {quantum_results['entanglement']:.3f}")

                    print("\nChaos Metrics:")
                    print(f"  Lyapunov: {chaos_results['lyapunov']:.3f}")
                    print(f"  Entropy: {chaos_results['entropy']:.3f}")

                    last_print = current_time

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
    print("Quantum Consciousness Mapper - Simple System Test")
    print("----------------------------------------------")
    print("This test will run basic system components and monitor performance.")
    print("The test will run for 30 seconds.")
    print("\nStarting test in 3 seconds...")
    time.sleep(3)

    tester = SimpleSystemTester(dimensions=64)
    tester.run_test(duration_seconds=30)