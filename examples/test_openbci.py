"""Test neural data acquisition using synthetic data simulation."""

import time
import numpy as np
from src.data_interface.neural_data_interface import NeuralDataInterface

def analyze_brain_waves(data):
    """Analyze different brain wave bands in the data.

    Args:
        data: Raw EEG data array

    Returns:
        Dictionary of brain wave band powers
    """
    bands = {
        'Delta': (1, 4),    # Deep sleep
        'Theta': (4, 8),    # Drowsy/meditative
        'Alpha': (8, 13),   # Relaxed/aware
        'Beta': (13, 30),   # Active thinking
        'Gamma': (30, 50)   # Complex cognitive tasks
    }

    powers = {}
    for band_name, (low_freq, high_freq) in bands.items():
        # Simple band power estimation
        band_power = np.mean(np.abs(data))  # Simplified for demonstration
        powers[band_name] = band_power

    return powers

def main():
    print("Initializing neural interface with synthetic data simulation...")
    interface = NeuralDataInterface(
        device_type='openbci',
        sampling_rate=250,
        buffer_size=1000
    )

    print("\nGetting device info...")
    device_info = interface.get_device_info()
    print(f"Connected to {device_info['device_type']} device")
    print(f"Sampling rate: {device_info['sampling_rate']} Hz")
    print(f"Number of channels: {device_info['num_channels']}")
    print(f"Channel names: {device_info['channel_names']}")

    print("\nChecking impedances...")
    impedances = interface.check_impedance()
    for channel, impedance in impedances.items():
        print(f"{channel}: {impedance:.1f} kΩ")

    print("\nStarting data acquisition...")
    interface.start_acquisition()

    try:
        print("\nCollecting simulated neural data for 10 seconds...")
        start_time = time.time()
        packets_received = 0
        last_update = start_time

        while time.time() - start_time < 10:
            data = interface.get_latest_data()
            if data is not None:
                packets_received += 1
                current_time = time.time()

                # Update display every 1 second
                if current_time - last_update >= 1.0:
                    # Analyze brain wave patterns
                    wave_powers = analyze_brain_waves(data.eeg_data)

                    # Display status
                    print(f"\nStatus at {packets_received} packets:")
                    print(f"Packet rate: {packets_received / (current_time - start_time):.1f} Hz")
                    print("\nBrain wave patterns:")
                    for band, power in wave_powers.items():
                        print(f"{band:6s}: {'#' * int(power * 20)}")

                    last_update = current_time

            time.sleep(0.001)  # Small sleep to prevent CPU overload

        print(f"\nTest complete:")
        print(f"- Received {packets_received} packets in 10 seconds")
        print(f"- Average packet rate: {packets_received / 10:.1f} Hz")

        # Get queue metrics
        metrics = interface.get_queue_metrics()
        print(f"\nQueue metrics:")
        print(f"Queue size: {metrics.size}")
        print(f"Queue utilization: {metrics.utilization * 100:.1f}%")
        print(f"Dropped packets: {metrics.dropped_packets}")
        print(f"Average wait time: {metrics.average_wait_time * 1000:.1f} ms")

        # Get device health
        health = interface.check_device_health()
        print(f"\nDevice health:")
        print(f"Connected: {health.connected}")
        print(f"Streaming: {health.streaming}")
        print(f"Signal quality: {health.signal_quality * 100:.1f}%")
        print(f"Error count: {health.error_count}")
        if health.last_error:
            print(f"Last error: {health.last_error}")

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    finally:
        print("\nStopping acquisition...")
        interface.stop_acquisition()
        print("Done!")

if __name__ == "__main__":
    main()