"""Real-time neural data interface for quantum consciousness mapping."""

import numpy as np
import threading
import queue
import time
import collections
from typing import Optional, Dict, List, Tuple, NamedTuple
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
import mne
from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
from brainflow.data_filter import DataFilter, FilterTypes, WindowOperations

@dataclass
class QueueMetrics(NamedTuple):
    """Metrics about queue status."""
    size: int
    utilization: float
    dropped_packets: int
    average_wait_time: float

class DeviceStatus(NamedTuple):
    """Status of the neural recording device."""
    connected: bool
    streaming: bool
    last_data_time: float
    signal_quality: float  # 0-1 scale
    error_count: int
    last_error: Optional[str]

class NeuralDataPacket:
    """Container for real-time neural data."""
    timestamp: float
    eeg_data: np.ndarray
    channel_names: List[str]
    sampling_rate: int
    channel_types: List[str]

class NeuralDataInterface:
    """Interface for real-time neural data acquisition and preprocessing."""

    def __init__(self,
                 device_type: str = 'openbci',
                 buffer_size: int = 1000,
                 sampling_rate: int = 250):
        """Initialize neural data interface.

        Args:
            device_type: Type of neural recording device ('openbci', 'muse', etc.)
            buffer_size: Size of data buffer
            sampling_rate: Data sampling rate in Hz
        """
        self.device_type = device_type
        self.buffer_size = buffer_size
        self.sampling_rate = sampling_rate

        # Data buffers with size limits
        self.data_queue = queue.Queue(maxsize=1000)  # ~4 seconds of data at 250Hz
        self.raw_buffer = collections.deque(maxlen=self.buffer_size)

        # Circular buffer for efficient history tracking
        history_size = max(1000, self.buffer_size * 2)  # At least 1000 samples
        self.history_buffer = collections.deque(maxlen=history_size)
        self.metrics_buffer = collections.deque(maxlen=history_size)

        # Device configuration
        self._setup_device()

        # Processing parameters
        self.bandpass_filter = (1, 50)  # Hz
        self.notch_filter = 60  # Hz

        # Threading and synchronization
        self._running = False
        self._running_lock = threading.Lock()
        self._acquisition_thread = None
        self._stop_event = threading.Event()

        # Metrics with thread-safe access
        self._metrics_lock = threading.Lock()
        self._dropped_packets = 0
        self._packet_timestamps = collections.deque(maxlen=1000)
        self._last_wait_time_update = time.time()
        self._error_count = 0
        self._last_error = None

    def _setup_device(self, max_retries: int = 3, retry_delay: float = 1.0):
        """Set up neural recording device with retry logic.

        Args:
            max_retries: Maximum number of connection attempts
            retry_delay: Delay between retries in seconds

        Raises:
            RuntimeError: If device setup fails after all retries
            ValueError: If device type is unsupported
        """
        if self.device_type == 'openbci':
            # OpenBCI configuration
            self.params = BrainFlowInputParams()
            self.board_id = BoardIds.SYNTHETIC_BOARD  # Replace with actual board ID

            last_error = None
            for attempt in range(max_retries):
                try:
                    print(f"Attempting to initialize device (attempt {attempt + 1}/{max_retries})")
                    self.board = BoardShim(self.board_id, self.params)

                    # Verify board communication
                    self.board.prepare_session()
                    self.board.release_session()
                    return  # Success

                except Exception as e:
                    last_error = e
                    print(f"Error initializing device (attempt {attempt + 1}): {e}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay * (attempt + 1))  # Exponential backoff
                    continue

            # All retries failed
            raise RuntimeError(f"Failed to initialize device after {max_retries} attempts: {last_error}")

        elif self.device_type == 'muse':
            # Muse headband configuration
            try:
                self.params = {
                    'backend': 'bluemuse',
                    'serial_port': None
                }
                # Add Muse-specific connection verification here
                # TODO: Implement Muse device validation
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Muse device: {e}")

        else:
            raise ValueError(f"Unsupported device type: {self.device_type}")

    def start_acquisition(self):
        """Start real-time data acquisition."""
        with self._running_lock:
            if self._running:
                return

            try:
                # Reset stop event
                self._stop_event.clear()

                # Initialize board
                self.board.prepare_session()
                self.board.start_stream()

                # Mark as running only after successful initialization
                self._running = True

                # Start acquisition thread
                self._acquisition_thread = threading.Thread(target=self._acquisition_loop)
                self._acquisition_thread.daemon = True
                self._acquisition_thread.start()

            except Exception as e:
                # Cleanup on initialization failure
                self._running = False
                if hasattr(self, 'board'):
                    try:
                        self.board.release_session()
                    except:
                        pass
                raise RuntimeError(f"Failed to start acquisition: {str(e)}")

    def stop_acquisition(self, timeout: float = 5.0):
        """Stop data acquisition.

        Args:
            timeout: Maximum time to wait for thread shutdown in seconds

        Raises:
            RuntimeError: If shutdown fails or times out
        """
        with self._running_lock:
            if not self._running:
                return

            # Signal thread to stop
            self._stop_event.set()
            self._running = False

        cleanup_success = True
        try:
            # Wait for acquisition thread with timeout
            if self._acquisition_thread:
                self._acquisition_thread.join(timeout=timeout)
                if self._acquisition_thread.is_alive():
                    raise RuntimeError("Acquisition thread failed to stop")

            # Stop and release board
            try:
                self.board.stop_stream()
            except Exception as e:
                cleanup_success = False
                print(f"Error stopping stream: {e}")

            try:
                self.board.release_session()
            except Exception as e:
                cleanup_success = False
                print(f"Error releasing session: {e}")

        finally:
            # Reset thread state
            self._acquisition_thread = None

            if not cleanup_success:
                raise RuntimeError("Failed to cleanly stop acquisition")

    def _acquisition_loop(self):
        """Main data acquisition loop."""
        while not self._stop_event.is_set():
            try:
                # Check if we should still be running
                with self._running_lock:
                    if not self._running:
                        break

                # Get new data
                if self.board.get_board_data_count() > 0:
                    try:
                        data = self.board.get_current_board_data(self.buffer_size)
                    except Exception as e:
                        print(f"Error getting board data: {e}")
                        if not self._stop_event.wait(timeout=0.1):  # 100ms retry delay
                            continue
                        break

                    try:
                        # Preprocess data
                        filtered_data = self._preprocess_data(data)

                        # Create data packet
                        packet = NeuralDataPacket(
                            timestamp=time.time(),
                            eeg_data=filtered_data,
                            channel_names=self.board.get_eeg_names(self.board_id),
                            sampling_rate=self.board.get_sampling_rate(self.board_id),
                            channel_types=['eeg'] * filtered_data.shape[0]
                        )
                    except Exception as e:
                        print(f"Error processing data: {e}")
                        if not self._stop_event.wait(timeout=0.1):
                            continue
                        break

                # Add to queue with backpressure
                try:
                    # Record packet timestamp before queuing
                    self._packet_timestamps.append(time.time())
                    self.data_queue.put(packet, timeout=0.1)  # 100ms timeout
                except queue.Full:
                    # Queue is full - implement backpressure
                    self._dropped_packets += 1
                    print(f"Warning: Data queue full - implementing backpressure (dropped {self._dropped_packets} packets)")
                    self._packet_timestamps.pop()  # Remove timestamp for dropped packet
                    time.sleep(0.01)  # 10ms sleep when backed up
                    continue

                # Dynamic sleep based on queue utilization
                utilization = self.data_queue.qsize() / self.data_queue.maxsize
                if utilization > 0.8:  # Over 80% full
                    time.sleep(0.005)  # 5ms sleep
                elif utilization > 0.5:  # Over 50% full
                    time.sleep(0.002)  # 2ms sleep
                else:
                    time.sleep(0.001)  # Default 1ms sleep

    def _preprocess_data(self, data: np.ndarray) -> np.ndarray:
        """Preprocess neural data with validation and filtering.

        Args:
            data: Raw neural data array

        Returns:
            Preprocessed data array

        Raises:
            ValueError: If data validation fails
            RuntimeError: If preprocessing operations fail
        """
        # Input validation
        if data is None or data.size == 0:
            raise ValueError("Empty or null input data")

        if not isinstance(data, np.ndarray):
            raise ValueError(f"Expected numpy array, got {type(data)}")

        # Check for NaN/inf values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise ValueError("Input data contains NaN or infinite values")

        try:
            # Convert to float32
            data = data.astype(np.float32)

            # Validate data shape
            expected_channels = self.board.get_eeg_channels(self.board_id)
            if data.shape[0] != len(expected_channels):
                raise ValueError(
                    f"Invalid channel count. Expected {len(expected_channels)}, got {data.shape[0]}"
                )

            # Optimize memory allocation and computation
            # Calculate signal quality using vectorized operations
            with np.errstate(divide='ignore', invalid='ignore'):
                # Use efficient array operations
                channel_std = np.std(data, axis=1, keepdims=True)
                signal_power = np.mean(np.square(data), axis=1, keepdims=True)
                noise_threshold = 0.1 * np.max(signal_power)

                # Vectorized channel quality check
                dead_channels = (channel_std.ravel() < noise_threshold) | (signal_power.ravel() < noise_threshold)
                if np.any(dead_channels):
                    print(f"Warning: Possible dead channels: {np.where(dead_channels)[0]}")

            # Preallocate filtered data array to avoid copies
            filtered_data = np.empty_like(data)
            np.copyto(filtered_data, data)

            # Apply filters in-place with error handling
            try:
                # Batch process channels for better cache utilization
                chunk_size = min(4, data.shape[0])  # Process up to 4 channels at once
                for i in range(0, data.shape[0], chunk_size):
                    chunk = slice(i, min(i + chunk_size, data.shape[0]))

                    # Bandpass filter
                    DataFilter.perform_bandpass(
                        filtered_data[chunk],
                        self.sampling_rate,
                        self.bandpass_filter[0],
                        self.bandpass_filter[1],
                        4,
                        FilterTypes.BUTTERWORTH.value,
                        0
                    )

                    # Notch filter
                    DataFilter.perform_bandstop(
                        filtered_data[chunk],
                        self.sampling_rate,
                        self.notch_filter,
                        4,
                        4,
                        FilterTypes.BUTTERWORTH.value,
                        0
                    )

            except Exception as e:
                raise RuntimeError(f"Filtering failed: {e}")

            # Validate output using masked operations for efficiency
            invalid_mask = np.isnan(filtered_data) | np.isinf(filtered_data)
            if np.any(invalid_mask):
                raise RuntimeError("Filtering produced NaN or infinite values")

            # Update signal quality metrics in history
            with self._metrics_lock:
                self.metrics_buffer.append({
                    'time': time.time(),
                    'signal_quality': np.mean(~dead_channels),
                    'channel_std': channel_std.copy()
                })

            # Detect patterns efficiently using rolling window
            pattern_length = min(self.sampling_rate // 2, filtered_data.shape[1])  # 0.5s window
            if filtered_data.shape[1] >= pattern_length:
                # Use stride tricks for memory-efficient rolling window
                from numpy.lib.stride_tricks import as_strided
                window_shape = (filtered_data.shape[0], filtered_data.shape[1] - pattern_length + 1, pattern_length)
                window_strides = (filtered_data.strides[0], filtered_data.strides[1], filtered_data.strides[1])
                windows = as_strided(filtered_data, shape=window_shape, strides=window_strides, writeable=False)

                # Calculate pattern features using vectorized operations
                mean_pattern = np.mean(windows, axis=1)
                std_pattern = np.std(windows, axis=1)

                # Store pattern data in circular buffer
                with self._metrics_lock:
                    self.history_buffer.append({
                        'time': time.time(),
                        'mean_pattern': mean_pattern,
                        'std_pattern': std_pattern,
                        'pattern_length': pattern_length
                    })

            data = filtered_data

            return data

        except Exception as e:
            # Add context to any unhandled errors
            raise RuntimeError(f"Preprocessing failed: {str(e)}") from e

    def get_latest_data(self) -> Optional[NeuralDataPacket]:
        """Get latest preprocessed data packet.

        Returns:
            Latest neural data packet if available, None otherwise
        """
        try:
            return self.data_queue.get_nowait()
        except queue.Empty:
            return None

    def _cleanup_old_data(self, max_age: float = 3600.0) -> None:
        """Clean up old data from history buffers.

        Args:
            max_age: Maximum age of data to keep in seconds (default 1 hour)
        """
        current_time = time.time()

        with self._metrics_lock:
            # Use efficient deque operations
            while (self.metrics_buffer and
                  current_time - self.metrics_buffer[0]['time'] > max_age):
                self.metrics_buffer.popleft()

            while (self.history_buffer and
                  current_time - self.history_buffer[0]['time'] > max_age):
                self.history_buffer.popleft()

            # Clean up packet timestamps
            while (self._packet_timestamps and
                  current_time - self._packet_timestamps[0] > max_age):
                self._packet_timestamps.popleft()

    def _calculate_average_wait_time(self) -> float:
        """Calculate average packet wait time in queue.

        Returns:
            Average wait time in seconds
        """
        with self._metrics_lock:
            now = time.time()
            if not self._packet_timestamps:
                return 0.0

            # Calculate average wait time from timestamps
            total_wait = sum(now - ts for ts in self._packet_timestamps)
            return total_wait / len(self._packet_timestamps)

    def check_device_health(self, timeout: float = 1.0) -> DeviceStatus:
        """Check health of device connection.

        Args:
            timeout: Maximum time to wait for device response in seconds

        Returns:
            DeviceStatus containing current device health metrics
        """
        try:
            # Try to get device status with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                if hasattr(self, 'board'):
                    try:
                        # Check if board is responsive
                        data_count = self.board.get_board_data_count()
                        is_streaming = self._running and data_count >= 0

                        # Calculate signal quality from recent data
                        signal_quality = 1.0
                        if is_streaming and data_count > 0:
                            recent_data = self.board.get_current_board_data(min(data_count, 250))
                            channel_std = np.std(recent_data, axis=1)
                            # Consider channels with very low variance as potentially dead
                            active_channels = channel_std > 0.1
                            signal_quality = np.mean(active_channels)

                        with self._metrics_lock:
                            return DeviceStatus(
                                connected=True,
                                streaming=is_streaming,
                                last_data_time=time.time(),
                                signal_quality=signal_quality,
                                error_count=self._error_count,
                                last_error=self._last_error
                            )

                    except Exception as e:
                        # Board exists but not responding properly
                        return DeviceStatus(
                            connected=False,
                            streaming=False,
                            last_data_time=0,
                            signal_quality=0.0,
                            error_count=self._error_count + 1,
                            last_error=str(e)
                        )

                time.sleep(0.1)  # Short sleep between checks

            # Timeout reached
            return DeviceStatus(
                connected=False,
                streaming=False,
                last_data_time=0,
                signal_quality=0.0,
                error_count=self._error_count + 1,
                last_error="Device health check timed out"
            )

        except Exception as e:
            # Unexpected error during health check
            return DeviceStatus(
                connected=False,
                streaming=False,
                last_data_time=0,
                signal_quality=0.0,
                error_count=self._error_count + 1,
                last_error=f"Health check failed: {str(e)}"
            )

    def get_queue_metrics(self) -> QueueMetrics:
        """Get current queue metrics efficiently.

        Returns:
            QueueMetrics with current queue status
        """
        # Get queue size without lock contention
        try:
            current_size = self.data_queue.qsize()
        except NotImplementedError:
            # Fallback for platforms where qsize is not implemented
            current_size = sum(1 for _ in self.data_queue._queue)

        utilization = current_size / self.data_queue.maxsize

        # Calculate metrics under a single lock
        with self._metrics_lock:

        return QueueMetrics(
            size=current_size,
            utilization=utilization,
            dropped_packets=self._dropped_packets,
            average_wait_time=self._calculate_average_wait_time()
        )

    def get_device_info(self) -> Dict:
        """Get neural recording device information.

        Returns:
            Dictionary containing device information
        """
        return {
            'device_type': self.device_type,
            'sampling_rate': self.sampling_rate,
            'num_channels': self.board.get_num_rows(self.board_id),
            'channel_names': self.board.get_eeg_names(self.board_id),
            'board_id': self.board_id
        }

class NeuralStateMapper:
    """Maps neural data to quantum consciousness states."""

    def __init__(self):
        """Initialize neural state mapper."""
        self.frequency_bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

    def extract_features(self, data: NeuralDataPacket) -> Dict[str, np.ndarray]:
        """Extract relevant features from neural data.

        Args:
            data: Neural data packet

        Returns:
            Dictionary of extracted features
        """
        features = {}

        # Calculate band powers
        for band_name, (low_freq, high_freq) in self.frequency_bands.items():
            band_power = self._calculate_band_power(
                data.eeg_data,
                data.sampling_rate,
                low_freq,
                high_freq
            )
            features[f'{band_name}_power'] = band_power

        # Calculate coherence between channels
        coherence = self._calculate_coherence(data.eeg_data)
        features['coherence'] = coherence

        return features

    def _calculate_band_power(self,
                            data: np.ndarray,
                            sampling_rate: int,
                            low_freq: float,
                            high_freq: float) -> np.ndarray:
        """Calculate power in specific frequency band.

        Args:
            data: EEG data array
            sampling_rate: Sampling rate in Hz
            low_freq: Lower frequency bound
            high_freq: Upper frequency bound

        Returns:
            Band power array
        """
        return DataFilter.get_band_power(data, sampling_rate, low_freq, high_freq)

    def _calculate_coherence(self, data: np.ndarray) -> np.ndarray:
        """Calculate coherence between channels.

        Args:
            data: EEG data array

        Returns:
            Channel coherence matrix
        """
        n_channels = data.shape[0]
        coherence = np.zeros((n_channels, n_channels))

        for i in range(n_channels):
            for j in range(i+1, n_channels):
                coh = DataFilter.get_coherence(data[i], data[j])
                coherence[i,j] = coh
                coherence[j,i] = coh

        return coherence

    def map_to_quantum_state(self,
                           features: Dict[str, np.ndarray]) -> Tuple[str, float]:
        """Map neural features to quantum state type.

        Args:
            features: Dictionary of neural features

        Returns:
            Tuple of (state_type, confidence)
        """
        # Calculate global coherence
        global_coherence = np.mean(features['coherence'])

        # Calculate band power ratios
        total_power = sum(features[f'{band}_power'] for band in self.frequency_bands)
        power_ratios = {
            band: features[f'{band}_power'] / total_power
            for band in self.frequency_bands
        }

        # Detect quantum states based on neural patterns
        if global_coherence > 0.7 and power_ratios['gamma'] > 0.3:
            # High coherence and gamma - GHZ state
            state_type = "GHZ"
            confidence = global_coherence * power_ratios['gamma']
        elif 0.4 < global_coherence < 0.7 and power_ratios['beta'] > 0.3:
            # Moderate coherence and beta - W state
            state_type = "W"
            confidence = global_coherence * power_ratios['beta']
        else:
            # Other states
            state_type = "other"
            confidence = 0.5

        return state_type, confidence