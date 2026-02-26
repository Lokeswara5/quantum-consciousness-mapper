"""
Quantum state detector module for identifying and analyzing quantum-like patterns in neural networks.

This module provides detection and analysis of GHZ (Greenberger-Horne-Zeilinger) and W states,
along with quantum metrics like entanglement and coherence.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from functools import lru_cache


@dataclass
class QuantumStateMetrics:
    """Metrics describing the quantum-like properties of a state."""

    state_type: str  # "GHZ", "W", "other"
    correlation_matrix: np.ndarray
    coherence_score: float
    entanglement_measures: Dict[str, float]
    confidence: float


class QuantumStateDetector:
    """Detects and analyzes quantum-like states in neural network patterns."""

    def __init__(self, history_size: int = 1000):
        """Initialize the quantum state detector.

        Args:
            history_size: Number of recent states to keep in history buffer
        """
        self.history_buffer: List[Tuple[np.ndarray, str]] = []
        self.history_size = history_size
        self.state_count = {"GHZ": 0, "W": 0, "other": 0}

    def _calculate_correlation_matrix_chunk(self,
                                   chunk_i: np.ndarray,
                                   chunk_j: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix for chunks of coordinates.

        Args:
            chunk_i: First chunk of coordinates
            chunk_j: Second chunk of coordinates

        Returns:
            Correlation matrix for the chunks
        """
        # Center and normalize chunks
        norm_i = chunk_i - chunk_i.mean(axis=1)[:, None]
        norm_j = chunk_j - chunk_j.mean(axis=1)[:, None]

        # Calculate norms for normalization
        norms_i = np.sqrt(np.sum(norm_i * norm_i, axis=1))
        norms_j = np.sqrt(np.sum(norm_j * norm_j, axis=1))

        # Avoid division by zero
        norms_i[norms_i == 0] = 1
        norms_j[norms_j == 0] = 1

        # Normalize chunks
        norm_i /= norms_i[:, None]
        norm_j /= norms_j[:, None]

        # Calculate correlations
        return np.dot(norm_i, norm_j.T)

    def _calculate_correlation_matrix(self, coordinates: np.ndarray) -> np.ndarray:
        """Calculate the correlation matrix between particles.

        Uses optimized calculation for small matrices.

        Args:
            coordinates: Array of shape (n_particles, n_dimensions)

        Returns:
            Correlation matrix of shape (n_particles, n_particles)
        """
        return self._calculate_correlation_matrix_chunk(coordinates, coordinates)

    def _analyze_eigenvalues(self, correlation_matrix: np.ndarray) -> Tuple[str, float]:
        """Analyze eigenvalue distribution to classify quantum state type.

        Uses efficient eigenvalue computation and analysis.

        Args:
            correlation_matrix: The particle correlation matrix

        Returns:
            Tuple of (state_type, confidence)
        """
        # For small matrices, compute all eigenvalues
        if correlation_matrix.shape[0] <= 100:
            eigenvals = np.abs(np.linalg.eigvals(correlation_matrix))
        else:
            # For large matrices, only compute largest eigenvalues using power iteration
            eigenvals = self._compute_largest_eigenvalues(correlation_matrix, k=3)

        eigenvals.sort()
        total = np.sum(eigenvals)

        if total == 0:
            return "other", 0.0

        # Normalize eigenvalues
        eigenvals /= total

        # Early exit for single eigenvalue case
        if len(eigenvals) == 1:
            return "GHZ" if eigenvals[0] > 0.8 else "other", eigenvals[0]

        # W state: Multiple similar large eigenvalues
        largest_three = eigenvals[-3:]
        if len(largest_three) >= 3:
            std_ratio = np.std(largest_three) / np.mean(largest_three)
            if std_ratio < 0.1:
                confidence = 1.0 - std_ratio
                return "W", confidence

        # GHZ state: One dominant eigenvalue
        if eigenvals[-1] > 0.8:
            confidence = eigenvals[-1]
            return "GHZ", confidence

        return "other", 0.5

    def _compute_largest_eigenvalues(self, matrix: np.ndarray, k: int = 3,
                                max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Compute k largest eigenvalues using power iteration.

        Args:
            matrix: Square matrix
            k: Number of eigenvalues to compute
            max_iter: Maximum iterations
            tol: Convergence tolerance

        Returns:
            Array of k largest eigenvalues
        """
        n = matrix.shape[0]
        eigenvals = []
        current_matrix = matrix.copy()

        for _ in range(k):
            # Initialize random vector
            v = np.random.randn(n)
            v = v / np.linalg.norm(v)

            for _ in range(max_iter):
                # Power iteration
                v_new = current_matrix @ v
                eigenval = np.linalg.norm(v_new)

                if eigenval == 0:
                    break

                v_new = v_new / eigenval

                # Check convergence
                if np.allclose(v, v_new, rtol=tol):
                    break

                v = v_new

            eigenvals.append(eigenval)

            # Deflate matrix
            current_matrix = current_matrix - eigenval * np.outer(v, v)

        return np.array(eigenvals)

    def _calculate_phase_coherence(self, velocities: np.ndarray) -> float:
        """Calculate phase coherence from particle velocities efficiently.

        Uses vectorized operations and optimized complex number handling.

        Args:
            velocities: Array of shape (n_particles, n_dimensions) containing velocities

        Returns:
            Phase coherence score in [0, 1]
        """
        if velocities is None or len(velocities) == 0:
            return 0.0

        # Process in chunks for large arrays
        if len(velocities) > 10000:
            chunk_size = 10000
            coherence_sum = 0.0
            n_chunks = 0

            for i in range(0, len(velocities), chunk_size):
                chunk = velocities[i:i + chunk_size]
                # Calculate phases from x-y velocities
                phases = np.arctan2(chunk[:, 1], chunk[:, 0])
                # Accumulate complex order parameter
                coherence_sum += np.sum(np.exp(1j * phases))
                n_chunks += len(chunk)

            # Calculate final coherence
            phase_coherence = np.abs(coherence_sum / n_chunks)

        else:
            # For smaller arrays, calculate directly
            phases = np.arctan2(velocities[:, 1], velocities[:, 0])
            # Use pre-allocated array for complex exponentials
            complex_exp = np.empty(len(phases), dtype=np.complex128)
            np.exp(1j * phases, out=complex_exp)
            phase_coherence = np.abs(np.mean(complex_exp))

        return float(phase_coherence)

    def _calculate_entanglement_measures(
        self, correlation_matrix: np.ndarray, state_type: str
    ) -> Dict[str, float]:
        """Calculate various entanglement measures efficiently.

        Uses optimized calculations and caching for better performance.

        Args:
            correlation_matrix: The particle correlation matrix
            state_type: Detected state type ("GHZ", "W", "other")

        Returns:
            Dictionary of entanglement measures
        """
        # For large matrices, use eigenvalue approximation
        if correlation_matrix.shape[0] > 100:
            eigenvals = self._compute_largest_eigenvalues(correlation_matrix, k=min(10, correlation_matrix.shape[0]))
        else:
            eigenvals = np.abs(np.linalg.eigvals(correlation_matrix))

        n = correlation_matrix.shape[0]  # Use shape directly to avoid computing length

        # Normalize eigenvalues (using in-place operations)
        total = np.sum(eigenvals)
        if total > 0:
            eigenvals /= total

        # Compute entropy using vectorized operations
        # Add small constant to avoid log(0)
        nz_eigenvals = eigenvals[eigenvals > 1e-10]
        entropy = -np.sum(nz_eigenvals * np.log2(nz_eigenvals))

        # Normalize entropy
        max_entropy = np.log2(n) or 1  # Avoid division by zero
        normalized_entropy = entropy / max_entropy

        # Participation ratio using vectorized operation
        participation = 1.0 / np.sum(np.square(eigenvals))
        normalized_participation = participation / n

        # Use dict literal for slightly better performance
        base_measures = {
            "entropy": normalized_entropy,
            "participation": normalized_participation
        }

        # Add state-specific measures
        if state_type == "GHZ":
            # Optimize GHZ score calculation
            base_measures["ghz_entanglement"] = (normalized_entropy + normalized_participation) / 2
        elif state_type == "W":
            # Optimize W score calculation
            base_measures["w_entanglement"] = normalized_participation * (1 - abs(normalized_entropy - 0.5))

        return base_measures

    def _check_ghz_signature(self, correlation_matrix: np.ndarray) -> bool:
        """Quick check for GHZ state signature without full eigenvalue analysis.

        Args:
            correlation_matrix: The correlation matrix to check

        Returns:
            True if GHZ signature is detected, False otherwise
        """
        # Check diagonal elements (should be close to 1)
        if not np.allclose(np.diag(correlation_matrix), 1.0, rtol=1e-5):
            return False

        # Check off-diagonal elements (should be close to each other)
        off_diag = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
        if len(off_diag) == 0:
            return False

        # Calculate statistics of off-diagonal elements
        mean_corr = np.mean(off_diag)
        std_corr = np.std(off_diag)

        # GHZ signature: high mean correlation and low standard deviation
        return mean_corr > 0.7 and std_corr < 0.1

    def _calculate_ghz_confidence(self, correlation_matrix: np.ndarray) -> float:
        """Calculate confidence score for GHZ state.

        Args:
            correlation_matrix: The correlation matrix

        Returns:
            Confidence score between 0 and 1
        """
        off_diag = correlation_matrix[~np.eye(correlation_matrix.shape[0], dtype=bool)]
        if len(off_diag) == 0:
            return 0.0

        # Confidence based on mean correlation and uniformity
        mean_corr = np.mean(off_diag)
        std_corr = np.std(off_diag)

        # Combine metrics with weights
        confidence = 0.7 * mean_corr + 0.3 * (1 - std_corr)
        return float(np.clip(confidence, 0, 1))

    @lru_cache(maxsize=1000)
    def _get_cached_metrics(self, state_hash: str) -> Optional[QuantumStateMetrics]:
        """Get cached metrics for a state hash.

        Args:
            state_hash: Hash of the state coordinates

        Returns:
            Cached QuantumStateMetrics if found, None otherwise
        """
        return None  # Using lru_cache decorator for automatic caching

    def detect_state_type(
        self, coordinates: np.ndarray, velocities: Optional[np.ndarray] = None
    ) -> QuantumStateMetrics:
        """Detect quantum state type from particle coordinates and velocities.

        Args:
            coordinates: Array of shape (n_particles, n_dimensions) containing positions
            velocities: Optional array of shape (n_particles, n_dimensions) containing velocities

        Returns:
            QuantumStateMetrics with detected state type and measures
        """
        # Quick validation
        if coordinates.size == 0:
            return QuantumStateMetrics(
                state_type="other",
                correlation_matrix=np.array([]),
                coherence_score=0.0,
                entanglement_measures={},
                confidence=0.0
            )

        # Generate state hash for caching
        state_hash = hash(coordinates.tobytes())
        cached_metrics = self._get_cached_metrics(state_hash)
        if cached_metrics is not None:
            return cached_metrics

        # Process in chunks for better memory efficiency
        chunk_size = 1000  # Process 1000 particles at a time
        n_particles = len(coordinates)

        if n_particles > chunk_size:
            # Process correlation matrix in chunks
            correlation_matrix = np.zeros((n_particles, n_particles))
            for i in range(0, n_particles, chunk_size):
                i_end = min(i + chunk_size, n_particles)
                chunk_i = coordinates[i:i_end]

                for j in range(0, n_particles, chunk_size):
                    j_end = min(j + chunk_size, n_particles)
                    chunk_j = coordinates[j:j_end]

                    # Calculate correlation for this chunk
                    corr_chunk = self._calculate_correlation_matrix_chunk(chunk_i, chunk_j)
                    correlation_matrix[i:i_end, j:j_end] = corr_chunk
        else:
            # Small enough to process all at once
            correlation_matrix = self._calculate_correlation_matrix(coordinates)

        # Quick check for GHZ signature
        if self._check_ghz_signature(correlation_matrix):
            state_type = "GHZ"
            confidence = self._calculate_ghz_confidence(correlation_matrix)
        else:
            # Only perform full eigenvalue analysis if needed
            state_type, confidence = self._analyze_eigenvalues(correlation_matrix)

        # Calculate phase coherence if velocities provided
        coherence_score = self._calculate_phase_coherence(velocities) if velocities is not None else 0.0

        # Calculate entanglement measures
        entanglement_measures = self._calculate_entanglement_measures(correlation_matrix, state_type)

        # Update history
        self.history_buffer.append((coordinates, state_type))
        if len(self.history_buffer) > self.history_size:
            self.history_buffer.pop(0)

        # Update state counts
        self.state_count[state_type] = self.state_count.get(state_type, 0) + 1

        return QuantumStateMetrics(
            state_type=state_type,
            correlation_matrix=correlation_matrix,
            coherence_score=coherence_score,
            entanglement_measures=entanglement_measures,
            confidence=confidence
        )

    def get_state_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistics about detected states.

        Returns:
            Dictionary containing state type counts and percentages
        """
        total = sum(self.state_count.values())
        if total == 0:
            return {state: {"count": 0, "percentage": 0.0} for state in ["GHZ", "W", "other"]}

        return {
            state: {
                "count": count,
                "percentage": (count / total) * 100
            }
            for state, count in self.state_count.items()
        }

    def get_transition_matrix(self) -> np.ndarray:
        """Calculate state transition probabilities from history.

        Returns:
            3x3 transition matrix with probabilities between GHZ, W, and other states
        """
        if len(self.history_buffer) < 2:
            return np.zeros((3, 3))

        states = ["GHZ", "W", "other"]
        state_to_idx = {state: idx for idx, state in enumerate(states)}

        # Initialize transition counts
        transitions = np.zeros((3, 3))

        # Count transitions
        for i in range(len(self.history_buffer) - 1):
            from_state = self.history_buffer[i][1]
            to_state = self.history_buffer[i + 1][1]
            transitions[state_to_idx[from_state], state_to_idx[to_state]] += 1

        # Convert to probabilities
        row_sums = transitions.sum(axis=1, keepdims=True)
        transition_probs = np.divide(transitions, row_sums, where=row_sums != 0)

        return transition_probs