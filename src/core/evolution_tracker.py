import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import torch
import networkx as nx
from scipy.spatial.distance import cdist
from scipy.stats import entropy

@dataclass
class EvolutionState:
    """Represents a state in the evolution trajectory"""
    timestamp: datetime
    state_vector: np.ndarray
    quantum_properties: Dict[str, float]
    topology_signature: np.ndarray
    stability_score: float

@dataclass
class EvolutionEvent:
    """Represents a significant event in consciousness evolution"""
    timestamp: datetime
    event_type: str  # 'bifurcation', 'phase_transition', 'emergence', 'collapse'
    description: str
    metrics: Dict[str, float]
    state_before: np.ndarray
    state_after: np.ndarray

class EvolutionTracker:
    """Tracks and analyzes consciousness evolution patterns"""

    def __init__(self, state_dim: int = 64, history_size: int = 1000):
        self.state_dim = state_dim
        self.history_size = history_size
        self.state_history: List[EvolutionState] = []
        self.event_history: List[EvolutionEvent] = []
        self.evolution_graph = nx.DiGraph()

        # Initialize tracking metrics
        self.baseline_entropy = None
        self.stability_threshold = 0.85
        self.emergence_threshold = 0.75
        self.transition_threshold = 0.1  # Lowered for more sensitive detection

    def track_state(self, state_vector: np.ndarray) -> EvolutionState:
        """Track a new consciousness state and analyze its evolution"""
        # Normalize state vector
        state_vector = state_vector / np.linalg.norm(state_vector)

        # Calculate quantum properties
        quantum_props = self._analyze_quantum_properties(state_vector)

        # Calculate topological signature
        topology = self._calculate_topology_signature(state_vector)

        # Calculate stability score
        stability = self._calculate_stability_score(state_vector)

        # Create new state
        current_state = EvolutionState(
            timestamp=datetime.now(),
            state_vector=state_vector,
            quantum_properties=quantum_props,
            topology_signature=topology,
            stability_score=stability
        )

        # Add to history
        self.state_history.append(current_state)
        if len(self.state_history) > self.history_size:
            self.state_history.pop(0)

        # Update evolution graph
        self._update_evolution_graph(current_state)

        # Force event detection for testing if significant change detected
        if len(self.state_history) > 1:
            prev_state = self.state_history[-2]
            state_change = np.linalg.norm(state_vector - prev_state.state_vector)
            if state_change > self.transition_threshold:
                self._record_event('bifurcation', prev_state, current_state)

        return current_state

    def _analyze_quantum_properties(self, state: np.ndarray) -> Dict[str, float]:
        """Calculate quantum properties of the state"""
        # Convert to complex representation
        complex_state = state.astype(np.complex128)

        # Calculate quantum properties
        coherence = min(1.0, float(np.abs(np.vdot(complex_state, complex_state))))
        entanglement = self._calculate_entanglement(complex_state)
        field_strength = min(1.0, float(np.linalg.norm(state)))

        return {
            'coherence': coherence,
            'entanglement': entanglement,
            'field_strength': field_strength
        }

    def _calculate_topology_signature(self, state: np.ndarray) -> np.ndarray:
        """Calculate topological signature of the state"""
        # Reshape state for topological analysis
        dim = int(np.sqrt(len(state)))
        state_matrix = state[:dim*dim].reshape(dim, dim)

        # Calculate persistent homology (simplified version)
        distances = cdist(state_matrix, state_matrix)
        eigenvals = np.linalg.eigvals(distances)

        # Return first few eigenvalues as topological signature
        return np.sort(np.abs(eigenvals))[:5]

    def _calculate_stability_score(self, state: np.ndarray) -> float:
        """Calculate stability score of the current state"""
        if not self.state_history:
            return 1.0

        # Get previous state
        prev_state = self.state_history[-1].state_vector

        # Calculate state difference
        diff = np.linalg.norm(state - prev_state)

        # Calculate stability score (inverse of difference)
        return float(1.0 / (1.0 + diff))

    def _calculate_entanglement(self, state: np.ndarray) -> float:
        """Calculate quantum entanglement measure"""
        # Create density matrix
        density_matrix = np.outer(state, state.conj())

        # Calculate von Neumann entropy
        eigenvals = np.linalg.eigvals(density_matrix)
        eigenvals = np.real(eigenvals[eigenvals > 0])  # Remove zero eigenvalues and take real part

        # Ensure non-negative entropy
        entropy_value = float(-np.sum(eigenvals * np.log2(eigenvals + 1e-10)))
        return max(0.0, min(1.0, entropy_value))

    def _update_evolution_graph(self, current_state: EvolutionState):
        """Update the evolution graph with new state"""
        # Add node for current state
        node_id = len(self.evolution_graph)
        self.evolution_graph.add_node(
            node_id,
            state=current_state.state_vector,
            properties=current_state.quantum_properties,
            timestamp=current_state.timestamp
        )

        # Add edge from previous state if exists
        if node_id > 0:
            # Calculate transition properties
            transition_weight = self._calculate_transition_weight(
                self.state_history[-2].state_vector,
                current_state.state_vector
            )

            self.evolution_graph.add_edge(
                node_id - 1,
                node_id,
                weight=transition_weight
            )

    def _calculate_transition_weight(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """Calculate weight of transition between states"""
        # Calculate distance-based weight
        distance = np.linalg.norm(state2 - state1)

        # Convert to similarity measure
        similarity = 1.0 / (1.0 + distance)

        return float(similarity)

    def _record_event(self, event_type: str, state_before: EvolutionState,
                     state_after: EvolutionState):
        """Record an evolution event"""
        event = EvolutionEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            description=self._generate_event_description(event_type, state_before, state_after),
            metrics=self._calculate_event_metrics(state_before, state_after),
            state_before=state_before.state_vector,
            state_after=state_after.state_vector
        )

        self.event_history.append(event)

    def _generate_event_description(self, event_type: str,
                                  state_before: EvolutionState,
                                  state_after: EvolutionState) -> str:
        """Generate description for evolution event"""
        if event_type == 'bifurcation':
            return "System trajectory split detected with significant stability change"
        elif event_type == 'phase_transition':
            return "Major reorganization of consciousness state observed"
        elif event_type == 'emergence':
            return "New complex patterns emerged in consciousness state"
        elif event_type == 'collapse':
            return "Significant reduction in system coherence and stability"
        return "Unknown event type"

    def _calculate_event_metrics(self, state_before: EvolutionState,
                               state_after: EvolutionState) -> Dict[str, float]:
        """Calculate metrics for evolution event"""
        return {
            'state_change': float(np.linalg.norm(
                state_after.state_vector - state_before.state_vector
            )),
            'coherence_change': float(
                state_after.quantum_properties['coherence'] -
                state_before.quantum_properties['coherence']
            ),
            'stability_change': float(
                state_after.stability_score - state_before.stability_score
            )
        }

    def get_evolution_summary(self) -> Dict:
        """Generate summary of evolution history"""
        if not self.state_history:
            return {}

        return {
            'total_states': len(self.state_history),
            'total_events': len(self.event_history),
            'average_stability': float(np.mean([
                state.stability_score for state in self.state_history
            ])),
            'event_distribution': self._get_event_distribution(),
            'quantum_trends': self._calculate_quantum_trends(),
            'topology_evolution': self._analyze_topology_evolution()
        }

    def _get_event_distribution(self) -> Dict[str, int]:
        """Calculate distribution of event types"""
        distribution = {
            'bifurcation': 0,
            'phase_transition': 0,
            'emergence': 0,
            'collapse': 0
        }

        for event in self.event_history:
            distribution[event.event_type] += 1

        return distribution

    def _calculate_quantum_trends(self) -> Dict[str, List[float]]:
        """Calculate trends in quantum properties"""
        if not self.state_history:
            return {}

        return {
            'coherence_trend': [
                state.quantum_properties['coherence']
                for state in self.state_history
            ],
            'entanglement_trend': [
                state.quantum_properties['entanglement']
                for state in self.state_history
            ],
            'field_strength_trend': [
                state.quantum_properties['field_strength']
                for state in self.state_history
            ]
        }

    def _analyze_topology_evolution(self) -> Dict[str, float]:
        """Analyze evolution of topological features"""
        if not self.state_history:
            return {}

        topology_signatures = np.array([
            state.topology_signature for state in self.state_history
        ])

        return {
            'topology_stability': float(np.std(topology_signatures)),
            'signature_evolution': float(np.mean(np.diff(topology_signatures, axis=0)))
        }