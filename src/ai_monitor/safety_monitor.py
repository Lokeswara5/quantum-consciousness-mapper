import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
from .neural_monitor import NeuralNetworkMonitor, NetworkState
from .training_monitor import TrainingMonitor

@dataclass
class SafetyAlert:
    """Represents a safety alert from the AI system"""
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    affected_components: List[str]
    timestamp: datetime
    recommendations: List[str]
    pattern_signature: np.ndarray

@dataclass
class SafetyStatus:
    """Represents the current safety status of the AI system"""
    overall_risk_level: float
    active_alerts: List[SafetyAlert]
    stability_metrics: Dict[str, float]
    safety_score: float
    last_update: datetime

class SafetyMonitor:
    """AI safety monitoring and early warning system"""

    def __init__(self,
                 model: torch.nn.Module,
                 risk_threshold: float = 0.7,
                 alert_window: int = 100):
        self.neural_monitor = NeuralNetworkMonitor()
        self.training_monitor = TrainingMonitor(model, 0.001, 32)
        self.model = model
        self.risk_threshold = risk_threshold
        self.alert_window = alert_window
        self.alert_history: List[SafetyAlert] = []
        self.known_patterns: Set[str] = set()

    def calculate_risk_level(self,
                           network_state: NetworkState) -> float:
        """Calculate current risk level based on network state"""
        risk_factors = []

        # Check stability
        if network_state.global_stability < 0.5:
            risk_factors.append(0.8)  # High risk for low stability

        # Check warning signals
        risk_factors.append(min(len(network_state.warning_signals) * 0.1, 1.0))

        # Check layer patterns
        for pattern in network_state.activation_patterns.values():
            if pattern.complexity > 0.9:  # Very high complexity
                risk_factors.append(0.7)
            if pattern.emergence_score > 0.8:  # Strong emergence
                risk_factors.append(0.6)

        # Combine risk factors
        if risk_factors:
            return min(np.mean(risk_factors) + 0.1 * len(risk_factors), 1.0)
        return 0.0

    def analyze_pattern_signature(self,
                                network_state: NetworkState) -> np.ndarray:
        """Generate a unique signature for the current network state"""
        signatures = []
        for pattern in network_state.activation_patterns.values():
            # Create pattern fingerprint
            fingerprint = np.array([
                pattern.complexity,
                pattern.stability,
                pattern.emergence_score
            ])
            signatures.append(fingerprint)

        return np.mean(signatures, axis=0) if signatures else np.zeros(3)

    def generate_alert(self,
                      network_state: NetworkState,
                      risk_level: float) -> Optional[SafetyAlert]:
        """Generate safety alert if needed"""
        if risk_level < self.risk_threshold:
            return None

        # Determine severity
        if risk_level >= 0.9:
            severity = 'critical'
        elif risk_level >= 0.8:
            severity = 'high'
        elif risk_level >= 0.7:
            severity = 'medium'
        else:
            severity = 'low'

        # Get affected components
        affected = [
            name for name, pattern in network_state.activation_patterns.items()
            if pattern.potential_issues
        ]

        # Generate recommendations
        recommendations = []
        if 'critical' in severity:
            recommendations.append("Consider immediate system pause")
            recommendations.append("Initiate emergency analysis protocol")

        for name in affected:
            pattern = network_state.activation_patterns[name]
            recommendations.extend(
                self.neural_monitor.get_layer_recommendations(name, pattern)
            )

        # Create alert
        alert = SafetyAlert(
            severity=severity,
            description=f"Risk level {risk_level:.2f} detected",
            affected_components=affected,
            timestamp=datetime.now(),
            recommendations=recommendations,
            pattern_signature=self.analyze_pattern_signature(network_state)
        )

        self.alert_history.append(alert)
        return alert

    def check_safety_status(self) -> SafetyStatus:
        """Check current safety status of the AI system"""
        # Get current network state
        network_state = self.neural_monitor.monitor_network(self.model)

        # Calculate risk level
        risk_level = self.calculate_risk_level(network_state)

        # Generate alert if needed
        alert = self.generate_alert(network_state, risk_level)
        active_alerts = [a for a in self.alert_history[-self.alert_window:]
                        if a.severity in ['high', 'critical']]

        # Calculate stability metrics
        stability_metrics = {
            'global': network_state.global_stability,
            'layer_min': min(p.stability for p in network_state.activation_patterns.values()),
            'layer_max': max(p.stability for p in network_state.activation_patterns.values())
        }

        # Calculate safety score
        safety_factors = [
            1 - risk_level,
            network_state.global_stability,
            1 - len(active_alerts) * 0.1
        ]
        safety_score = np.mean(safety_factors)

        return SafetyStatus(
            overall_risk_level=risk_level,
            active_alerts=active_alerts,
            stability_metrics=stability_metrics,
            safety_score=safety_score,
            last_update=datetime.now()
        )

    def get_safety_recommendations(self,
                                 status: SafetyStatus) -> List[str]:
        """Generate safety recommendations based on current status"""
        recommendations = []

        if status.overall_risk_level > self.risk_threshold:
            if status.overall_risk_level > 0.9:
                recommendations.append("CRITICAL: Consider immediate system shutdown")
                recommendations.append("Initiate full system diagnostic")
            elif status.overall_risk_level > 0.8:
                recommendations.append("HIGH RISK: Pause normal operations")
                recommendations.append("Review recent system changes")

        if status.stability_metrics['global'] < 0.6:
            recommendations.append(
                "Low system stability detected. Consider recalibration."
            )

        if status.active_alerts:
            recommendations.append(
                f"Address {len(status.active_alerts)} active high-priority alerts"
            )

        if status.safety_score < 0.7:
            recommendations.append(
                "Overall safety score below threshold. Review system parameters."
            )

        return recommendations

    def monitor_training_safety(self) -> List[str]:
        """Monitor safety during training"""
        issues = self.training_monitor.detect_training_issues()
        safety_issues = []

        for issue in issues:
            if "exploding" in issue or "vanishing" in issue:
                safety_issues.append(f"Training instability: {issue}")
            elif "emergence" in issue:
                safety_issues.append(f"Unexpected pattern: {issue}")

        return safety_issues