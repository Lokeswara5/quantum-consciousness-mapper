import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from .neural_monitor import NeuralNetworkMonitor, NetworkState

@dataclass
class TrainingPattern:
    """Represents a pattern detected during training"""
    epoch: int
    layer_patterns: Dict[str, float]
    loss_trend: np.ndarray
    gradient_norm: float
    emergence_indicators: Dict[str, float]
    timestamp: datetime

class TrainingMonitor:
    """Monitor and analyze patterns during neural network training"""

    def __init__(self,
                 model: torch.nn.Module,
                 learning_rate: float,
                 batch_size: int):
        self.neural_monitor = NeuralNetworkMonitor()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.training_history: List[TrainingPattern] = []
        self.current_epoch = 0

    def compute_gradient_norm(self) -> float:
        """Compute the norm of gradients across all parameters"""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
        return np.sqrt(total_norm)

    def analyze_training_step(self,
                            loss: float,
                            batch_idx: int) -> Dict[str, float]:
        """Analyze patterns during a single training step"""
        # Get current network state
        network_state = self.neural_monitor.monitor_network(self.model)

        # Calculate emergence indicators
        emergence = {}
        for name, pattern in network_state.activation_patterns.items():
            emergence[name] = pattern.emergence_score

        # Track gradients
        gradient_norm = self.compute_gradient_norm()

        # Create training pattern
        pattern = TrainingPattern(
            epoch=self.current_epoch,
            layer_patterns={
                name: p.complexity
                for name, p in network_state.activation_patterns.items()
            },
            loss_trend=np.array([loss]),
            gradient_norm=gradient_norm,
            emergence_indicators=emergence,
            timestamp=datetime.now()
        )

        self.training_history.append(pattern)
        return emergence

    def detect_training_issues(self) -> List[str]:
        """Detect potential issues in training patterns"""
        if len(self.training_history) < 2:
            return []

        issues = []

        # Analyze loss trend
        recent_losses = np.array([
            p.loss_trend[0]
            for p in self.training_history[-10:]
        ])
        loss_change = (recent_losses[-1] - recent_losses[0]) / recent_losses[0]

        if abs(loss_change) < 0.001:
            issues.append("Training might be stagnating")

        # Analyze gradient behavior
        recent_grads = np.array([
            p.gradient_norm
            for p in self.training_history[-10:]
        ])
        if np.mean(recent_grads) < 0.0001:
            issues.append("Gradients are vanishing")
        elif np.mean(recent_grads) > 10.0:
            issues.append("Gradients might be exploding")

        # Analyze emergence patterns
        for pattern in self.training_history[-1].emergence_indicators.items():
            name, score = pattern
            if score > 0.8:  # High emergence threshold
                issues.append(f"High pattern emergence in {name}")

        return issues

    def get_training_recommendations(self) -> List[str]:
        """Generate recommendations for improving training"""
        if len(self.training_history) < 2:
            return []

        recommendations = []
        issues = self.detect_training_issues()

        for issue in issues:
            if "stagnating" in issue:
                recommendations.append(
                    f"Consider increasing learning rate (current: {self.learning_rate})"
                )
            elif "vanishing" in issue:
                recommendations.append(
                    "Consider using skip connections or adjusting initialization"
                )
            elif "exploding" in issue:
                recommendations.append(
                    "Consider gradient clipping or reducing learning rate"
                )
            elif "emergence" in issue:
                recommendations.append(
                    "Monitor for overfitting and consider regularization"
                )

        return recommendations

    def visualize_training_patterns(self) -> Dict[str, np.ndarray]:
        """Generate visualizations of training patterns"""
        if len(self.training_history) < 2:
            return {}

        visualizations = {}

        # Loss trajectory
        losses = np.array([
            p.loss_trend[0]
            for p in self.training_history
        ])
        visualizations['loss_trajectory'] = losses

        # Emergence patterns over time
        emergence_trends = {}
        for pattern in self.training_history:
            for layer, score in pattern.emergence_indicators.items():
                if layer not in emergence_trends:
                    emergence_trends[layer] = []
                emergence_trends[layer].append(score)

        for layer, trends in emergence_trends.items():
            visualizations[f'emergence_{layer}'] = np.array(trends)

        # Gradient norms
        gradient_norms = np.array([
            p.gradient_norm
            for p in self.training_history
        ])
        visualizations['gradient_norms'] = gradient_norms

        return visualizations

    def update_epoch(self, epoch: int):
        """Update the current epoch counter"""
        self.current_epoch = epoch

    def reset_history(self):
        """Clear training history"""
        self.training_history = []