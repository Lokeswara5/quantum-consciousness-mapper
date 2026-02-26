import torch
import torch.nn as nn
import torch.optim as optim
from src.ai_monitor.neural_monitor import NeuralNetworkMonitor
from src.ai_monitor.training_monitor import TrainingMonitor
from src.ai_monitor.safety_monitor import SafetyMonitor

# Create a simple neural network for demonstration
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5)
        )

    def forward(self, x):
        return self.layers(x)

def main():
    # Initialize model and monitoring systems
    model = SimpleNN()
    neural_monitor = NeuralNetworkMonitor()
    training_monitor = TrainingMonitor(model, learning_rate=0.001, batch_size=32)
    safety_monitor = SafetyMonitor(model)

    # Simulate training loop
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    print("Starting AI system monitoring...")
    print("=" * 50)

    for epoch in range(5):
        # Simulate batch of data
        inputs = torch.randn(32, 10)
        targets = torch.randn(32, 5)

        # Training step
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        # Monitor neural network state
        network_state = neural_monitor.monitor_network(model)
        print(f"\nEpoch {epoch + 1}:")
        print(f"Global Stability: {network_state.global_stability:.3f}")

        if network_state.warning_signals:
            print("Warnings detected:")
            for warning in network_state.warning_signals:
                print(f"- {warning}")

        # Monitor training patterns
        emergence = training_monitor.analyze_training_step(loss.item(), 0)
        print("\nEmergence Patterns:")
        for layer, score in emergence.items():
            print(f"- {layer}: {score:.3f}")

        # Check safety status
        safety_status = safety_monitor.check_safety_status()
        print("\nSafety Status:")
        print(f"Risk Level: {safety_status.overall_risk_level:.3f}")
        print(f"Safety Score: {safety_status.safety_score:.3f}")

        if safety_status.active_alerts:
            print("\nActive Safety Alerts:")
            for alert in safety_status.active_alerts:
                print(f"[{alert.severity.upper()}] {alert.description}")
                for rec in alert.recommendations:
                    print(f"  - {rec}")

        # Get recommendations
        recommendations = safety_monitor.get_safety_recommendations(safety_status)
        if recommendations:
            print("\nRecommendations:")
            for rec in recommendations:
                print(f"- {rec}")

        print("=" * 50)

if __name__ == "__main__":
    main()