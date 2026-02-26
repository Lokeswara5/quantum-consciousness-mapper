import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import time
from src.ai_monitor.neural_monitor import NeuralNetworkMonitor
from src.ai_monitor.safety_monitor import SafetyMonitor
from src.visualizers.dashboard import MonitoringDashboard
from src.visualizers.pattern_visualizer import PatternVisualizer

# Create a test neural network
class TestNetwork(nn.Module):
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

def simulate_training(model: nn.Module,
                     n_steps: int = 100,
                     noise_level: float = 0.1):
    """Simulate training with varying patterns"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Initialize monitoring
    neural_monitor = NeuralNetworkMonitor()
    safety_monitor = SafetyMonitor(model)
    dashboard = MonitoringDashboard()

    print("Starting visualization demo...")
    print("Access dashboard at http://localhost:8050")

    # Start dashboard in separate thread
    import threading
    dashboard_thread = threading.Thread(
        target=dashboard.run_server,
        kwargs={'debug': False}
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()

    try:
        for step in range(n_steps):
            # Generate synthetic data
            batch_size = 32
            x = torch.randn(batch_size, 10)
            y = torch.randn(batch_size, 5)

            # Training step
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Add synthetic pattern variation
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * noise_level)

            # Monitor system
            network_state = neural_monitor.monitor_network(model)
            safety_status = safety_monitor.check_safety_status()

            # Update dashboard
            dashboard.update_data(network_state, safety_status)

            # Print status
            print(f"\rStep {step+1}/{n_steps} - "
                  f"Safety Score: {safety_status.safety_score:.3f} - "
                  f"Risk Level: {safety_status.overall_risk_level:.3f}",
                  end="")

            time.sleep(0.1)  # Simulate real-time updates

    except KeyboardInterrupt:
        print("\nStopping simulation...")

def main():
    # Create model
    model = TestNetwork()

    # Run simulation
    simulate_training(model)

if __name__ == "__main__":
    main()