import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import time
from src.ai_monitor.neural_monitor import NeuralNetworkMonitor
from src.ai_monitor.safety_monitor import SafetyMonitor
from src.visualizers.dashboard import MonitoringDashboard
from src.visualizers.pattern_visualizer import PatternVisualizer

class InteractiveTestNetwork(nn.Module):
    """Test network with multiple pattern types"""
    def __init__(self):
        super().__init__()
        # Create multiple pathways for interesting patterns
        self.encoder = nn.Sequential(
            nn.Linear(10, 30),
            nn.ReLU(),
            nn.Linear(30, 20),
            nn.ReLU()
        )

        self.pathway1 = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 10)
        )

        self.pathway2 = nn.Sequential(
            nn.Linear(20, 12),
            nn.Tanh(),
            nn.Linear(12, 10)
        )

        self.combiner = nn.Sequential(
            nn.Linear(20, 15),
            nn.ReLU(),
            nn.Linear(15, 5)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        path1 = self.pathway1(encoded)
        path2 = self.pathway2(encoded)
        combined = torch.cat([path1, path2], dim=1)
        return self.combiner(combined)

def generate_dynamic_input(step: int, batch_size: int = 32):
    """Generate input data with varying patterns"""
    # Base pattern
    t = step * 0.1
    base = torch.tensor([
        np.sin(t),
        np.cos(t),
        np.sin(2*t),
        np.cos(2*t),
        np.sin(3*t),
        np.cos(3*t),
        np.sin(4*t),
        np.cos(4*t),
        np.sin(5*t),
        np.cos(5*t)
    ]).float()

    # Create batch with variations
    x = base.repeat(batch_size, 1)
    x = x + 0.1 * torch.randn_like(x)  # Add noise

    # Add some structured variations
    if step % 20 < 10:  # Periodic pattern shift
        x[:, :5] *= 1.5
    else:
        x[:, 5:] *= 1.5

    return x

def simulate_interactive_training(model: nn.Module, n_steps: int = 200):
    """Simulate training with various pattern types"""
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    # Initialize monitoring
    neural_monitor = NeuralNetworkMonitor()
    safety_monitor = SafetyMonitor(model)
    dashboard = MonitoringDashboard(update_interval=500)  # 0.5 second updates

    print("Starting interactive visualization demo...")
    print("Access dashboard at http://localhost:8050")
    print("Available interactions:")
    print("1. Use time range slider to analyze specific periods")
    print("2. Filter patterns by type (stable, emergent, critical)")
    print("3. Compare patterns using the analysis tools")
    print("4. Track pattern evolution in 3D space")
    print("5. Export analysis results")

    # Start dashboard
    import threading
    dashboard_thread = threading.Thread(
        target=dashboard.run_server,
        kwargs={'debug': False}
    )
    dashboard_thread.daemon = True
    dashboard_thread.start()

    try:
        for step in range(n_steps):
            # Generate dynamic input
            x = generate_dynamic_input(step)
            y = torch.randn(32, 5)  # Random targets

            # Training step
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            # Add pattern variations
            with torch.no_grad():
                if step % 30 == 0:  # Periodic stability variation
                    for param in model.parameters():
                        param.mul_(1.1)  # Increase weights
                elif step % 30 == 15:
                    for param in model.parameters():
                        param.mul_(0.9)  # Decrease weights

            # Monitor system
            network_state = neural_monitor.monitor_network(model)
            safety_status = safety_monitor.check_safety_status()

            # Update dashboard
            dashboard.update_data(network_state, safety_status)

            # Print status
            print(f"\rStep {step+1}/{n_steps} - "
                  f"Patterns: {len(network_state.activation_patterns)} - "
                  f"Risk Level: {safety_status.overall_risk_level:.3f}",
                  end="")

            time.sleep(0.1)  # Simulate real-time updates

    except KeyboardInterrupt:
        print("\nStopping simulation...")

def main():
    # Create model with multiple pattern types
    model = InteractiveTestNetwork()

    # Run interactive simulation
    simulate_interactive_training(model)

if __name__ == "__main__":
    main()