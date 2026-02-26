"""Real-time monitoring dashboard for quantum consciousness mapping."""

import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from .pattern_visualizer import PatternVisualizer
from ..ai_monitor.neural_monitor import NetworkState
from ..ai_monitor.safety_monitor import SafetyStatus


class MonitoringDashboard:
    """Real-time dashboard for AI system monitoring."""

    def __init__(self,
                 update_interval: int = 1000,  # milliseconds
                 max_history: int = 1000):
        """Initialize the dashboard.

        Args:
            update_interval: Update interval in milliseconds
            max_history: Maximum number of historical data points to keep
        """
        self.app = dash.Dash(
            __name__,
            external_stylesheets=[
                'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap'
            ]
        )
        self.app.config.suppress_callback_exceptions = True
        self.app.title = 'Quantum Consciousness Mapper'

        self.visualizer = PatternVisualizer()
        self.update_interval = update_interval
        self.max_history = max_history

        # Initialize data storage
        self.metrics_history = {
            'stability': [],
            'complexity': [],
            'risk_level': [],
            'safety_score': []
        }
        self.timestamps = []
        self.pattern_history = []
        self.current_network_state = None
        self.current_safety_status = None

        self._setup_layout()
        self._setup_callbacks()

    def _setup_layout(self):
        """Setup dashboard layout."""
        from .dashboard_layout import create_layout
        self.app.layout = create_layout(self.app)

    def _setup_callbacks(self):
        """Setup dashboard callbacks."""
        @self.app.callback(
            [Output('risk-gauge', 'figure'),
             Output('system-status', 'children'),
             Output('active-alerts', 'children'),
             Output('pattern-space', 'figure'),
             Output('quantum-state-analysis', 'figure'),
             Output('metrics-timeline', 'figure'),
             Output('quantum-transitions', 'figure'),
             Output('quantum-stats', 'children')],
            [Input('interval-component', 'n_intervals')]
        )
        def update_dashboard(n):
            if self.current_network_state is None or self.current_safety_status is None:
                return self._generate_empty_plots()

            # Update risk gauge
            risk_gauge = self.visualizer.create_risk_gauge(
                self.current_safety_status.overall_risk_level
            )

            # Update system status
            status_div = self._create_status_div()

            # Update alerts
            alerts_div = self._create_alerts_div()

            # Get current pattern with quantum metrics
            current_pattern = None
            current_quantum_metrics = None
            if self.pattern_history:
                current_pattern = np.vstack(self.pattern_history[-1])
                if self.current_network_state.activation_patterns:
                    first_pattern = next(iter(self.current_network_state.activation_patterns.values()))
                    if first_pattern.quantum_metrics:
                        current_quantum_metrics = first_pattern.quantum_metrics

            # Update pattern space with quantum visualization
            pattern_space = self.visualizer.create_3d_pattern_plot(
                current_pattern,
                quantum_metrics=current_quantum_metrics,
                title="Current Neural Pattern Space"
            ) if current_pattern is not None else go.Figure()

            # Update quantum state analysis
            quantum_analysis = self.visualizer.create_quantum_state_plot(
                current_quantum_metrics,
                title="Quantum State Analysis"
            ) if current_quantum_metrics else go.Figure()

            # Update metrics timeline
            metrics_timeline = self.visualizer.create_time_series_dashboard(
                self.metrics_history,
                self.timestamps
            )

            # Update quantum transitions
            quantum_transitions = self.visualizer.create_quantum_transition_matrix(
                self.current_safety_status.quantum_transition_matrix,
                title="Quantum State Transitions"
            ) if self.current_safety_status.quantum_transition_matrix is not None else go.Figure()

            # Create quantum stats display
            quantum_stats = None
            if self.current_safety_status.quantum_state_counts:
                counts = self.current_safety_status.quantum_state_counts
                quantum_stats = html.Div([
                    html.Table([
                        html.Tr([html.Th("State"), html.Th("Count"), html.Th("Percentage")]),
                        *[
                            html.Tr([
                                html.Td(state),
                                html.Td(f"{info['count']}"),
                                html.Td(f"{info['percentage']:.1f}%")
                            ])
                            for state, info in counts.items()
                        ]
                    ], style={'width': '100%', 'textAlign': 'center'})
                ])
            else:
                quantum_stats = html.Div("No quantum state data available")

            return (risk_gauge, status_div, alerts_div,
                   pattern_space, quantum_analysis, metrics_timeline,
                   quantum_transitions, quantum_stats)

    def _generate_empty_plots(self):
        """Generate empty placeholder plots."""
        empty_fig = go.Figure()
        empty_fig.update_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            annotations=[dict(
                text="Awaiting data...",
                xref="paper",
                yref="paper",
                showarrow=False,
                font=dict(size=28)
            )]
        )

        empty_div = html.Div("No data available")

        return (empty_fig, empty_div, empty_div,
                empty_fig, empty_fig, empty_fig,
                empty_fig, empty_div)

    def _create_status_div(self) -> html.Div:
        """Create system status display."""
        return html.Div([
            html.P(f"Safety Score: {self.current_safety_status.safety_score:.2f}"),
            html.P(f"Global Stability: {self.current_network_state.global_stability:.2f}"),
            html.P(f"Active Patterns: {len(self.current_network_state.activation_patterns)}")
        ])

    def _create_alerts_div(self) -> html.Div:
        """Create alerts display."""
        alerts = self.current_safety_status.active_alerts
        if not alerts:
            return html.Div("No active alerts")

        return html.Div([
            html.Div([
                html.Strong(f"{alert.severity.upper()}: "),
                html.Span(alert.description)
            ]) for alert in alerts
        ])

    def update_data(self,
                   network_state: NetworkState,
                   safety_status: SafetyStatus):
        """Update dashboard with new data."""
        self.current_network_state = network_state
        self.current_safety_status = safety_status

        # Update timestamps
        self.timestamps.append(datetime.now())
        if len(self.timestamps) > self.max_history:
            self.timestamps.pop(0)

        # Update metrics
        for pattern in network_state.activation_patterns.values():
            self.metrics_history['stability'].append(pattern.stability)
            self.metrics_history['complexity'].append(pattern.complexity)

        self.metrics_history['risk_level'].append(
            safety_status.overall_risk_level
        )
        self.metrics_history['safety_score'].append(
            safety_status.safety_score
        )

        # Trim history if needed
        for metric in self.metrics_history.values():
            if len(metric) > self.max_history:
                metric.pop(0)

        # Update pattern history
        self.pattern_history.append([
            pattern.activation_pattern
            for pattern in network_state.activation_patterns.values()
        ])
        if len(self.pattern_history) > self.max_history:
            self.pattern_history.pop(0)

    def run_server(self, debug: bool = True, port: int = 8050):
        """Run the dashboard server."""
        self.app.run(debug=debug, port=port, use_reloader=False)